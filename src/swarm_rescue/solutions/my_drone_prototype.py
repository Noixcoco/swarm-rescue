import math
import numpy as np
from enum import Enum
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy import ndimage
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.drone.controller import CommandsDict
import arcade
import heapq
import sys
from pathlib import Path

# Ensure examples can be imported when running from the repository root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from examples.example_mapping import OccupancyGrid
from swarm_rescue.simulation.utils.pose import Pose


class MyDronePrototype(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        EXPLORING = 1
        GOING_TO_WOUNDED = 2
        GOING_TO_RESCUE_CENTER = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.current_pose = np.array([0.0, 0.0, 0.0])

        self.iteration: int = 0
        resolution = 8
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
        
        # --- Sécurité et Inflation ---
        self.robot_radius_pixels = 30
        self.inflation_radius_cells = int(self.robot_radius_pixels / self.grid.resolution)
        if self.inflation_radius_cells < 1:
            self.inflation_radius_cells = 1
        
        # parametre PID rotation
        self.prev_angle_error = 0.0
        self.Kp = 5
        self.Kd = 3

        # PID translation
        self.Kp_pos = 7.0
        self.Kd_pos = 11.0
        self.prev_speed_error = 0.0

        self.path = []
        self.frontiers_world = []

        # NEW: Path smoothing parameters
        self.path_smoothing_enabled = True
        self.path_lookahead_distance = 35.0  # Look ahead for smoother turns

        # `wounded_to_rescue`: list of (x,y) tuples for detected wounded persons
        # `rescue_zone_points`: list of (x,y) tuples representing detected rescue area points
        self.wounded_to_rescue = []
        self.rescue_zone_points = []


        # State machine
        self.state = self.Activity.EXPLORING
        self.current_target_wounded = None
        self.last_replan_iteration = 0  # Track when we last calculated a path
        
        # Kalman filter for GPS position (x, y)
        # State: [x, y, vx, vy] (position and velocity)
        self.kf_state = np.array([0.0, 0.0, 0.0, 0.0])  # Initial state
        # State covariance matrix (uncertainty)
        self.kf_P = np.eye(4) * 100.0  # Initial high uncertainty
        # Process noise covariance (how much we trust the model)
        self.kf_Q = np.eye(4) * 0.1  # Small process noise
        self.kf_Q[2, 2] = 1.0  # Higher uncertainty on velocity
        self.kf_Q[3, 3] = 1.0
        # Measurement noise covariance (GPS noise)
        self.kf_R = np.eye(2) * 25.0  # GPS measurement noise
        # Time step for prediction (will be updated)
        self.kf_dt = 0.1
        self.kf_last_time = 0
        self.kf_initialized = False


        self.evaluated_wounded = set() # pour l'attribution des blessés aux drones
        

        self.wounded_assignments = {}  # {wounded_pos: drone_id}

        self.removed_wounded = []  # Liste des wounded supprimés
        self.removed_wounded_set = set() 
       
        
        # --- NEW: General unstuck mechanism ---
        self.general_stuck_counter = 0
        self.last_unstuck_check_pos = None
        self.unstuck_target = None
        self.is_unstucking = False

        self._last_drone_positions = None
        self._last_drone_danger_zone = None
        self._last_danger_zone_iter = -100

        self.frontier_clusters = [] 

        self.path_cache = {}
        self.path_cache_max_age = 50
        self.path_cache_max_size = 15



    def creer_chemin(self, start_world, goal_world, explored_only=False):
        """
        Calcule un chemin avec LISSAGE pour des mouvements plus linéaires
        """
        #cache
        start_key = (round(start_world[0] / 10) * 10, round(start_world[1] / 10) * 10)
        goal_key = (round(goal_world[0] / 10) * 10, round(goal_world[1] / 10) * 10)
        cache_key = (start_key, goal_key, explored_only)
        
        if cache_key in self.path_cache:
            cached_path, cached_iteration = self.path_cache[cache_key]
            if self.iteration - cached_iteration < self.path_cache_max_age:
                return [np.array(pt) for pt in cached_path]

        grid = self.grid.grid
        # Conversion monde -> grille
        start = self.grid._conv_world_to_grid(*start_world)
        goal = self.grid._conv_world_to_grid(*goal_world)
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))

        # --- FIXED THRESHOLDS ---
        SEUIL_MUR = 30.0
        SEUIL_FREE = -5.0  # Free cells are BELOW this threshold
        SEUIL_UNEXPLORED_MAX = 4.0  # Unexplored cells are near 0 (between -4 and +4)
        SEUIL_UNEXPLORED_MIN = -4.99
    
        # Masque des murs (high positive values)
        is_wall = (grid >= SEUIL_MUR)
        
        # ✅ CORRECT: Only cells with NEGATIVE values are explored free space
        is_explored_free = (grid < SEUIL_FREE)
        
        # ✅ CORRECT: Unexplored cells are near zero
        is_unexplored = (grid >= SEUIL_UNEXPLORED_MIN) & (grid <= SEUIL_UNEXPLORED_MAX)
        
        # Dilate les murs pour éviter les zones proches
        struct = np.ones((6, 6), dtype=bool)
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)
        
        # --- MODIFIED DRONE AVOIDANCE ZONE - ONLY AVOID DRONES IN FRONT ---
        # Cache drone danger zone for a few iterations if positions haven't changed
       
# --- MODIFIED DRONE AVOIDANCE ZONE ---
        # Fix 1: Removed the crash causing comparison (self._last_drone_positions == ...)
        cache_valid = (
            self._last_drone_danger_zone is not None and
            self._last_danger_zone_iter >= self.iteration - 5
        )

        if cache_valid:
            drone_danger_zone = self._last_drone_danger_zone
        else:
            drone_danger_zone = np.zeros_like(grid, dtype=bool)
            if hasattr(self, 'other_drones_positions') and self.other_drones_positions:
                drone_avoidance_radius = 20
                my_pos = np.array(self.current_pose[:2])
                my_heading = float(self.current_pose[2])

                for drone_info in self.other_drones_positions:
                    # Fix 2: Unpack the tuple correctly (Position, ID)
                    other_pos_full = drone_info[0] 
                    other_pos = np.array(other_pos_full[:2])
                    
                    vec_to_drone = other_pos - my_pos
                    dist_to_drone = np.linalg.norm(vec_to_drone)
                    
                    if dist_to_drone < 1e-3 or dist_to_drone > 100.0:
                        continue
                    
                    angle_to_drone = math.atan2(vec_to_drone[1], vec_to_drone[0])
                    relative_angle = normalize_angle(angle_to_drone - my_heading)
                    
                    if abs(relative_angle) < math.pi / 2:
                        # Fix 3: Use correct coordinates (other_pos_full), not the ID!
                        drone_grid_pos = self.grid._conv_world_to_grid(other_pos_full[0], other_pos_full[1])
                        
                        drone_y, drone_x = int(drone_grid_pos[0]), int(drone_grid_pos[1])
                        
                        radius_cells = int(drone_avoidance_radius / self.grid.resolution)
                        y0 = max(0, drone_y - radius_cells)
                        y1 = min(grid.shape[0], drone_y + radius_cells + 1)
                        x0 = max(0, drone_x - radius_cells)
                        x1 = min(grid.shape[1], drone_x + radius_cells + 1)
                        drone_danger_zone[y0:y1, x0:x1] = True
            
            self._last_drone_danger_zone = drone_danger_zone
            self._last_danger_zone_iter = self.iteration


        danger_zone = danger_zone | drone_danger_zone

        # Si explored_only=True, ajouter les zones non explorées à danger_zone
        if explored_only:
            danger_zone = danger_zone | (~is_explored_free)  # Block unexplored cells!
            print(f"[{self.identifier}] explored_only=True: Blocking {np.sum(is_unexplored)} unexplored cells")
        
        # Si le start ou le goal sont dans la danger_zone (par ex. drone collé au mur),
        # on autorise une petite zone autour d'eux pour permettre à A* de s'extraire.
        try:
            radius_clear = 2
            sy, sx = start
            gy, gx = goal
            y0 = max(0, sy - radius_clear)
            y1 = min(grid.shape[0], sy + radius_clear + 1)
            x0 = max(0, sx - radius_clear)
            x1 = min(grid.shape[1], sx + radius_clear + 1)
            danger_zone[y0:y1, x0:x1] = False
            # Clear zone autour du goal avec un rayon plus grand pour le rescue center
            radius_clear_goal = 5  # Plus grand rayon pour le goal (rescue center)
            y0 = max(0, gy - radius_clear_goal)
            y1 = min(grid.shape[0], gy + radius_clear_goal + 1)
            x0 = max(0, gx - radius_clear_goal)
            x1 = min(grid.shape[1], gx + radius_clear_goal + 1)
            danger_zone[y0:y1, x0:x1] = False
        except Exception:
            # en cas de problème d'indices, on ignore et laisse danger_zone inchangé
            pass

        def heuristic(a, b):
            # Euclidean distance as an admissible heuristic for 8-connected grid
            return math.hypot(a[0] - b[0], a[1] - b[1])

        # Allow 8-connected moves (including diagonals)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = [(fscore[start], start)]

        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # --- IMPROVED PATH COMPRESSION WITH ANGLE-BASED SMOOTHING ---
                if len(path) <= 2:
                    compressed = path
                else:
                    compressed = [path[0]]
                    
                    for i in range(1, len(path) - 1):
                        prev_v = np.array([path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]])
                        next_v = np.array([path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]])
                        
                        # Normalize vectors
                        prev_norm = np.linalg.norm(prev_v)
                        next_norm = np.linalg.norm(next_v)
                        
                        if prev_norm > 0 and next_norm > 0:
                            prev_v = prev_v / prev_norm
                            next_v = next_v / next_norm
                            
                            # Calculate angle between vectors
                            dot_product = np.clip(np.dot(prev_v, next_v), -1.0, 1.0)
                            angle_diff = math.acos(dot_product)
                            
                            # Only keep waypoint if angle change is significant (>15 degrees)
                            if angle_diff > math.radians(15):
                                compressed.append(path[i])
                        else:
                            if prev_v.tolist() != next_v.tolist():
                                compressed.append(path[i])
                
                    compressed.append(path[-1])
            
                # --- APPLY SMOOTHING FILTER ---
                if len(compressed) > 2 and self.path_smoothing_enabled:
                    smoothed = self.smooth_path(compressed, danger_zone)
                else:
                    smoothed = compressed
                
                # Convert grid -> world
                world_path = [np.array(self.grid._conv_grid_to_world(*pt)) for pt in smoothed]
                
                self.path_cache[cache_key] = (world_path, self.iteration)
                if len(self.path_cache) > self.path_cache_max_size:
                    oldest_key = min(self.path_cache.keys(), key=lambda k: self.path_cache[k][1])
                    self.path_cache.pop(oldest_key, None)
                return world_path
        

            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    continue

                if danger_zone[neighbor]:
                    continue

                if abs(dx) == 1 and abs(dy) == 1:
                    neigh1 = (current[0] + dx, current[1])
                    neigh2 = (current[0], current[1] + dy)
                    if (0 <= neigh1[0] < grid.shape[0] and 0 <= neigh1[1] < grid.shape[1]):
                        if danger_zone[neigh1]:
                            continue
                    if (0 <= neigh2[0] < grid.shape[0] and 0 <= neigh2[1] < grid.shape[1]):
                        if danger_zone[neigh2]:
                            continue

                move_cost = math.hypot(dx, dy)
                tentative_g_score = gscore[current] + move_cost

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        return []

    def smooth_path(self, path_grid, danger_zone):
        """
        Apply Chaikin's corner-cutting algorithm to smooth the path
        """
        if len(path_grid) <= 2:
            return path_grid
        
        smoothed = [path_grid[0]]
        
        for i in range(len(path_grid) - 1):
            p0 = np.array(path_grid[i], dtype=float)
            p1 = np.array(path_grid[i + 1], dtype=float)
            
            # Chaikin's quarter points
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            
            q_safe = self.is_point_safe(q, danger_zone)
            r_safe = self.is_point_safe(r, danger_zone)
            
            if q_safe and r_safe:
                smoothed.append(tuple(q.astype(int)))
                smoothed.append(tuple(r.astype(int)))
            else:
                smoothed.append(path_grid[i + 1])
        
        # Remove duplicates
        deduplicated = [smoothed[0]]
        for pt in smoothed[1:]:
            if pt != deduplicated[-1]:
                deduplicated.append(pt)
        
        return deduplicated

    def is_point_safe(self, point, danger_zone):
        """Check if a point is in a safe region"""
        y, x = int(round(point[0])), int(round(point[1]))
        
        if not (0 <= y < danger_zone.shape[0] and 0 <= x < danger_zone.shape[1]):
            return False
        
        return not danger_zone[y, x]

    def define_message_for_all(self):
        """Optimized communication - only send essential data at appropriate frequencies"""
    
        # Get positions of currently grasped wounded
        grasped_positions = set(
            (w.position[0], w.position[1]) for w in getattr(self.grasper, "grasped_wounded_persons", []) if hasattr(w, "position")
        )
    
        # Only broadcast wounded not currently grasped
        wounded_list = [
            w for w in self.wounded_to_rescue
            if w not in grasped_positions
        ]
    
        # Base message (sent every iteration)
        message = {
            "drone_id": self.identifier,  
            "drone_pose": self.current_pose.tolist(),
            "wounded_assignments": self.wounded_assignments,
            "grasped_wounded": list(grasped_positions),
        }
    
        # Add wounded list only if changed or every 5 iterations
        if not hasattr(self, '_last_wounded_list') or self._last_wounded_list != wounded_list or self.iteration % 5 == 0:
            message["wounded_list"] = wounded_list
            self._last_wounded_list = wounded_list
    
        # Add rescue list only if changed or every 10 iterations
        if not hasattr(self, '_last_rescue_list') or self._last_rescue_list != self.rescue_zone_points or self.iteration % 10 == 0:
            message["rescue_list"] = self.rescue_zone_points
            self._last_rescue_list = self.rescue_zone_points
    
        # Grid data: only every 20 iterations (was 10)
        if self.iteration % 20 == 0:
            message["grid_data"] = self.grid.grid.copy()
    
        # Removed wounded: only when non-empty
        if self.removed_wounded:
            message["removed_wounded"] = self.removed_wounded
            
    
        # Frontier clusters: send ALL clusters every 10 iterations (no limiting)
        if self.iteration % 10 == 0 and self.frontier_clusters:
            message["frontier_clusters"] = [
                {"barycenter": cluster["barycenter"].tolist()}
                for cluster in self.frontier_clusters
            ]
    
        # Assigned barycenters: only when exploring and target exists
        if self.state == self.Activity.EXPLORING and hasattr(self, "target_point"):
            message["assigned_barycenters"] = {
                str(self.identifier): self.target_point.tolist()
            }
    
        return message

    def control(self) -> CommandsDict:
        """
        Cerveau : Logique de test simplifiée.
        """

        # increment the iteration counter
        self.iteration += 1

        # Process received messages from other drones
        self.process_communication_sensor()

        # --- 1. PERCEPTION ---
        self.update_pose()
        
        # Mise à jour de la grille probabiliste self.grid.grid (utilisée pour l'exploration)
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())
        self.grid.update_grid(pose=self.estimated_pose) # Mise à jour de la carte utilisée!
      

        lidar_data = self.lidar_values()
        if lidar_data is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # Also populate the simpler public lists requested by the user
        try:
            self.detect_semantic_entities()
        except Exception:
            pass

        if self.iteration % 20 == 0:
            self.find_safe_frontier_points()

        # --- NEW: Check for general stuck condition FIRST ---
        if self.check_and_handle_general_stuck():
            # If unstucking, follow the unstuck path
            if self.path:
                command = self.follow_path(lidar_data)
                command["grasper"] = 1
                return command
            else:
                # Unstuck path failed, try simple reverse maneuver
                return {"forward": -0.5, "lateral": 0.3, "rotation": 0.4, "grasper": 1}
               

        # STATE MACHINE LOGIC
        # Transitions
        if self.state == self.Activity.EXPLORING:
            # Only consider wounded not already assigned or grasped
            grasped = getattr(self, "other_grasped_wounded", set())

            exclusion_radius = 50.0

            def is_near_grasped(w):
                return any(math.hypot(w[0] - gx, w[1] - gy) < exclusion_radius for (gx, gy) in grasped)

            available_wounded = [
                w for w in self.wounded_to_rescue
                if w not in self.wounded_assignments and not is_near_grasped(w)
            ]
            
            if available_wounded:
                # Choose closest available wounded
                distances = [np.linalg.norm(np.array(w) - self.current_pose[:2]) for w in available_wounded]
                closest_idx = int(np.argmin(distances))
                closest_wounded = available_wounded[closest_idx]
                my_distance = distances[closest_idx]
                
                # Create a hashable key for this wounded
                wounded_key = (round(closest_wounded[0], 1), round(closest_wounded[1], 1))
                
                # --- FIXED: Only check if this wounded has NEVER been evaluated ---
                if wounded_key not in self.evaluated_wounded:
                    self.evaluated_wounded.add(wounded_key)  # Mark as evaluated IMMEDIATELY
                    
                    # Check if any other drone is closer before assigning
                    should_assign = True
                    for msg in getattr(self.communicator, "received_messages", []):
                        other = msg[1] if isinstance(msg, tuple) else msg
                        other_id = other.get("drone_id")
                        other_pose = np.array(other.get("drone_pose", [None, None, None]))
                        
                        if other_id != self.identifier and other_pose[0] is not None:
                            # Check if other drone is closer to this wounded
                            other_dist = np.linalg.norm(np.array(closest_wounded) - other_pose[:2])
                            
                            # If other drone is significantly closer (with margin), don't assign
                            if other_dist < my_distance - 30.0:
                                should_assign = False
                                break
                            
                            # If distances are similar, use drone ID as tiebreaker (lower ID wins)
                            if abs(other_dist - my_distance) < 30.0 and other_id < self.identifier:
                                should_assign = False
                                break
                    
                    if should_assign:
                        self.current_target_wounded = closest_wounded
                        self.wounded_assignments[self.current_target_wounded] = self.identifier
                        self.state = self.Activity.GOING_TO_WOUNDED
                        self.path = self.creer_chemin(self.current_pose[:2], self.current_target_wounded)
                        self.last_replan_iteration = self.iteration
                        print(f"[{self.identifier}] Assigned to wounded at {self.current_target_wounded}, distance: {my_distance:.1f}")
                    else:
                        print(f"[{self.identifier}] Another drone is closer to wounded at {closest_wounded}")
                # If already evaluated, do nothing (no print, no check)

        elif self.state == self.Activity.GOING_TO_WOUNDED:
            grasped = getattr(self, "other_grasped_wounded", set())
            exclusion_radius = 20.0

            def is_near_grasped(w):
                return any(math.hypot(w[0] - gx, w[1] - gy) < exclusion_radius for (gx, gy) in grasped)

            # --- Check before grasping: if drone is near a grasped wounded, abort and explore ---
            for (gx, gy) in grasped:
                if math.hypot(self.current_pose[0] - gx, self.current_pose[1] - gy) < exclusion_radius:
                    self.state = self.Activity.EXPLORING
                    self.current_target_wounded = None
                    self.path = []
                    

            # --- IMPROVED: Continuous conflict resolution ---
            if self.current_target_wounded is not None:
                # Check every 30 iterations if another drone is now much closer
                if self.iteration % 30 == 0:
                    my_dist = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])
                    should_abandon = False
                    
                    for msg in getattr(self.communicator, "received_messages", []):
                        other = msg[1] if isinstance(msg, tuple) else msg
                        other_id = other.get("drone_id")
                        other_pose = np.array(other.get("drone_pose", [None, None, None]))
                        
                        if other_id != self.identifier and other_pose[0] is not None:
                            other_dist = np.linalg.norm(np.array(self.current_target_wounded) - other_pose[:2])
                            
                            # If another drone is now significantly closer, abandon
                            if other_dist < my_dist - 50.0:  # Larger margin during approach
                                should_abandon = True
                                winner_id = other_id
                                break
                    
                    if should_abandon:
                        print(f"[{self.identifier}] Abandoning target - drone {winner_id} is closer")
                        self.state = self.Activity.EXPLORING
                        self.wounded_assignments.pop(self.current_target_wounded, None)
                        self.current_target_wounded = None
                        self.path = []
                        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1}
                    
            if self.grasper.grasped_wounded_persons:
        
                self.state = self.Activity.GOING_TO_RESCUE_CENTER
                # Successfully grasped, go to rescue center
                if self.current_target_wounded is not None:
                    
                    self.removed_wounded.append(self.current_target_wounded)

                    # Immediate local cleanup
                    self.wounded_to_rescue = [
                        w for w in self.wounded_to_rescue 
                        if math.hypot(w[0] - self.current_target_wounded[0], 
                                    w[1] - self.current_target_wounded[1]) > 50.0
                    ]

            
                    print(f"[{self.identifier}] STEP 2: Removed wounded at {self.current_target_wounded}")


                    

                if self.rescue_zone_points:
                    self.path = self.creer_chemin(self.current_pose[:2], self.rescue_zone_points[0], explored_only=True)
                    self.last_replan_iteration = self.iteration

            elif self.current_target_wounded is not None:
                distance_to_target = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])
                
             

                # Check if wounded is currently detected by semantic sensor
                wounded_detected = False
                detection_radius = 50.0
                
                if distance_to_target < 30.0:
                    try:
                        detections = self.semantic_values()
                        if detections:
                            px = float(self.current_pose[0])
                            py = float(self.current_pose[1])
                            ptheta = float(self.current_pose[2])
                            
                            for data in detections:
                                try:
                                    etype = getattr(data, 'entity_type', None)
                                    name = etype.name if hasattr(etype, 'name') else str(etype)
                                    
                                    if 'WOUNDED' in name.upper():
                                        angle = float(getattr(data, 'angle', 0.0))
                                        dist = float(getattr(data, 'distance', 0.0))
                                        
                                        # Convert to world coordinates
                                        global_angle = normalize_angle(ptheta + angle)
                                        xw = px + dist * math.cos(global_angle)
                                        yw = py + dist * math.sin(global_angle)
                                        
                                        dist_to_target = math.hypot(self.current_target_wounded[0] - xw, 
                                                                   self.current_target_wounded[1] - yw)
                                        
                                        if dist_to_target < detection_radius:
                                            wounded_detected = True
                                            break
                                except Exception as e:
                                    print(f"  Error processing detection: {e}")
                                    continue
                        else:
                            print("No semantic detections available")
                    except Exception as e:
                        print(f"Error reading semantic sensor: {e}")
                    
                    
                  
                  
                    if not wounded_detected:
                        print(f"\n*** WOUNDED NOT FOUND - REMOVING FROM LIST ***")
                        check_radius = 50.0
                        
                        count_before = len(self.wounded_to_rescue)
                        print(f"Wounded list before removal: {self.wounded_to_rescue}")
                        
                        # Remove wounded persons close to the target location
                        self.wounded_to_rescue = [
                            (wx, wy) for (wx, wy) in self.wounded_to_rescue
                            if math.hypot(self.current_target_wounded[0] - wx, 
                                        self.current_target_wounded[1] - wy) > check_radius
                        ]
                        
                        count_after = len(self.wounded_to_rescue)
                        print(f"Removed {count_before - count_after} wounded from list")
                        print(f"Wounded list after removal: {self.wounded_to_rescue}")
                        self.removed_wounded.append(self.current_target_wounded)
                        

                        
                        # Return to exploring
                        self.state = self.Activity.EXPLORING
                        self.current_target_wounded = None
                        self.path = []
                        print(f"Switched to EXPLORING state\n")


            else:
                # No target defined, return to exploring
                self.state = self.Activity.EXPLORING
                self.current_target_wounded = None

        elif self.state == self.Activity.GOING_TO_RESCUE_CENTER:

            if not self.grasper.grasped_wounded_persons:
                # Dropped wounded, return to exploring
                if self.current_target_wounded is not None:
                    # Remove assignment so other drones don't try to grab it
                    self.wounded_assignments.pop(self.current_target_wounded, None)
                self.state = self.Activity.EXPLORING
                self.current_target_wounded = None
                
            else:
                # --- ENSURE SAFE RETURN: Only use explored areas ---
                if self.rescue_zone_points:
                   
                    
                    # Replan with explored_only=True for safe return
                    should_replan = False
                    if not self.path or len(self.path) == 0:
                        iterations_since_replan = self.iteration - self.last_replan_iteration
                        if iterations_since_replan >= 30 or self.last_replan_iteration == 0:
                            should_replan = True
                    
                    if should_replan:
                        # KEY CHANGE: Force explored_only=True when going to rescue center
                        self.path = self.creer_chemin(
                            self.current_pose[:2], 
                            self.rescue_zone_points[0], 
                            explored_only=True  # Only use explored safe areas
                        )
                        self.last_replan_iteration = self.iteration
                        
                        # If no safe path found through explored areas, try without restriction
                        if not self.path:
                            print(f"[{self.identifier}] No safe explored path to rescue center, using full map")
                            self.path = self.creer_chemin(
                                self.current_pose[:2], 
                                self.rescue_zone_points[0], 
                                explored_only=False
                            )

        # --- 2. STRATÉGIE ---
        # Replanification for exploration (only when in EXPLORING state)
        if self.state == self.Activity.EXPLORING:
            need_replan = False
            if not self.path or len(self.path) < 1:
                need_replan = True
            # Met à jour le chemin tous les 50 itérations (si cible existante)
            if self.path and hasattr(self, 'target_point') and self.iteration % 50 == 0:
                need_replan = True

            if need_replan:
                # Use shared frontier clusters from communication
                shared_clusters = getattr(self, "shared_frontier_barycenters", [])
                
                if shared_clusters:
                    barycenters = [np.array(bc) for bc in shared_clusters]
                    
                    # Get assignments from other drones
                    assigned_targets = {}
                    for msg in getattr(self.communicator, "received_messages", []):
                        other = msg[1] if isinstance(msg, tuple) else msg
                        other_id = other.get("drone_id")
                        other_assignments = other.get("assigned_barycenters", {})
                        
                        if other_id != self.identifier:
                            for drone_id_str, target in other_assignments.items():
                                assigned_targets[int(drone_id_str)] = np.array(target)
    
                  
                    best_score = float('inf')
                    best_target = None
                    
                    min_separation = 300.0  # Minimum distance between drone targets
                    
                    for bc in barycenters:
                      
                        distance = np.linalg.norm(bc - self.current_pose[:2])
                        # 2. Conflict penalty (avoid targets near other drones' assignments)
                        conflict_penalty = 0.0
                        for other_target in assigned_targets.values():
                            dist_to_assigned = np.linalg.norm(bc - other_target)
                                
                            if dist_to_assigned < min_separation:
                                conflict_penalty += 10000.0  # Heavy penalty
            

                        cluster_size = 10  # Default if size unknown
                        for cluster in self.frontier_clusters:
                            if np.linalg.norm(cluster["barycenter"] - bc) < 20:
                                cluster_size = cluster["size"]
                                break

                        size_bonus = cluster_size * 50.0 

                        # 3. Drone proximity penalty (avoid crowded areas)
                        drone_penalty = 0.0
                        if hasattr(self, 'other_drones_positions') and self.other_drones_positions:
                            for drone_pos in self.other_drones_positions:
                                dist_drone_to_frontier = np.linalg.norm(bc - np.array(drone_pos[0][:2]))
                                if dist_drone_to_frontier < 200.0:
                                    drone_penalty += 300.0 / (dist_drone_to_frontier + 1.0)
                        
                      
        
                        # Combined score (lower is better)
                        score = distance + size_bonus - conflict_penalty - drone_penalty+size_bonus
            
                        if score > best_score:
                            best_score = score
                            best_target = bc

                    if best_target is not None:
                        self.target_point = best_target
                        self.path = self.creer_chemin(self.current_pose[:2], best_target)
                    
                        
                    else:
                        # Fallback to first barycenter
                        if barycenters:
                            self.target_point = barycenters[0]
                            self.path = self.creer_chemin(self.current_pose[:2], self.target_point)
                
                else:
                    # ✅ FALLBACK: Use local frontier detection
                    self.frontiers_world = self.find_safe_frontier_points()
                    if self.frontiers_world:
                        distances = [np.linalg.norm(f - self.current_pose[:2]) for f in self.frontiers_world]
                        target_index = np.argmin(distances)
                        target_point = self.frontiers_world[target_index]
                        self.target_point = target_point
                        self.path = self.creer_chemin(self.current_pose[:2], target_point)
                       

        # Generate movement commands based on current state
        if self.state == self.Activity.EXPLORING:
            if self.path:
                command = self.follow_path(lidar_data)
            else:
                command = {"forward": 0.5, "lateral": 0.0, "rotation": 0.0}

        elif self.state == self.Activity.GOING_TO_WOUNDED:

            #rotate to face wounded when close enough
            if self.current_target_wounded:
                dist_to_target = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])
                
                if dist_to_target < 100.0:
                    # Calculate angle directly to the person
                    diff = np.array(self.current_target_wounded) - self.current_pose[:2]
                    target_angle = math.atan2(diff[1], diff[0])
                    
                    # # Rotate to face the person before the grasper 'clicks'
                    # angle_error = normalize_angle(target_angle - self.current_pose[2])
                    
                    # We want to approach backwards, so we aim for the opposite angle
                    target_heading = normalize_angle(target_angle + math.pi)
                    
                    # Calculate angle error relative to the BACK of the drone
                    angle_error = normalize_angle(target_heading - self.current_pose[2])
                    
                    #on veut prendre le blessé par l'arrière donc on fait demi-tour
                    #on gère le fait où l'angle est proche de pi ou -pi
                    if angle_error > np.pi:
                        angle_error -= 2.0 * np.pi
                    elif angle_error < -np.pi:
                        angle_error += 2.0 * np.pi

                    rotation_speed = float(np.clip(self.Kp * angle_error, -1.0, 1.0))
                    


                    # Slow approach to ensure the front-mounted grasper makes contact
                    # Logic: Stop and rotate if not aligned, then back up
                    if abs(angle_error) > 0.2:
                        # Stop and rotate
                        command = {"forward": 0.0, "lateral": 0.0, "rotation": rotation_speed, "grasper": 0}
                    else:
                        # Aligned (back facing target), move backwards
                        command = {"forward": -0.3, "lateral": 0.0, "rotation": rotation_speed, "grasper": 1}
                    return self.wall_avoidance(command, lidar_data)
            
    
            if self.path:
                command = self.follow_path(lidar_data)
            

            else:
                # Replan if needed
                if self.current_target_wounded:
                    should_replan = False
                    if not self.path or len(self.path) == 0:
                        iterations_since_replan = self.iteration - self.last_replan_iteration
                        if iterations_since_replan >= 20 or self.last_replan_iteration == 0:
                            should_replan = True
                    
                    if should_replan:
                        self.path = self.creer_chemin(self.current_pose[:2], self.current_target_wounded)
                        self.last_replan_iteration = self.iteration
                    
                    if self.path:
                        command = self.follow_path(lidar_data)
                    else:
                        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
                else:
                    command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        elif self.state == self.Activity.GOING_TO_RESCUE_CENTER:
            dist_to_rescue = float('inf')
            if self.rescue_zone_points:
                dist_to_rescue = np.linalg.norm(np.array(self.rescue_zone_points[0]) - self.current_pose[:2])
            
            if dist_to_rescue < 100.0 and self.rescue_zone_points:
                 target_pos = np.array(self.rescue_zone_points[0])
                 diff = target_pos - self.current_pose[:2]
                 target_angle = math.atan2(diff[1], diff[0])
                 
                 target_heading = normalize_angle(target_angle + math.pi)
                 
                 angle_error = normalize_angle(target_heading - self.current_pose[2])
                 rotation_speed = float(np.clip(self.Kp * angle_error, -1.0, 1.0))
                 
                 if abs(angle_error) > 0.2:
                     command = {"forward": 0.0, "lateral": 0.0, "rotation": rotation_speed}
                 else:
                     command = {"forward": -0.5, "lateral": 0.0, "rotation": rotation_speed}
            elif self.path:
                command = self.follow_path(lidar_data)
            else:
                command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        else:
            command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        # NOW set grasper
        command["grasper"] = 1

        # --- NEW: Smart grasper control based on proximity to other drones ---
        # Default: grasper always active
        grasper_command = 1
        
        # Check if currently holding a wounded person
        has_wounded = bool(self.grasper.grasped_wounded_persons)
        
        # If not holding wounded, check proximity to other drones
        if not has_wounded and hasattr(self, 'other_drones_positions') and self.other_drones_positions:
            proximity_threshold = 80.0  # Distance threshold for deactivating grasper
            
            for drone_pos in self.other_drones_positions:
                drone_pos = drone_pos[0]
                distance_to_drone = math.hypot(
                    self.current_pose[0] - drone_pos[0],
                    self.current_pose[1] - drone_pos[1]
                )
                
                if distance_to_drone < proximity_threshold:
                    grasper_command = 0  # Deactivate grasper when near another drone
                    break
        
        command["grasper"] = grasper_command
        # --- END SMART GRASPER CONTROL ---

        # Dynamic replanning if other drones are too close to current path
        if self.path and hasattr(self, 'other_drones_positions') and self.other_drones_positions:
            replan_needed = False
            for drone_pos in self.other_drones_positions:
                drone_pos = drone_pos[0]
                for waypoint in self.path[:min(3, len(self.path))]:  # Check first 3 waypoints
                    dist_to_waypoint = math.hypot(waypoint[0] - drone_pos[0], waypoint[1] - drone_pos[1])
                    if dist_to_waypoint < 60.0:  # Threshold for replanning
                        replan_needed = True
                        break
                if replan_needed:
                    break
            
            if replan_needed and (self.iteration - self.last_replan_iteration) > 10:
                # Replan path avoiding the drone
                if self.state == self.Activity.GOING_TO_WOUNDED and self.current_target_wounded:
                    self.path = self.creer_chemin(self.current_pose[:2], self.current_target_wounded)
                    self.last_replan_iteration = self.iteration
                elif self.state == self.Activity.GOING_TO_RESCUE_CENTER and self.rescue_zone_points:
                    self.path = self.creer_chemin(self.current_pose[:2], self.rescue_zone_points[0], explored_only=True)
                    self.last_replan_iteration = self.iteration

        command = self.rvo_avoidance(command)

        # Apply this LAST to prevent hitting walls while dodging drones
        command = self.wall_avoidance(command, lidar_data)

        if self.iteration % 50 == 0:
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")

        return command

    def detect_semantic_entities(self):
        """Optimized semantic entity detection"""
        try:
            detections = self.semantic_values()
        except Exception:
            detections = None

      
        if not detections:
            return

        # Parameters
        dedup_radius = 60.0
        alpha_update = 0.3
 

    

        newly_seen_wounded = []
        newly_seen_rescue = []

        px = float(self.current_pose[0])
        py = float(self.current_pose[1])
        ptheta = float(self.current_pose[2])

    
        if not hasattr(self, 'other_grasped_wounded'):
            self.other_grasped_wounded = set()

        for data in detections:
            try:
                etype = getattr(data, 'entity_type', None)
                angle = float(getattr(data, 'angle', 0.0))
                dist = float(getattr(data, 'distance', 0.0))
            except Exception:
                continue

            global_angle = normalize_angle(ptheta + angle)
            xw = px + dist * math.cos(global_angle)
            yw = py + dist * math.sin(global_angle)

            try:
                name = etype.name if hasattr(etype, 'name') else str(etype)
            except Exception:
                name = str(etype)

            if 'WOUNDED' in name.upper():
                # ✅ Now fast - other_grasped_wounded already initialized
                is_grasped = any(
                    math.hypot(xw - gx, yw - gy) < dedup_radius
                    for (gx, gy) in self.other_grasped_wounded
                )
                if not is_grasped:
                    newly_seen_wounded.append((xw, yw))
                
            elif 'RESCUE' in name.upper():
                newly_seen_rescue.append((xw, yw))


        # Merge newly seen wounded
        for nx, ny in newly_seen_wounded:
            merged = False
            for i, (wx, wy) in enumerate(self.wounded_to_rescue):
                if math.hypot(wx - nx, wy - ny) < dedup_radius:
                    # weighted update
                    newx = (1.0 - alpha_update) * wx + alpha_update * nx
                    newy = (1.0 - alpha_update) * wy + alpha_update * ny
                    self.wounded_to_rescue[i] = (newx, newy)
                    merged = True
                    break
            if not merged:
                pt = (nx, ny)
                self.wounded_to_rescue.append(pt)



# --- MODIFIED: Keep multiple rescue zone points instead of merging ---
        rescue_dedup_radius = 150.0  # Larger radius to identify distinct entry points
        
        for nx, ny in newly_seen_rescue:
            merged = False
            for i, (rx, ry) in enumerate(self.rescue_zone_points):
                if math.hypot(rx - nx, ry - ny) < rescue_dedup_radius:
                    newx = (1.0 - alpha_update) * rx + alpha_update * nx
                    newy = (1.0 - alpha_update) * ry + alpha_update * ny
                    self.rescue_zone_points[i] = (newx, newy)
                    merged = True
                    break
            if not merged:
                pt = (nx, ny)
                self.rescue_zone_points.append(pt)

 
   



    # --------------------------------------------------------------------------
    # FONCTION DE DÉTECTION DES FRONTIÈRES SÛRES (Mise à jour pour self.frontiers_world)
    # --------------------------------------------------------------------------

    def find_safe_frontier_points(self) -> list:
        
        grid_map = self.grid.grid 
        
        # ✅ RELAXED THRESHOLDS - Encourage exploring unexplored areas
        SEUIL_FREE = -2.0        # Lightly explored (was -7.0)
        SEUIL_MUR = 3.0         
        SEUIL_UNEXPLORED_MIN = -1.0  # Wider unexplored range
        SEUIL_UNEXPLORED_MAX = 1.0
    
        frontiers = []

        # Masks
        is_unknown = (grid_map >= SEUIL_UNEXPLORED_MIN) & (grid_map <= SEUIL_UNEXPLORED_MAX)
        is_wall = (grid_map >= SEUIL_MUR)  
        is_free = (grid_map < SEUIL_FREE) # Lightly explored areas
        
        # ✅ KEY: Exclude heavily explored dark blue corridor
        is_heavily_explored = (grid_map < -15.0)

        # Frontier detection
        structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=bool)

        unknown_neighbors = binary_dilation(is_unknown, structure=structure)
        
        # ✅ FIXED: Find free cells near unexplored, but NOT in heavily explored corridor
        frontier_mask = is_free & (~is_heavily_explored) & unknown_neighbors

        # Safety margin around walls
        struct = np.ones((6, 6), dtype=bool)
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=2)
        frontier_mask = frontier_mask & (~danger_zone)

        # Clustering
        structure = generate_binary_structure(2, 2)
        labeled, num_features = ndimage.label(frontier_mask, structure=structure)

        self.frontier_clusters = []
        min_cluster_size = 3

        for label_idx in range(1, num_features + 1):
            ys, xs = np.where(labeled == label_idx)
            size = ys.size
            if size < min_cluster_size:
                continue

            mean_x = float(np.mean(ys))
            mean_y = float(np.mean(xs))
            x_world, y_world = self.grid._conv_grid_to_world(mean_x, mean_y)
            barycenter = np.array([x_world, y_world])
            
            # ✅ VALIDATION: Ensure nearby unexplored cells exist
            bc_grid = self.grid._conv_world_to_grid(x_world, y_world)
            bc_y, bc_x = int(bc_grid[0]), int(bc_grid[1])
            
            window = 15
            y0, y1 = max(0, bc_y - window), min(grid_map.shape[0], bc_y + window)
            x0, x1 = max(0, bc_x - window), min(grid_map.shape[1], bc_x + window)
            neighborhood = grid_map[y0:y1, x0:x1]
            
            # Count unexplored cells nearby
            unexplored_nearby = np.sum((neighborhood >= SEUIL_UNEXPLORED_MIN) & 
                                       (neighborhood <= SEUIL_UNEXPLORED_MAX))
            
            if unexplored_nearby < 10:  # Require some unexplored cells
                continue
            
            # ✅ REJECT dark blue corridor: if 70%+ heavily explored, skip
            heavily_explored_nearby = np.sum(neighborhood < -15.0)
            if heavily_explored_nearby > 0.7 * neighborhood.size:
                continue
            
            self.frontier_clusters.append({
                "barycenter": barycenter,
                "size": int(size)
            })

        self.frontier_clusters.sort(key=lambda c: c["size"], reverse=True)
        frontiers = [c["barycenter"] for c in self.frontier_clusters]

        return frontiers

    # FONCTION DE DESSIN
    # --------------------------------------------------------------------------
    
    def draw_bottom_layer(self):
        """ Dessine le chemin calculé (tous les points) """

        # Define a palette of colors (extend as needed)
        palette = [
            (200, 60, 60),   # Red
            (60, 200, 60),   # Green
            (60, 60, 200),   # Blue
            (200, 200, 60),  # Yellow
            (200, 60, 200),  # Magenta
            (60, 200, 200),  # Cyan
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
        ]
        # Assign a color based on the drone's identifier (hash to index)
        color_idx = int(self.identifier) % len(palette)
        detection_color = palette[color_idx]

        if hasattr(self, 'frontier_clusters') and self.frontier_clusters :
            # Only draw the 5 closest clusters to reduce rendering overhead
            if len(self.frontier_clusters) > 5:
                # Sort by distance to current position
                sorted_clusters = sorted(
                    self.frontier_clusters,
                    key=lambda c: np.linalg.norm(c['barycenter'] - self.current_pose[:2])
                )[:5]  # Take only 5 closest
            else:
                sorted_clusters = self.frontier_clusters
            
            for cluster in sorted_clusters:
                bc = cluster.get('barycenter')
                if bc is not None:
                    ptb = bc + self._half_size_array
                    arcade.draw_circle_filled(ptb[0], ptb[1], radius=6, color=detection_color)

        # Draw wounded detected via new simple API (wounded_to_rescue)
        try:
            if hasattr(self, 'wounded_to_rescue') and self.wounded_to_rescue:
                for (xw, yw) in self.wounded_to_rescue:
                    pt = np.array([xw, yw]) + self._half_size_array
                    # visible marker (outline + small filled center)
                    #arcade.draw_circle_outline(pt[0], pt[1], radius=18, color=detection_color, border_width=2)
                    #arcade.draw_text("det", pt[0] + 14, pt[1] + 14, detection_color, 10)


                   # Match the wounded to an assignment using a distance threshold
                    assigned_drone_id = None
                    
                    # Iterate through assignments to find a match for this (xw, yw)
                    for w_pos, drone_id in self.wounded_assignments.items():
                        # Handle potential string keys from communication
                        if isinstance(w_pos, str):
                            try:
                                # Convert "(1.2, 3.4)" -> [1.2, 3.4]
                                coords = [float(x) for x in w_pos.strip("()").split(",")]
                                kx, ky = coords[0], coords[1]
                            except: continue
                        else:
                            kx, ky = w_pos[0], w_pos[1]
                        
                        # Distance threshold check (must be the same person)
                        if math.hypot(xw - kx, yw - ky) < 10.0:
                            assigned_drone_id = drone_id
                            break
                    
                    # Set color based on assigned drone
                    if assigned_drone_id is not None:
                        color = palette[int(assigned_drone_id) % len(palette)]
                        label = f"ASSIGNED: DRONE {assigned_drone_id}"
                    else:
                        color = (255, 255, 255) # White if unassigned
                        label = "AVAILABLE"

                    # Draw the marker and the text
                    arcade.draw_circle_outline(pt[0], pt[1], 20, color, 2)
                    arcade.draw_text(label, pt[0] + 25, pt[1] - 10, color, 11, bold=True)
        except Exception:
            pass

        # Draw additional rescue zone points detected via new API
        try:
            if hasattr(self, 'rescue_zone_points') and self.rescue_zone_points:
                for (xr, yr) in self.rescue_zone_points:
                    pt = np.array([xr, yr]) + self._half_size_array
                    arcade.draw_rectangle_outline(pt[0], pt[1], width=30, height=30, color=(0,160,0), border_width=2)
        except Exception:
            pass

        if self.path and len(self.path) > 0:
            radius = 7
            blue = (0,0,255)
            green = (0,255,0)
            # Affiche chaque point du chemin
            for pt in self.path:
                point_arcade = pt + self._half_size_array
                arcade.draw_circle_filled(point_arcade[0], point_arcade[1], radius=radius, color=blue)
            # Relie les points par des segments
            for i in range(len(self.path)-1):
                p1 = self.path[i] + self._half_size_array
                p2 = self.path[i+1] + self._half_size_array
                arcade.draw_line(p1[0], p1[1], p2[0], p2[1], color=green, line_width=3)

    
        
        try:
            current_pose_screen = self.current_pose[:2] + self._half_size_array
            # Get state name
            state_name = self.state.name if hasattr(self.state, 'name') else str(self.state)
            # Display above drone (offset +25 pixels above)
            arcade.draw_text(state_name, 
                           current_pose_screen[0] - 30, 
                           current_pose_screen[1] + 25, 
                           (255, 255, 255), 
                           12, 
                           bold=True)
        except Exception:
            pass
            


    # --------------------------------------------------------------------------
    # FONCTIONS DE PILOTAGE
    # --------------------------------------------------------------------------
 
    def follow_path(self, lidar_data) -> CommandsDict:
        """
        IMPROVED path following with lookahead for smoother motion
        NO adaptive speed control - constant speed profile
        """
        if not self.path:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        
        # --- LOOKAHEAD MECHANISM FOR SMOOTHER TURNS ---
        lookahead_target = self.path[0]
        accumulated_distance = 0.0
        
        for i in range(len(self.path)):
            if i > 0:
                segment_length = np.linalg.norm(self.path[i] - self.path[i-1])
                accumulated_distance += segment_length
        
            if accumulated_distance >= self.path_lookahead_distance:
                lookahead_target = self.path[i]
                break
            elif i == len(self.path) - 1:
                lookahead_target = self.path[i]
        
        # Calculate target angle to lookahead point
        delta_pos = lookahead_target - self.current_pose[:2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])
    
        # --- PID sur la rotation avec DAMPING amélioré ---
        angle_error = normalize_angle(target_angle - self.current_pose[2])
        deriv_error = angle_error - self.prev_angle_error
    
        # Reduce rotation gain when angle error is small (smoother motion)
        if abs(angle_error) < math.radians(10):
            Kp_active = self.Kp * 0.6
            Kd_active = self.Kd * 0.8
        else:
            Kp_active = self.Kp
            Kd_active = self.Kd
    
        rotation_speed = Kp_active * angle_error + Kd_active * deriv_error
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error
    
        # Distance to immediate waypoint (for waypoint removal)
        distance_to_waypoint = np.linalg.norm(self.path[0] - self.current_pose[:2])
    
        # --- CONSTANT SPEED PROFILE (NO ADAPTIVE CONTROL) ---
        max_speed = 100.0
        target_speed = max(0.0, min(max_speed, distance_to_waypoint * 0.12 + 0.3))
    
        measured_vel = self.measured_velocity()
        measured_speed = math.sqrt(measured_vel[0] ** 2 + measured_vel[1] ** 2)
    
        # --- PID sur la vitesse ---
        speed_error = target_speed - measured_speed
        deriv_speed = speed_error - self.prev_speed_error
        forward_cmd = self.Kp_pos * speed_error + self.Kd_pos * deriv_speed
    
        # Slow down if large angle error
        if abs(angle_error) > 1.0:  # Changed from 0.4 to 1.0 radians
            forward_cmd *= 0.3  # Reduce speed instead of stopping
    
        forward_cmd = float(np.clip(forward_cmd, -1.0, 1.0))
        self.prev_speed_error = speed_error
    
        # Remove waypoint when close
        if distance_to_waypoint < 30.0:
            self.path.pop(0)
        elif len(self.path) > 1:
            dist_to_next = np.linalg.norm(self.path[1] - self.current_pose[:2])
            if dist_to_next < distance_to_waypoint:
                self.path.pop(0)
        return {"forward": forward_cmd, "lateral": 0.0, "rotation": rotation_speed}
    

    # --------------------------------------------------------------------------
    # FONCTIONS PRINCIPALES (Localisation, Cartographie Binaire)
    # --------------------------------------------------------------------------

    def update_pose(self):
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()

        # Calculate dt for Kalman filter
        current_time = self.iteration * 0.1  # Assuming 10 Hz
        if self.kf_last_time > 0:
            self.kf_dt = current_time - self.kf_last_time
        self.kf_last_time = current_time
        
        if not np.isnan(gps_pos[0]):
            # GPS available - use Kalman filter
            
            # Initialize filter on first GPS measurement
            if not self.kf_initialized:
                self.kf_state[0] = gps_pos[0]
                self.kf_state[1] = gps_pos[1]
                self.kf_state[2] = 0.0  # Initial velocity
                self.kf_state[3] = 0.0
                self.kf_initialized = True
            
            # Kalman Filter Prediction Step
            # State transition matrix F (constant velocity model)
            F = np.array([
                [1, 0, self.kf_dt, 0],
                [0, 1, 0, self.kf_dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            # Predict state
            self.kf_state = F @ self.kf_state
            
            # Predict covariance
            self.kf_P = F @ self.kf_P @ F.T + self.kf_Q
            
            # Kalman Filter Update Step
            # Measurement matrix H (we only measure position, not velocity)
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            
            # Measurement residual
            z = np.array([gps_pos[0], gps_pos[1]])
            y = z - H @ self.kf_state
            
            # Residual covariance
            S = H @ self.kf_P @ H.T + self.kf_R
            
            # Kalman gain
            K = self.kf_P @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.kf_state = self.kf_state + K @ y
            
            # Update covariance
            I = np.eye(4)
            self.kf_P = (I - K @ H) @ self.kf_P
            
            # Use filtered position
            self.current_pose[0] = self.kf_state[0]
            self.current_pose[1] = self.kf_state[1]
            self.current_pose[2] = compass_angle
        else:
            # GPS unavailable - use odometry with Kalman prediction
            odom_data = self.odometer_values() 
            if odom_data is None: return
            dist_traveled = odom_data[0]
            rotation_change = odom_data[2]
            
            self.current_pose[2] += rotation_change
            self.current_pose[2] = normalize_angle(self.current_pose[2])
            
            if self.kf_initialized:
                # Use Kalman predicted position when GPS unavailable
                F = np.array([
                    [1, 0, self.kf_dt, 0],
                    [0, 1, 0, self.kf_dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                self.kf_state = F @ self.kf_state
                self.kf_P = F @ self.kf_P @ F.T + self.kf_Q
                
                self.current_pose[0] = self.kf_state[0]
                self.current_pose[1] = self.kf_state[1]
            else:
                # Fallback to odometry if Kalman not yet initialized
                dx = dist_traveled * math.cos(self.current_pose[2])
                dy = dist_traveled * math.sin(self.current_pose[2])
                self.current_pose[0] += dx
                self.current_pose[1] += dy

    def process_communication_sensor(self):
        """
        Optimized message processing with reduced overhead
        """
        if not self.communicator:
            return

        dedup_radius = 50.0
        received_messages = self.communicator.received_messages

        # Pre-allocate collections
        all_wounded = []
        all_assignments = {}
        all_grasped = set()
        all_rescue_zones = []
        all_frontier_clusters = []
        other_drones_positions = []

        # Single pass through messages
        for msg in received_messages:
            other_message = msg[1] if isinstance(msg, tuple) else msg
            other_id = other_message.get("drone_id")
            
            # Skip own messages
            if other_id == self.identifier:
                continue
            
            # Drone positions (always needed for collision avoidance)
            pos = other_message.get("drone_pose")
            if pos is not None:
                # Store as tuple: (position_array, id)
                other_drones_positions.append((np.array(pos), other_id))
            
            # Wounded list (only if present in message)
            if "wounded_list" in other_message:
                all_wounded.extend(other_message["wounded_list"])
        
            # Assignments (always present)
            all_assignments.update(other_message.get("wounded_assignments", {}))
        
            # Grasped wounded (always present)
            all_grasped.update(
                tuple(w) for w in other_message.get("grasped_wounded", []) if w is not None
            )
            
            # Rescue zones (only if present)
            if "rescue_list" in other_message:
                all_rescue_zones.extend(other_message["rescue_list"])
        

            if "removed_wounded" in other_message:
                for rw_new in other_message["removed_wounded"]:
                    # Round coordinates for consistent hashing
                    key = (round(rw_new[0] / dedup_radius) * dedup_radius, 
                        round(rw_new[1] / dedup_radius) * dedup_radius)
                    
                    if key not in self.removed_wounded_set:
                        self.removed_wounded.append(rw_new)
                        self.removed_wounded_set.add(key)
                           
        
            # Frontier clusters (only if present)
            if "frontier_clusters" in other_message:
                all_frontier_clusters.extend(other_message["frontier_clusters"])
        
            # Grid fusion (only if present and not too often)
            if self.iteration % 20 == 0 and "grid_data" in other_message and other_message["grid_data"] is not None:
                other_grid = np.array(other_message["grid_data"])
                # Simple weighted average (favor own observations)
                self.grid.grid = 0.7 * self.grid.grid + 0.3 * other_grid
    
        # Store drone positions immediately (needed for avoidance)
        self.other_drones_positions = other_drones_positions
    
        # Add own frontier clusters for deduplication
        all_frontier_clusters.extend([
            {"barycenter": cluster["barycenter"].tolist()}
            for cluster in getattr(self, "frontier_clusters", [])
        ])
    
        # --- WOUNDED MANAGEMENT ---
        # Merge new wounded (vectorized distance check would be faster but this is clearer)
        merged_wounded = list(self.wounded_to_rescue)
        for w in all_wounded:
            if all(math.hypot(w[0] - wx, w[1] - wy) > dedup_radius for (wx, wy) in merged_wounded):
                merged_wounded.append(tuple(w))
    
        # Remove explicitly removed wounded
        
        for rw in self.removed_wounded:
            merged_wounded = [
                (wx, wy) for (wx, wy) in merged_wounded
                if math.hypot(rw[0] - wx, rw[1] - wy) > dedup_radius
                ]
    
        # Final deduplication
        deduped_wounded = []
        for w in merged_wounded:
            if all(math.hypot(w[0] - wx, w[1] - wy) > dedup_radius for (wx, wy) in deduped_wounded):
                deduped_wounded.append(w)
    
        self.wounded_to_rescue = deduped_wounded
    
        # --- ASSIGNMENTS ---
        for w, drone_id in all_assignments.items():
            if w not in self.wounded_assignments:
                self.wounded_assignments[w] = drone_id
    
        # --- GRASPED WOUNDED ---
        if not hasattr(self, "other_grasped_wounded"):
            self.other_grasped_wounded = set()
        self.other_grasped_wounded = all_grasped
    
        # --- RESCUE ZONES ---
        for r in all_rescue_zones:
            if r not in self.rescue_zone_points:
                self.rescue_zone_points.append(r)
    
        # --- FRONTIER DEDUPLICATION (ALL clusters, no limiting) ---
        if all_frontier_clusters:
            deduped_barycenters = []
            dedup_radius_frontier = 100.0
            for cl in all_frontier_clusters:
                bc = np.array(cl["barycenter"])
                if all(np.linalg.norm(bc - np.array(b)) > dedup_radius_frontier for b in deduped_barycenters):
                    deduped_barycenters.append(bc.tolist())
            self.shared_frontier_barycenters = deduped_barycenters

        self.wounded_to_rescue = merged_wounded

    def find_free_position_for_unstuck(self):
        """
        Find the first safe free position at a medium distance to escape when stuck.
        Returns a position (x, y) in world coordinates, or None if not found.
        """
        grid_map = self.grid.grid
        SEUIL_FREE = -10.0
        SEUIL_MUR = 5.0

        is_free = (grid_map < SEUIL_FREE)
        is_wall = (grid_map >= SEUIL_MUR)
        struct = np.ones((6, 6), dtype=bool)
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)
        safe_free = is_free & (~danger_zone)

        if not np.any(safe_free):
            return None

        current_grid_pos = self.grid._conv_world_to_grid(self.current_pose[0], self.current_pose[1])
        current_y, current_x = int(current_grid_pos[0]), int(current_grid_pos[1])

        safe_positions = np.argwhere(safe_free)
        if len(safe_positions) == 0:
            return None

        distances = np.sqrt(
            (safe_positions[:, 0] - current_y) ** 2 +
            (safe_positions[:, 1] - current_x) ** 2
        )

        min_dist = 3
        max_dist = 6
        for idx, d in enumerate(distances):
            if min_dist <= d <= max_dist:
                chosen_pos = safe_positions[idx]
                x_free, y_free = self.grid._conv_grid_to_world(chosen_pos[0], chosen_pos[1])
                return (x_free, y_free)

        return None  # No suitable cell found

    def check_and_handle_general_stuck(self):
        """
        Check if the drone is stuck (not making progress) and handle it by
        finding a new target position and creating a path to it.
        Returns True if currently unstucking, False otherwise.
        """
        # Check position every 10 iterations
        if self.iteration % 10 != 0:
            if self.is_unstucking and self.path:
                return True
            return False
        
        # Initialize tracking variables if needed
        if self.last_unstuck_check_pos is None:
            self.last_unstuck_check_pos = self.current_pose[:2].copy()
            return False
        
        # Calculate movement since last check
        movement = np.linalg.norm(self.current_pose[:2] - self.last_unstuck_check_pos)
        self.last_unstuck_check_pos = self.current_pose[:2].copy()
        
        # If moving normally, reset counter
        if movement > 10.0:  # Threshold for "good movement"
            self.general_stuck_counter = 0
            self.is_unstucking = False
            return False
        
        # Increment stuck counter
        self.general_stuck_counter += 1
        
        # If stuck for too long, trigger unstuck behavior
        if self.general_stuck_counter > 5:  # 5 iterations of being stuck
            print(f"[{self.identifier}] General stuck detected! Counter: {self.general_stuck_counter}")
            
            # Find a free position to navigate to
            if self.unstuck_target is None or self.general_stuck_counter % 10 == 0:
                self.unstuck_target = self.find_free_position_for_unstuck()
                
                if self.unstuck_target:
                    print(f"[{self.identifier}] Found unstuck target: {self.unstuck_target}")
                    # Create path to unstuck target
                    self.path = self.creer_chemin(self.current_pose[:2], self.unstuck_target)
                    self.is_unstucking = True
                else:
                    print(f"[{self.identifier}] No unstuck target found")
                    self.is_unstucking = False
            
            return True
        
        return False

    def rvo_avoidance(self, command):


        if not hasattr(self, 'other_drones_positions') or not self.other_drones_positions:
            return command

        # Tuning Parameters
        EMERGENCY_DIST = 35.0  # Stop immediately (sim units)
        YIELD_DIST = 70.0      # Start yielding
        REPULSION_DIST = 90.0  # Start sliding away
        REPULSION_FORCE = 1.5  # Strength of sliding

        my_pos = self.current_pose[:2]
        
        # Accumulate avoidance forces
        avoid_vector = np.array([0.0, 0.0])
        should_stop = False
        
        for other_pos, other_id in self.other_drones_positions:
            dist_vector = my_pos - other_pos[:2]
            dist = np.linalg.norm(dist_vector)

            # Ignore distant drones
            if dist > REPULSION_DIST or dist == 0:
                continue

            # 1. EMERGENCY BRAKE
            if dist < EMERGENCY_DIST:
                # If we are head-on, just stop to prevent crash
                should_stop = True
                break

            # 2. PRIORITY SYSTEM (Yield to lower IDs)
            # If I have a higher ID, I am "secondary" and must give way
            if dist < YIELD_DIST and self.identifier > other_id:
                # If the other drone is roughly in front of me, I stop/reverse
                # Calculate angle to other drone
                angle_to_other = math.atan2(-dist_vector[1], -dist_vector[0])
                relative_angle = normalize_angle(angle_to_other - self.current_pose[2])
                
                # If it's in front (-90 to +90 degrees), I yield
                if abs(relative_angle) < math.pi / 2:
                    should_stop = True
            
            # 3. LATERAL REPULSION (Potential Field)
            # Push away: stronger as we get closer
            strength = (REPULSION_DIST - dist) / REPULSION_DIST
            normalized_vec = dist_vector / dist
            avoid_vector += normalized_vec * strength * REPULSION_FORCE

        # Apply Actions
        if should_stop:
            # STOP and slightly reverse
            command["forward"] = -0.1
            command["lateral"] = 0.0
            command["rotation"] = 0.0
            return command

        # Apply Repulsion
        if np.linalg.norm(avoid_vector) > 0.1:
            # Convert world repulsion vector to local drone frame (Forward/Lateral)
            dx = avoid_vector[0]
            dy = avoid_vector[1]
            heading = self.current_pose[2]
            
            # Rotation matrix to convert world vector to body frame
            local_x = dx * math.cos(heading) + dy * math.sin(heading)  # Forward component
            local_y = -dx * math.sin(heading) + dy * math.cos(heading) # Lateral component

            # Add to existing command (blend it)
            command["forward"] += local_x
            command["lateral"] += local_y

            # Clip values
            command["forward"] = float(np.clip(command["forward"], -1.0, 1.0))
            command["lateral"] = float(np.clip(command["lateral"], -1.0, 1.0))

        return command





    def wall_avoidance(self, command, lidar_data):
            """
            Uses Lidar to push the drone away from walls (Potential Field).
            Acts as a safety reflex preventing collisions during corner cuts.
            """
            if lidar_data is None:
                return command
                
            # 1. SETTINGS
            SAFE_DIST = 20.0   # Distance to start pushing back (pixels)
            GAIN = 2.0         # Strength of the repulsion
            
            # 2. CALCULATE FORCES
            # Lidar angles are local to the drone (0 is forward)
            angles = self.lidar().ray_angles
            
            repulsion_forward = 0.0
            repulsion_lateral = 0.0
            
            for i, dist in enumerate(lidar_data):
                if dist < SAFE_DIST:
                    # The closer the wall, the stronger the push
                    force = (SAFE_DIST - dist) / SAFE_DIST 
                    
                    # Decompose force into Forward and Lateral components
                    # We subtract because we want to push AWAY
                    angle = angles[i]
                    repulsion_forward -= force * math.cos(angle)
                    repulsion_lateral -= force * math.sin(angle)
            
            # 3. APPLY TO COMMAND
            # Only modify if there is a significant threat
            if abs(repulsion_forward) > 0.05 or abs(repulsion_lateral) > 0.05:
                command["forward"] += repulsion_forward * GAIN
                command["lateral"] += repulsion_lateral * GAIN
                
                # Clip to valid range [-1, 1]
                command["forward"] = float(np.clip(command["forward"], -1.0, 1.0))
                command["lateral"] = float(np.clip(command["lateral"], -1.0, 1.0))
                
            return command