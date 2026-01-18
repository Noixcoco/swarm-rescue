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
        resolution = 10
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
        self.prev_lat_error = 0.0

        self.path = []


        # Path smoothing parameters
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

    # --- HANSEL & GRETEL pour retourner a la rescue zone si find explored path fail ---
        self.breadcrumbs = [] # Stores (x, y) tuples
        self.last_breadcrumb_pos = None
        self.breadcrumb_spacing = 100.0 # Distance between crumbs (pixels)

        # --- KILL ZONE DETECTION ---
        self.drone_last_heard = {}  # {drone_id: {"iteration": int, "position": (x,y)}}
        self.known_kill_zones = []  # List of (x, y) tuples marking death locations
        self.DEATH_TIMEOUT = 20 # 100 iterations = ~10 seconds of silence
        self.kill_zone_grid = None
        self.declared_dead_drones = set() 
        # Two-step verification before declaring death
        self.suspected_dead_drones = {}  # {drone_id: {"first_timeout_iter": int, "position": (x,y)}}
        self.CONFIRMATION_TIMEOUT = 20 # Additional iterations to confirm death


        #tracking grasped wounded angle for better approach
        self.grasped_wounded_angle = None 






    def creer_chemin(self, start_world, goal_world, explored_only=False):
        """
        Calcule un chemin avec lissage pour des mouvements plus linéaires
        """
        #cache
        start_key = (round(start_world[0] / 10) * 10, round(start_world[1] / 10) * 10)
        goal_key = (round(goal_world[0] / 10) * 10, round(goal_world[1] / 10) * 10)
        cache_key = (start_key, goal_key, explored_only)
        
        if cache_key in self.path_cache:
            cached_path, cached_iteration = self.path_cache[cache_key]
            if self.iteration - cached_iteration < self.path_cache_max_age:
                return [np.array(pt) for pt in cached_path]

        grid = self.grid.grid.copy()



        # Conversion monde -> grille
        start = self.grid._conv_world_to_grid(*start_world)
        goal = self.grid._conv_world_to_grid(*goal_world)
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))

        # --- FIXED THRESHOLDS ---
        SEUIL_MUR = 4.01
        SEUIL_FREE = -5.0  # Free cells are BELOW this threshold
        SEUIL_UNEXPLORED_MAX = 4.0  # Unexplored cells are near 0 (between -4 and +4)
        SEUIL_UNEXPLORED_MIN = -4.99
    
        # Masque des murs (high positive values)
        is_wall = (grid >= SEUIL_MUR)
        
        # CORRECT: Only cells with NEGATIVE values are explored free space
        is_explored_free = (grid < SEUIL_FREE)
        
        # CORRECT: Unexplored cells are near zero
        is_unexplored = (grid >= SEUIL_UNEXPLORED_MIN) & (grid <= SEUIL_UNEXPLORED_MAX)
        
        # Dilate les murs pour éviter les zones proches
        struct = np.ones((5, 5), dtype=bool)
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)


        # --- 4. SOFT CONSTRAINT SETUP (Distance Map) ---
        # Calculate distance from every pixel to the nearest wall (IN CELLS)
        # This creates the "Gradient" that pushes the drone to the center.
        # We invert is_wall because we want distance to the WALLS.
        dist_map = ndimage.distance_transform_edt(~is_wall)

        # DEFINITION: How far (in world units) do we want to be?
        COMFORT_DISTANCE_WORLD = 200.0  # e.g., 60cm or 60px
        
        # CONVERSION: Convert that to grid cells so we can compare with dist_map
        comfort_dist_cells = COMFORT_DISTANCE_WORLD / self.grid.resolution
        
        # Max penalty to apply if we are right next to the wall
        MAX_PENALTY = 50.0

        
        # --- MODIFIED DRONE AVOIDANCE ZONE - ONLY AVOID DRONES IN FRONT ---
        # Cache drone danger zone for a few iterations if positions haven't changed
       
        # --- NEW: ADD OTHER DRONES AS TEMPORARY OBSTACLES ---
        # This treats other drones as "walls" for the pathfinder
        if hasattr(self, 'other_drones_positions') and self.other_drones_positions:
        
            DRONE_OBSTACLE_RADIUS = 40.0 
            radius_cells = int(DRONE_OBSTACLE_RADIUS / self.grid.resolution)
            
            for other_info in self.other_drones_positions:
                other_pos = other_info[0]
                
                # Only consider drones that are somewhat close (optimization)
                # e.g., within 200 pixels. Far away drones don't matter.
                if math.hypot(other_pos[0] - start_world[0], other_pos[1] - start_world[1]) > 200.0:
                    continue

                try:
                    # Convert drone world pos to grid
                    p_grid = self.grid._conv_world_to_grid(other_pos[0], other_pos[1])
                    py, px = int(p_grid[0]), int(p_grid[1])
                    
                    # Define a square around the drone
                    y0 = max(0, py - radius_cells)
                    y1 = min(grid.shape[0], py + radius_cells + 1)
                    x0 = max(0, px - radius_cells)
                    x1 = min(grid.shape[1], px + radius_cells + 1)
                    
                    # Mark this area as BLOCKED
                    danger_zone[y0:y1, x0:x1] = True
                    
                except Exception:
                    continue

    

        # Si explored_only=True, ajouter les zones non explorées à danger_zone
        if explored_only:
            danger_zone = danger_zone | (~is_explored_free)  # Block unexplored cells!
            print(f"[{self.identifier}] explored_only=True: Blocking {np.sum(is_unexplored)} unexplored cells")
        
        # Si le start ou le goal sont dans la danger_zone (par ex. drone collé au mur),
        # on autorise une petite zone autour d'eux pour permettre à A* de s'extraire.
        try:
            radius_clear = 1
            sx, sy = start
            gx, gy = goal
            x0 = max(0, sx - radius_clear)
            x1 = min(grid.shape[0], sx + radius_clear + 1)
            y0 = max(0, sy - radius_clear)
            y1 = min(grid.shape[1], sy + radius_clear + 1)
            danger_zone[x0:x1, y0:y1] = False
            # Clear zone autour du goal avec un rayon plus grand pour le rescue center
            radius_clear_goal = 5  # Plus grand rayon pour le goal (rescue center)
            y0 = max(0, gy - radius_clear_goal)
            y1 = min(grid.shape[1], gy + radius_clear_goal + 1)
            x0 = max(0, gx - radius_clear_goal)
            x1 = min(grid.shape[0], gx + radius_clear_goal + 1)
            danger_zone[x0:x1, y0:y1] = False
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
                # --- 1. FULL PATH RECONSTRUCTION ---
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # --- 2. FIXED-STEP SUBSAMPLING ---
                # Keep 1 point every 4 steps to maintain a stable trail.
                STEP = 7 
                if len(path) > STEP:
                    compressed = [path[0]]
                    for i in range(1, len(path) - 1):
                        prev_v = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                        next_v = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                        if prev_v != next_v:
                            compressed.append(path[i])
                    compressed.append(path[-1])
                    path = compressed
                else:
                    path = path  # Keep short paths as-is
            
                # --- 3. APPLY SMOOTHING ---
                if len(path) > 2 and self.path_smoothing_enabled:
                    smoothed = self.smooth_path(path, danger_zone)
                else:
                    smoothed = path
                
                #Convert grid -> world
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


                # --- NEW: COST CALCULATION (Soft Constraints) ---
                base_cost = math.hypot(dx, dy)
                # Retrieve distance to nearest wall (in CELLS)
                dist_to_wall_cells = dist_map[neighbor[0], neighbor[1]]
                
                penalty = 0.0

                # Apply penalty if closer than comfort distance
                if dist_to_wall_cells < comfort_dist_cells:
                    # Linear gradient: closer to wall = higher cost
                    # 0 penalty at comfort distance, MAX_PENALTY at wall
                    factor = 1.0 - (dist_to_wall_cells / comfort_dist_cells)
                    penalty = MAX_PENALTY * factor
                
                move_cost = base_cost + penalty
                # -----------------------------------------------

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
        x, y = int(round(point[0])), int(round(point[1]))
        
        if not (0 <= y < danger_zone.shape[1] and 0 <= x < danger_zone.shape[0]):
            return False
        
        return not danger_zone[x, y]
    

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

        # --- RECORD HISTORY ---
        self.update_breadcrumbs()

         # ---Check if lidar is available before updating grid, if drone killed  ---
        lidar_data = self.lidar_values()
        if lidar_data is None:
            # Drone is destroyed - cannot continue
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        

            
        # Mise à jour de la grille probabiliste self.grid.grid (utilisée pour l'exploration)
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())
        self.grid.update_grid(pose=self.estimated_pose) # Mise à jour de la carte utilisée!

        #Gestion des kill zones
        if self.kill_zone_grid is None:
            self.kill_zone_grid = np.zeros_like(self.grid.grid)
        else:
            self.apply_kill_zones_to_grid()
      

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

        # --- Check for general stuck condition FIRST ---
        if self.check_and_handle_general_stuck():
            # If unstucking, follow the unstuck path
            if self.path:
                command = self.follow_path(lidar_data)
                return command
            else:
                # Unstuck path failed, try simple reverse maneuver
                print("lost lost lost wsh wsh wsh")
                return {"forward": -0.5, "lateral": 0.3, "rotation": 0.4, "grasper": 1}
            
               

        # STATE MACHINE LOGIC
        # Transitions
        if self.state == self.Activity.EXPLORING:

            if self.grasper.grasped_wounded_persons:
                self.state = self.Activity.GOING_TO_RESCUE_CENTER
                if self.grasped_wounded_angle is None:
                    self.grasped_wounded_angle = self.get_grasped_wounded_orientation()

                if self.rescue_zone_points:
                    target_index = int(self.identifier) % len(self.rescue_zone_points)
                    target_zone = self.rescue_zone_points[target_index]
                    self.path = self.creer_chemin(self.current_pose[:2], target_zone, explored_only=True)
                    self.last_replan_iteration = self.iteration
            
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

            # Replan every 30 iterations to adapt to updated wounded position
            if self.current_target_wounded is not None and self.iteration % 30 == 0:
                # Check if target position has moved significantly
                if not self.path or len(self.path) == 0:
                    # No path exists - create one
                    self.path = self.creer_chemin(self.current_pose[:2], self.current_target_wounded)
                    self.last_replan_iteration = self.iteration
                else:
                    # Check if wounded moved significantly from path end
                    path_end = self.path[-1]
                    distance_moved = np.linalg.norm(np.array(self.current_target_wounded) - path_end)
                    
                    if distance_moved > 50.0:  # Wounded moved more than 50 pixels
                        print(f"[{self.identifier}] Wounded moved {distance_moved:.1f}px - Replanning!")
                        self.path = self.creer_chemin(self.current_pose[:2], self.current_target_wounded)
                        self.last_replan_iteration = self.iteration

                    

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
                    # ATTRIBUTE RESCUE ZONE BASED ON DRONE ID
                    # Drone 0 -> Zone 0, Drone 1 -> Zone 1, etc.
                    target_index = int(self.identifier) % len(self.rescue_zone_points)
                    target_zone = self.rescue_zone_points[target_index]
    
                    self.path = self.creer_chemin(self.current_pose[:2], target_zone, explored_only=True)
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
                self.grasped_wounded_angle = None
                if self.current_target_wounded is not None:
                    # Remove assignment so other drones don't try to grab it
                    self.wounded_assignments.pop(self.current_target_wounded, None)
               
                self.state = self.Activity.EXPLORING
                self.current_target_wounded = None
                # Reset Hansel & Gretel for the next run
                self.breadcrumbs = []
                self.last_breadcrumb_pos = None
                self.path = []
                
                
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
                            print(f"[{self.identifier}] No safe explored path to rescue center, using breadcrumbs!")
                            if len(self.breadcrumbs) > 2:
                                # Reverse the recorded history
                                return_path = self.breadcrumbs[::-1]
                                
                                # Convert to numpy and set as path
                                self.path = [np.array([x, y]) for x, y in return_path]
                                
                                # Append the actual rescue center at the end to be sure
                                self.path.append(np.array(self.rescue_zone_points[0]))
                                
                                print(f"[{self.identifier}] Success: Using Breadcrumbs (Length: {len(self.path)})")
                            else:
                                print(f"[{self.identifier}] No breadcrumbs available.")


        # --- 2. STRATÉGIE ---
        # Replanification for exploration (only when in EXPLORING state)
        if self.state == self.Activity.EXPLORING:
            need_replan = False
            if not self.path or len(self.path) < 1:
                need_replan = True
            # Met à jour le chemin tous les 20 itérations (si cible existante)
            if self.path and hasattr(self, 'target_point') and self.iteration % 30 == 0:
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
                    # FALLBACK: Use local frontier detection
                    local_frontiers = self.find_safe_frontier_points() 
                    if local_frontiers: 
                        distances = [np.linalg.norm(f - self.current_pose[:2]) for f in local_frontiers]
                        target_index = np.argmin(distances)
                        target_point = local_frontiers[target_index]
                        self.target_point = target_point
                        self.path = self.creer_chemin(self.current_pose[:2], target_point)
                    else:
                        print(f"[{self.identifier}] MAP FULLY EXPLORED - No more frontiers to explore!")

        # Generate movement commands based on current state
        if self.state == self.Activity.EXPLORING:
            if self.path:   
                command = self.follow_path(lidar_data)

            else:
                command = {"forward": 0.3, "lateral": 0.0, "rotation": 0.0}

        elif self.state == self.Activity.GOING_TO_WOUNDED:

            command = self.go_to_wounded(lidar_data)
            

        elif self.state == self.Activity.GOING_TO_RESCUE_CENTER:

            if self.rescue_zone_points:
                dist_to_rescue = np.linalg.norm(
                    np.array(self.rescue_zone_points[0]) - self.current_pose[:2]
                )
            else:
                dist_to_rescue = 999

            if self.path and dist_to_rescue < 100.0:
                # Close to rescue center, use simple approach
                command = self.go_to_rescue_center_oriented(lidar_data)
            elif self.path: 
                command = self.follow_path(lidar_data)
            else:
                command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}



########## GRASPER LOGIC ############
        # Grasper is ONLY active when going to wounded or rescue center
        if self.state == self.Activity.GOING_TO_WOUNDED or self.state == self.Activity.GOING_TO_RESCUE_CENTER:
            command["grasper"] = 1
        else:
            command["grasper"] = 0


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



        # Apply this LAST to prevent hitting walls while dodging drones
        if self.state != self.Activity.GOING_TO_WOUNDED and self.state != self.Activity.GOING_TO_RESCUE_CENTER:
            command = self.wall_avoidance(command, lidar_data)

        # 2. Apply Drone Repulsion (Safety against other agents)
        # This will override/modify the command to push us away from collisions
        command = self.drone_repulsion(command)

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
                # Now fast - other_grasped_wounded already initialized
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


# --- NEW RESCUE ZONE LOGIC: 5 EVENLY SPACED POINTS ---
        for nx, ny in newly_seen_rescue: 
            pt = (nx, ny)
            self._add_or_merge_rescue_point(pt)


    # --------------------------------------------------------------------------
    # FONCTION DE DÉTECTION DES FRONTIÈRES SÛRES 
    # --------------------------------------------------------------------------

    def find_safe_frontier_points(self) -> list:
        
        grid_map = self.grid.grid 
        
        # RELAXED THRESHOLDS - Encourage exploring unexplored areas
        SEUIL_FREE = -3.0        # Lightly explored (was -7.0)
        SEUIL_MUR = 6.0         
        SEUIL_UNEXPLORED_MIN = -2.99  # Wider unexplored range
        SEUIL_UNEXPLORED_MAX = 5.99
    
        frontiers = []

        # Masks
        is_unknown = (grid_map >= SEUIL_UNEXPLORED_MIN) & (grid_map <= SEUIL_UNEXPLORED_MAX)
        is_wall = (grid_map >= SEUIL_MUR)  
        is_free = (grid_map < SEUIL_FREE) # Lightly explored areas
        
        # KEY: Exclude heavily explored dark blue corridor
        is_heavily_explored = (grid_map < -40.0)

        # Frontier detection
        structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=bool)

        unknown_neighbors = binary_dilation(is_unknown, structure=structure)
        
        # FIXED: Find free cells near unexplored, but NOT in heavily explored corridor
        frontier_mask = is_free & (~is_heavily_explored) & unknown_neighbors

        # Safety margin around walls
        struct = np.ones((5, 5), dtype=bool)
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
            
            # VALIDATION: Ensure nearby unexplored cells exist
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
            
            # REJECT dark blue corridor: if 70%+ heavily explored, skip
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
        if not self.path:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        
        # Reduce lookahead when near walls for tighter cornering
        min_lidar_dist = min(lidar_data) if lidar_data is not None and len(lidar_data) > 0 else 999
        
        # Adaptive lookahead logic
        if min_lidar_dist < 40.0:
            lookahead_dist = 15.0  # Tighter following near obstacles
        else:
            # Defaults to 40.0 or a class attribute if defined
            lookahead_dist = getattr(self, 'path_lookahead_distance', 40.0)

        # 2. TARGET SELECTION (Pure Pursuit style)
        lookahead_idx = 0
        for i, wp in enumerate(self.path):
            if np.linalg.norm(wp - self.current_pose[:2]) > lookahead_dist:
                lookahead_idx = i
                break
        target_pos = self.path[min(lookahead_idx, len(self.path)-1)]

        # 3. ERROR COMPUTATION (Body Frame - From Version 2)
        delta_pos = target_pos - self.current_pose[:2]
        heading = self.current_pose[2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        # Project world error into robot's local frame
        x_err = cos_h * delta_pos[0] + sin_h * delta_pos[1]   # Longitudinal (Forward/Back)
        y_err = -sin_h * delta_pos[0] + cos_h * delta_pos[1]  # Lateral (Left/Right)

        # 4. ROTATION CONTROL (With Damping & Load Sensitivity)
        angle_error = normalize_angle(target_angle - heading)
        deriv_angle = angle_error - self.prev_angle_error
        
        # Base gains
        Kp_rot = self.Kp
        Kd_rot = self.Kd

        # Apply Version 1's damping for small errors and Version 2's load reduction
        if abs(angle_error) < math.radians(10):
            Kp_rot *= 0.6
            Kd_rot *= 0.8

        rotation_speed = Kp_rot * angle_error + Kd_rot * deriv_angle
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error

        # 5. LATERAL CONTROL (Cross-track Correction)
        Kp_lat = 0.05
        Kd_lat = 0.02
        
        if not hasattr(self, 'prev_lat_error'): self.prev_lat_error = 0.0
        lat_deriv = y_err - self.prev_lat_error
        lateral_cmd = Kp_lat * y_err + Kd_lat * lat_deriv
        
        # Apply damping for small angle errors to prevent "crabbing" when straight
        if abs(angle_error) < 0.1:
            lateral_cmd *= 0.7
            
        lateral_cmd = float(np.clip(lateral_cmd, -1.0, 1.0))
        self.prev_lat_error = y_err

        # 6. FORWARD SPEED PROFILE
        max_speed = 12.0 
        # Use x_err (longitudinal distance) to scale speed
        target_speed = max(0.0, min(max_speed, x_err * 0.15 + 0.3))

        measured_vel = self.measured_velocity()
        measured_speed = math.sqrt(measured_vel[0] ** 2 + measured_vel[1] ** 2)
        
        speed_error = target_speed - measured_speed
        deriv_speed = speed_error - self.prev_speed_error
        
        Kp_f = self.Kp_pos
        Kd_f = self.Kd_pos 
        forward_cmd = Kp_f * speed_error + Kd_f * deriv_speed

        # Safety: Reduce forward speed during sharp turns (From Version 1 & 2)
        if abs(angle_error) > 1.0:
            forward_cmd *= 0.0  # Stop and pivot for very sharp angles
        elif abs(angle_error) > 0.4:
            forward_cmd *= 0.5  # Slow down for moderate turns

        forward_cmd = float(np.clip(forward_cmd, -1.0, 1.0))
        self.prev_speed_error = speed_error

        # 7. WAYPOINT MANAGEMENT (Multi-drop logic from Version 2)
        close_thresh = 50.0 
        drop_until = -1
        max_check = min(5, len(self.path))
        for i in range(max_check):
            d = np.linalg.norm(self.path[i] - self.current_pose[:2])
            if d < close_thresh:
                drop_until = i
            else:
                break
                
        if drop_until >= 0:
            self.path = self.path[drop_until+1:]

        return {"forward": forward_cmd, "lateral": lateral_cmd, "rotation": rotation_speed}


    # --------------------------------------------------------------------------
    # FONCTIONS PRINCIPALES (Localisation, Cartographie Binaire)
    # --------------------------------------------------------------------------

    def update_pose(self):
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        
        # Check if GPS data is None (drone destroyed by kill zone) ---
        if gps_pos is None:
            # Drone is destroyed - stop processing
            return
        
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
        current_iteration = self.iteration

        # Pre-allocate collections
        all_wounded = []
        all_assignments = {}
        all_grasped = set()
        all_rescue_zones = []
        all_frontier_clusters = []
        other_drones_positions = []

        # Single pass through messages
        for msg in received_messages:

            other_message = msg[1]
            other_id = other_message.get("drone_id")
            
            # Skip own messages
            if other_id == self.identifier:
                continue     

            # Drone positions (always needed for collision avoidance)
            pos = other_message.get("drone_pose")
            if pos is not None:
                # Store as tuple: (position_array, id)
                other_drones_positions.append((np.array(pos), other_id))

                 # CHECK IF THIS DRONE WAS DECLARED DEAD (FALSE POSITIVE)
                if other_id in self.declared_dead_drones:
                    print(f"[{self.identifier}] FALSE POSITIVE DETECTED! Drone {other_id} is ALIVE!")
                    
                    # Remove from dead list
                    self.declared_dead_drones.remove(other_id)
                    
                    # Find and remove the kill zone associated with this drone
                    # We need to find the kill zone closest to where we last heard from them
                    if other_id in self.drone_last_heard:
                        false_death_pos = self.drone_last_heard[other_id]["position"]
                        
                        # Remove from known_kill_zones list
                        kill_zones_to_remove = []
                        for kz_pos in self.known_kill_zones:
                            if math.hypot(false_death_pos[0] - kz_pos[0], 
                                        false_death_pos[1] - kz_pos[1]) < 100.0:
                                kill_zones_to_remove.append(kz_pos)
                        
                        for kz in kill_zones_to_remove:
                            self.known_kill_zones.remove(kz)
                            print(f"[{self.identifier}] Removed kill zone at {kz}")
                        
                        # Clear the kill zone from the grid
                        self.clear_kill_zone_from_grid(false_death_pos)


                # Update last heard status
                self.drone_last_heard[other_id] = {
                    "iteration": current_iteration,
                    "position": (pos[0], pos[1])
                }


            
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
                for r in all_rescue_zones:
                    self._add_or_merge_rescue_point(r)
        

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


        # --- DETECT DEATHS (Check for silent drones) ---

        if not self.base.in_kill_zone:
            for drone_id, info in list(self.drone_last_heard.items()):
                silence_duration = current_iteration - info["iteration"]
                
                # If a drone has been silent for too long, assume death
                if silence_duration > self.DEATH_TIMEOUT:
                    death_pos = info["position"]

                #  Don't mark kill zone if we're too far away to hear them anyway
                    my_distance_to_death = math.hypot(
                        self.current_pose[0] - death_pos[0],
                        self.current_pose[1] - death_pos[1]
                )
                    
                    # If they were far away, they might just be out of range
                    MAX_COMM_RANGE = 200.0 
                    if my_distance_to_death > MAX_COMM_RANGE:
                        continue
                    
                    # First timeout - add to suspected list
                    if drone_id not in self.suspected_dead_drones:
                        self.suspected_dead_drones[drone_id] = {
                            "first_timeout_iter": current_iteration,
                            "position": death_pos
                        }
                        print(f"[{self.identifier}] ⚠️  SUSPECTED DEATH: Drone {drone_id} at {death_pos}")
                        print(f"    Waiting {self.CONFIRMATION_TIMEOUT} iterations for confirmation...")
                        continue  # Don't declare yet!

                    # STEP 2: Confirmation timeout - declare death
                    suspected_info = self.suspected_dead_drones[drone_id]
                    confirmation_duration = current_iteration - suspected_info["first_timeout_iter"]
                    
                    if confirmation_duration >= self.CONFIRMATION_TIMEOUT:
                        # Check if we already marked this area
                        is_new_kill_zone = True
                        for kz_pos in self.known_kill_zones:
                            if math.hypot(death_pos[0] - kz_pos[0], death_pos[1] - kz_pos[1]) < 50.0:
                                is_new_kill_zone = False
                                break

                        if is_new_kill_zone:
                            print(f"[{self.identifier}] DETECTED KILL ZONE! Drone {drone_id} died at {death_pos}, at iteration {info["iteration"]}")
                            self.known_kill_zones.append(death_pos)
                            self.mark_kill_zone_on_grid(death_pos,drone_id)
                            self.declared_dead_drones.add(drone_id)

            # --- FALSE ALARM CHECK ---
            for drone_id in list(self.suspected_dead_drones.keys()):
                if drone_id in [d_id for (_, d_id) in other_drones_positions]:
                    print(f"[{self.identifier}]  FALSE ALARM: Drone {drone_id} is alive! Removing from suspected list.")
                    self.suspected_dead_drones.pop(drone_id, None)


            
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
        struct = np.ones((5, 5), dtype=bool)
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


    def wall_avoidance(self, command, lidar_data):
            """
            Uses Lidar to push the drone away from walls (Potential Field).
            Acts as a safety reflex preventing collisions during corner cuts.
            """
            if lidar_data is None:
                return command

                
            # 1. SETTINGS
            SAFE_DIST = 30.0   # Distance to start pushing back (pixels)
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
    
    def _add_or_merge_rescue_point(self, new_point):
            """
            STABILIZATION LOGIC:
            Only add a point if it is FAR from all existing points.
            If it is close to an existing point, IGNORE IT (keep the original stable).
            """
            nx, ny = new_point
            MAX_RESCUE_POINTS = 5
            # Radius to consider a point "already known"
            # 100.0 is safe to ensure we don't accidentally add the same zone twice
            DEDUP_RADIUS = 80.0 

            # 1. Check against ALL existing points
            for (rx, ry) in self.rescue_zone_points:
                dist = math.hypot(rx - nx, ry - ny)
                if dist < DEDUP_RADIUS:
                    # We already have a point roughly here. 
                    # Do NOT update it. Do NOT add a new one.
                    # Just return to keep the existing point stable.
                    return

            # 2. If we are here, it is a completely NEW area.
            if len(self.rescue_zone_points) < MAX_RESCUE_POINTS:
                self.rescue_zone_points.append((nx, ny))



    def drone_repulsion(self, command):
            """
            Simple Potential Field Repulsion for Drones.
            Pushes the drone away from others if they get too close.
            """
            if not hasattr(self, 'other_drones_positions') or not self.other_drones_positions:
                return command

            # SETTINGS
            SAFE_DIST = 80.0  # Start pushing away at 70 pixels (approx 0.7 meter)
            GAIN = 2.0         # Strong push (Stronger than walls to prevent tangling)

            repulsion_forward = 0.0
            repulsion_lateral = 0.0
            
            my_pos = self.current_pose[:2]
            my_theta = self.current_pose[2]

            for other_info in self.other_drones_positions:
                # Extract position (format is usually (pos_array, id))
                other_pos = other_info[0]
                
                # Vector: From THEM -> to ME (We want to be pushed AWAY)
                dx = my_pos[0] - other_pos[0]
                dy = my_pos[1] - other_pos[1]
                dist = math.hypot(dx, dy)

                if 0 < dist < SAFE_DIST:
                    # 1. Magnitude: Stronger as we get closer (0.0 to 1.0)
                    force = (SAFE_DIST - dist) / SAFE_DIST
                    
                    # 2. Direction: Global angle of the push
                    angle_global = math.atan2(dy, dx)
                    
                    # 3. Local Projection: Convert to Forward/Lateral relative to head
                    angle_local = normalize_angle(angle_global - my_theta)
                    
                    repulsion_forward += force * math.cos(angle_local)
                    repulsion_lateral += force * math.sin(angle_local)

            # Apply to command
            if abs(repulsion_forward) > 0.01 or abs(repulsion_lateral) > 0.01:
                # Add to existing movement command
                command["forward"] += repulsion_forward * GAIN
                command["lateral"] += repulsion_lateral * GAIN
                
                # Clip to ensure we don't exceed motor limits
                command["forward"] = float(np.clip(command["forward"], -1.0, 1.0))
                command["lateral"] = float(np.clip(command["lateral"], -1.0, 1.0))
                
            return command
    

    def update_breadcrumbs(self):

    # Don't record if we are already going home!
        if self.state == self.Activity.GOING_TO_RESCUE_CENTER:
            return

        current_pos_tuple = (self.current_pose[0], self.current_pose[1])

        # Initialize if empty
        if self.last_breadcrumb_pos is None:
            self.breadcrumbs.append(current_pos_tuple)
            self.last_breadcrumb_pos = current_pos_tuple
            return

        # Calculate distance from last crumb
        dist = math.hypot(current_pos_tuple[0] - self.last_breadcrumb_pos[0], 
                        current_pos_tuple[1] - self.last_breadcrumb_pos[1])

        # Only drop a crumb if we moved enough (prevents clumps when stuck)
        if dist >= self.breadcrumb_spacing:
            self.breadcrumbs.append(current_pos_tuple)
            self.last_breadcrumb_pos = current_pos_tuple
    


    def mark_kill_zone_on_grid(self, death_pos, drone_id):  # ✅ ADD drone_id parameter
        """
        Mark a SQUARE area as a permanent kill zone on BOTH grids.
        Size is estimated from the drone's last heard (safe) position to death position.
        """
        try:
            # Initialize kill zone grid if needed
            if self.kill_zone_grid is None:
                self.kill_zone_grid = np.zeros_like(self.grid.grid)

           
            square_size = self.detect_kill_zone_size(death_pos, drone_id)
            
            # Convert world position to grid coordinates
            grid_pos = self.grid._conv_world_to_grid(death_pos[0], death_pos[1])
            center_y, center_x = int(grid_pos[0]), int(grid_pos[1])
            
            # Calculate SQUARE bounds (in GRID CELLS)
            size_cells = int(square_size / self.grid.resolution)
            half_size = size_cells // 2
            
            y0 = max(0, center_y - half_size)
            y1 = min(self.grid.grid.shape[0], center_y + half_size)
            x0 = max(0, center_x - half_size)
            x1 = min(self.grid.grid.shape[1], center_x + half_size)
            
            # MARK ON BOTH GRIDS SIMULTANEOUSLY
            self.kill_zone_grid[y0:y1, x0:x1] = 1.0      # Permanent record
            self.grid.grid[y0:y1, x0:x1] = 100.0         # Active pathfinding obstacle
            
            print(f"[{self.identifier}] Marked kill zone at {death_pos}")
            print(f"    Size: {square_size:.0f}x{square_size:.0f}px")
            print(f"    Grid bounds: y[{y0}:{y1}], x[{x0}:{x1}]")
            
        except Exception as e:
            print(f"[{self.identifier}] Error marking kill zone: {e}")
            import traceback
            traceback.print_exc()


    def apply_kill_zones_to_grid(self):
        """
        RE-APPLY all known kill zones after lidar updates overwrite them.
        Uses vectorized operations for speed (no loops over zones).
        """
        if self.kill_zone_grid is None or not np.any(self.kill_zone_grid):
            return  # No kill zones marked yet
    
        # Where kill_zone_grid == 1.0, set grid.grid to 100.0
        self.grid.grid[self.kill_zone_grid == 1.0] = 100.0



    def detect_kill_zone_size(self, death_pos, drone_id):
        """
        ULTRA-SIMPLE METHOD: Use the drone's LAST HEARD position as the safe position.
        The last heard position is inherently safe (they were alive and transmitting).
        Returns square size in PIXELS.
        """
        try:
            
            # SIMPLIFIED APPROACH: Use a reasonable default based on drone speed
            # Drones move at ~50-100 pixels per timeout period
            # DEATH_TIMEOUT = 20 iterations = ~2 seconds
            # Max speed ≈ 100 px/s → 200px in 2 seconds
            
            SAFETY_MARGIN = 1.5  # 50% larger for safety
            ASSUMED_TRAVEL_DISTANCE = 150.0  # Conservative estimate
            
            square_size = ASSUMED_TRAVEL_DISTANCE * SAFETY_MARGIN
            
            # Validation
            MIN_SIZE = 100.0
            MAX_SIZE = 500.0
            square_size = np.clip(square_size, MIN_SIZE, MAX_SIZE)
            
            print(f"[{self.identifier}] Kill zone detection for drone {drone_id}:")
            print(f"    Death position: {death_pos}")
            print(f"    Assumed travel distance: {ASSUMED_TRAVEL_DISTANCE}px")
            print(f"    Square size (with {SAFETY_MARGIN}x margin): {square_size:.0f}x{square_size:.0f}px")
            
            return float(square_size)
            
        except Exception as e:
            print(f"[{self.identifier}] Error detecting kill zone size: {e}")
            import traceback
            traceback.print_exc()
            return 200.0  # Safe fallback
        


    def go_to_wounded(self, lidar_data) -> CommandsDict:
        """
        Navigation vers le blessé.
        - Si distance > 40 pixels : utilise follow_path (suivi de chemin A*)
        - Si distance < 40 pixels : s'arrête, s'oriente vers le blessé, et fonce tout droit.
        """
        if self.current_target_wounded is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        # Calcul de la distance au blessé
        dist_to_wounded = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])

        if dist_to_wounded > 40.0:
            return self.follow_path(lidar_data)
        else:
            # Comportement proche : orientation puis charge
            delta_pos = np.array(self.current_target_wounded) - self.current_pose[:2]
            target_angle = math.atan2(delta_pos[1], delta_pos[0])
            heading = self.current_pose[2]
            angle_error = normalize_angle(target_angle - heading)

            # PID Rotation
            deriv_error = angle_error - self.prev_angle_error
            rotation_speed = self.Kp * angle_error + self.Kd * deriv_error
            rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
            self.prev_angle_error = angle_error

            # Seuil d'alignement (1 degré)
            ALIGNMENT_THRESHOLD = math.radians(1.0)

            if abs(angle_error) > ALIGNMENT_THRESHOLD:
                # Phase 1 : S'arrêter et s'orienter
                return {"forward": 0.0, "lateral": 0.0, "rotation": rotation_speed}
            else:
                # Phase 2 : Fonce tout droit
                return {"forward": 1.0, "lateral": 0.0, "rotation": rotation_speed}



    def clear_kill_zone_from_grid(self, false_death_pos):
        """
        Remove a kill zone from both grids when a false positive is detected.
        """
        try:
            if self.kill_zone_grid is None:
                return
            
            # Use same size calculation as marking
            SAFETY_MARGIN = 1.5
            ASSUMED_TRAVEL_DISTANCE = 150.0
            square_size = ASSUMED_TRAVEL_DISTANCE * SAFETY_MARGIN
            
            # Convert world position to grid coordinates
            grid_pos = self.grid._conv_world_to_grid(false_death_pos[0], false_death_pos[1])
            center_y, center_x = int(grid_pos[0]), int(grid_pos[1])
            
            # Calculate SQUARE bounds (in GRID CELLS)
            size_cells = int(square_size / self.grid.resolution)
            half_size = size_cells // 2
            
            y0 = max(0, center_y - half_size)
            y1 = min(self.grid.grid.shape[0], center_y + half_size)
            x0 = max(0, center_x - half_size)
            x1 = min(self.grid.grid.shape[1], center_x + half_size)
            
            # CLEAR from both grids
            self.kill_zone_grid[y0:y1, x0:x1] = 0.0      # Remove permanent record
            self.grid.grid[y0:y1, x0:x1] = 0.0
            # Don't reset grid.grid values - let lidar naturally re-explore
            # This is safer than guessing what the values should be
            
            print(f"[{self.identifier}] Cleared kill zone area at {false_death_pos}")
            print(f"    Grid bounds cleared: y[{y0}:{y1}], x[{x0}:{x1}]")
            
        except Exception as e:
            print(f"[{self.identifier}] Error clearing kill zone: {e}")
            import traceback
            traceback.print_exc()


    def get_grasped_wounded_orientation(self):
        """
        Use semantic sensor to detect the angle of the grasped wounded person
        relative to the drone's heading. Returns angle in radians.
        """
        try:
            detections = self.semantic_values()
            if not detections:
                return None
      
            for data in detections:
                
                etype = getattr(data, 'entity_type', None)
                name = etype.name if hasattr(etype, 'name') else str(etype)
                    
                if 'WOUNDED' in name.upper() and self.grasper.grasped_wounded_persons:
                    # Get angle relative to drone's heading
                    dist = float(getattr(data, 'distance', 0.0))
                        
                    # Only consider if very close (must be the grasped one)
                    if dist < 20.0:
                        angle = float(getattr(data, 'angle', 0.0))
                        return angle  # Return relative angle
                
                            
        except Exception as e:
                print(f"[{self.identifier}] Error processing semantic data: {e}")
                    
        return None
    

    def go_to_rescue_center_oriented(self, lidar_data) -> CommandsDict:
        """
        FASTEST RETURN: Navigate to rescue center at MAXIMUM SPEED.
        Only slow down when EXTREMELY CLOSE (<15px) to align wounded for handoff.
        """
        if not self.path:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        
        rescue_center_pos = self.rescue_zone_points[0] if self.rescue_zone_points else None
        if rescue_center_pos is None:
            return self.follow_path(lidar_data)
        
        dist_to_rescue = np.linalg.norm(np.array(rescue_center_pos) - self.current_pose[:2])
        
        # --- MAXIMUM SPEED until 15px (reduced from 30px) ---
        if dist_to_rescue > 15.0:
            return self.follow_path(lidar_data)
        
        # --- FINAL ALIGNMENT: Only when <15px away ---
        to_rescue = np.array(rescue_center_pos) - self.current_pose[:2]
        angle_to_rescue = math.atan2(to_rescue[1], to_rescue[0])
        
        # Adjust heading to present wounded correctly
        if self.grasped_wounded_angle is not None:
            desired_heading = normalize_angle(angle_to_rescue - self.grasped_wounded_angle)
        else:
            desired_heading = angle_to_rescue
        
        # Rotation control
        heading = self.current_pose[2]
        angle_error = normalize_angle(desired_heading - heading)
        rotation_speed = self.Kp * angle_error
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        
        # FASTER approach while rotating (increased from 0.3 and 0.1)
        forward_speed = 0.6 if abs(angle_error) < math.radians(30) else 0.3
        
        return {"forward": forward_speed, "lateral": 0.0, "rotation": rotation_speed}