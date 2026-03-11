import math
import random
import numpy as np
from enum import Enum
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy import ndimage
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
import arcade
import heapq
import sys
from pathlib import Path
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR

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
        GOING_TO_RETURN_AREA = 4


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.current_pose = np.array([0.0, 0.0, 0.0])

        self.iteration: int = 0
        resolution = 10
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
        
        # Sécurité et Inflation 
        self.robot_radius_pixels = 30
        self.inflation_radius_cells = int(self.robot_radius_pixels / self.grid.resolution)
        if self.inflation_radius_cells < 1:
            self.inflation_radius_cells = 1
        
        # parametre PID rotation
        self.prev_angle_error = 0.0
        self.Kp = 5.0
        self.Kd = 3.0

        # PID translation
        self.Kp_pos = 6.0
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
        self.return_area_points = []


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
        self.kf_Q = np.eye(4) * 8.0  # Small process noise
        self.kf_Q[2, 2] = 1.0  # Higher uncertainty on velocity
        self.kf_Q[3, 3] = 1.0
        # Measurement noise covariance (GPS noise)
        self.kf_R = np.eye(2) * 10.0  # GPS measurement noise
        # Time step for prediction (will be updated)
        self.kf_dt = 0.1
        self.kf_last_time = 0
        self.kf_initialized = False


        self.evaluated_wounded = set() # pour l'attribution des blessés aux drones
        
        self.wounded_assignments = {}  # {wounded_pos: drone_id}

        self.removed_wounded = []  # Liste des wounded supprimés
        self.removed_wounded_set = set() 
       
        
        # General unstuck mechanism 
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

         # HANSEL & GRETEL pour retourner a la rescue zone si find explored path fail 
        self.breadcrumbs = [] # Stores (x, y) tuples
        self.last_breadcrumb_pos = None
        self.breadcrumb_spacing = 100.0 # Distance between crumbs (pixels)

        #tracking grasped wounded angle for better approach
        self.grasped_wounded_angle = None 

        # Dead Drones / Kill Zones Detection
        self.dead_drones = []  # Confirmed dead drones (x, y)
        self.suspected_dead_drones = [] # Candidates: {'pos': (x,y), 'first_seen_iter': int, 'last_seen_iter': int}
        self.dead_drone_radius = 80.0 # Radius to match a drone
        self.dead_confirm_iterations = 5 # How many iterations of immobility to confirm death

        self.prev_wall_error = None
        self.integral_wall_error = 0.0
        self.last_slam_score = 0.0





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

        # thresholds
        SEUIL_MUR = 3.0
        SEUIL_FREE = -5.0  # Free cells are BELOW this threshold
        SEUIL_UNEXPLORED_MAX = 2.99  # Unexplored cells are near 0 (between -4 and +4)
        SEUIL_UNEXPLORED_MIN = -4.99
    
        # Masque des murs (high positive values)
        is_wall = (grid >= SEUIL_MUR)
        
        # Only cells with negative values are explored free space
        is_explored_free = (grid < SEUIL_FREE)
        
        #  Unexplored cells are near zero
        is_unexplored = (grid >= SEUIL_UNEXPLORED_MIN) & (grid <= SEUIL_UNEXPLORED_MAX)
        
        # Dilate les murs pour éviter les zones proches
        struct = np.ones((3, 3), dtype=bool)
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)
   


        # SOFT CONSTRAINT SETUP (Distance Map)
        # Calculate distance from every pixel to the nearest wall (IN CELLS)
        # This creates the "Gradient" that pushes the drone to the center.
        # We invert is_wall because we want distance to the WALLS.
        dist_map = ndimage.distance_transform_edt(~is_wall)

        # How far (in world units) do we want to be?
        COMFORT_DISTANCE_WORLD = 200.0 
        
        # Convert that to grid cells so we can compare with dist_map
        comfort_dist_cells = COMFORT_DISTANCE_WORLD / self.grid.resolution
        
        # Max penalty to apply if we are right next to the wall
        MAX_PENALTY = 100.0

        # Si explored_only=True, ajouter les zones non explorées à danger_zone
        if explored_only:
            danger_zone = danger_zone | (~is_explored_free)  # Block unexplored cells
        
        # Si le start ou le goal sont dans la danger_zone (par ex. drone collé au mur),
        # on autorise une petite zone autour d'eux pour permettre à A* de s'extraire.
        try:
            radius_clear = 2
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
                # FULL PATH RECONSTRUCTION
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # FIXED-STEP SUBSAMPLING
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
            
                # APPLY SMOOTHING
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


                # COST CALCULATION (Soft Constraints)
                base_cost = math.hypot(dx, dy)
                # Retrieve distance to nearest wall (in CELLS)
                dist_to_wall_cells = dist_map[neighbor[0], neighbor[1]]
                
                penalty = 0.0

                # Apply penalty if closer than comfort distance
                if dist_to_wall_cells < comfort_dist_cells:
                    proximity = 1.0 - (dist_to_wall_cells / comfort_dist_cells)
                    penalty = MAX_PENALTY * (proximity ** 2)
                
                move_cost = base_cost + penalty

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
            "status": "alive",
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
        if self.iteration % 5 == 0:
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
                str(self.identifier): np.array(self.target_point).tolist()
            }
    
        return message

    def control(self) -> CommandsDict:
        """
        Cerveau : Logique de test simplifiée.
        """

        # increment the iteration counter
        self.iteration += 1

        # INITIALIZE COMMAND HERE TO AVOID UNBOUNDLOCALERROR
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # Process received messages from other drones
        self.process_communication_sensor()

        # PERCEPTION
        self.update_pose()

        # RECORD HISTORY
        self.update_breadcrumbs()

         # Check if lidar is available before updating grid, if drone killed
        lidar_data = self.lidar_values()
        if lidar_data is None:
            # Drone is destroyed - cannot continue
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        

        # Update return area knowledge
        if getattr(self, 'is_inside_return_area', False):
            if len(self.return_area_points) < 3:
                self.add_return_area_point(self.current_pose[:2])
            
        # USE KALMAN-FILTERED POSE for grid update
        self.estimated_pose = Pose(np.asarray([self.current_pose[0], self.current_pose[1]]),
                                self.current_pose[2])
        self.grid.update_grid(pose=self.estimated_pose)

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
                print(f"[{self.identifier}] No barycenters available - using simple reverse")
                return {"forward": -0.5, "lateral": 0.3, "rotation": 0.4, "grasper": 1}
               

        # STATE MACHINE LOGIC
        # Transitions
        if self.state == self.Activity.EXPLORING:

            if self.grasper.grasped_wounded_persons:
                self.state = self.Activity.GOING_TO_RESCUE_CENTER

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
            
            if available_wounded and (self.iteration % 50 == 0):
                print(f"[{self.identifier}] [DEBUG] Available wounded: {available_wounded}")
                # Choose closest available wounded
                distances = [np.linalg.norm(np.array(w) - self.current_pose[:2]) for w in available_wounded]
                closest_idx = int(np.argmin(distances))
                closest_wounded = available_wounded[closest_idx]
                my_distance = distances[closest_idx]
                
                # Create a hashable key for this wounded
                wounded_key = (round(closest_wounded[0], 1), round(closest_wounded[1], 1))
                
                # Only check if this wounded has NEVER been evaluated
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
                            if other_dist < my_distance - 10.0:
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
                        

        elif self.state == self.Activity.GOING_TO_WOUNDED:
            grasped = getattr(self, "other_grasped_wounded", set())
            exclusion_radius = 20.0

            def is_near_grasped(w):
                return any(math.hypot(w[0] - gx, w[1] - gy) < exclusion_radius for (gx, gy) in grasped)

            # Check before grasping: if drone is near a grasped wounded, abort and explore
            for (gx, gy) in grasped:
                if math.hypot(self.current_pose[0] - gx, self.current_pose[1] - gy) < exclusion_radius:
                    self.state = self.Activity.EXPLORING
                    self.current_target_wounded = None
                    self.path = []

            # Replan every 30 iterations to adapt to updated wounded position
            if self.current_target_wounded is not None and (self.iteration % 30 == 0 or not self.path):
                # RECALCULATE PATH REGULARLY
                self.path = self.creer_chemin(self.current_pose[:2], self.current_target_wounded)
                self.last_replan_iteration = self.iteration

                    

            # Continuous conflict resolution
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
                            if other_dist < my_dist - 10.0:  # Larger margin during approach
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
                self.path = []
                
                
            else:
                # ENSURE SAFE RETURN: Only use explored areas
                if self.rescue_zone_points:
                   
                    # Replan with explored_only=True for safe return
                    should_replan = False
                    
                    # RECALCULATE PATH REGULARLY (every 30 iterations)
                    if self.iteration % 30 == 0:
                        should_replan = True
                    # retry if no path exists (and we haven't tried just recently)
                    elif (not self.path or len(self.path) == 0):
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
                            if len(self.breadcrumbs) > 4:
                                print("enter the loop")
                                # Reverse the recorded history
                                current_pos = self.current_pose[:2]
                                breadcrumbs_np = np.array(self.breadcrumbs)
                                dists = np.linalg.norm(breadcrumbs_np - current_pos, axis=1)
                                nearest_idx = int(np.argmin(dists))
                                nearest_crumb = self.breadcrumbs[nearest_idx]       
                                
                                # Convert to numpy and set as path
                                self.path = [np.array(nearest_crumb), np.array(self.rescue_zone_points[0])]
                                
                                # Append the actual rescue center at the end to be sure
                                self.path.append(np.array(self.rescue_zone_points[0]))
                                
                                print(f"[{self.identifier}] Success: Using Breadcrumbs (Length: {len(self.path)})")
                            else:
                                print(f"[{self.identifier}] No safe explored path to rescue center, trying unexplored areas!")
                                self.path = self.creer_chemin(
                                    self.current_pose[:2], 
                                    self.rescue_zone_points[0], 
                                    explored_only=False
                                )

        # STRATÉGIE
        # Replanification for exploration (only when in EXPLORING state)
        if self.state == self.Activity.EXPLORING:
            need_replan = False
            
            if not self.path or len(self.path) < 1:
                need_replan = True
            elif hasattr(self, 'target_point') and self.target_point is not None:
                dist_to_target = np.linalg.norm(self.target_point - self.current_pose[:2])
                if dist_to_target > 200.0:
                    if self.iteration % 100 == 0:
                        need_replan = True
                else:
                    need_replan = False

            if need_replan:
                # FUSION DES CIBLES (Locales + Partagées)
                local_frontiers = self.find_safe_frontier_points()
                shared_clusters = getattr(self, "shared_frontier_barycenters", [])
                
                # On crée une liste globale de candidats
                all_candidates = []
                
                # Ajout des partagés
                for bc in shared_clusters:
                    all_candidates.append({"point": np.array(bc), "source": "shared"})
                
                # Ajout des locaux s'ils ne font pas doublon (rayon 40px)
                for lf in local_frontiers:
                    if not any(np.linalg.norm(lf - c["point"]) < 40.0 for c in all_candidates):
                        all_candidates.append({"point": lf, "source": "local"})

                if all_candidates:
                    # CALCUL DU MEILLEUR SCORE
                    scored_targets = []
                    
                    # Récupération des positions des autres pour les pénalités
                    assigned_targets = {}
                    for msg in getattr(self.communicator, "received_messages", []):
                        other = msg[1] if isinstance(msg, tuple) else msg
                        other_assignments = other.get("assigned_barycenters", {})
                        for drone_id_str, target in other_assignments.items():
                            if int(drone_id_str) != self.identifier:
                                assigned_targets[int(drone_id_str)] = np.array(target)

                    for cand in all_candidates:
                        p = cand["point"]
                        distance = np.linalg.norm(p - self.current_pose[:2])
                        
                        # calcul de pénalité de conflit
                        conflict_penalty = 0.0
                        for other_target in assigned_targets.values():
                            if np.linalg.norm(p - other_target) < 300.0:
                                conflict_penalty += 10000.0

                        # Bonus de taille (uniquement si on a l'info en local)
                        size_bonus = 0.0
                        for cluster in self.frontier_clusters:
                            if np.linalg.norm(cluster["barycenter"] - p) < 20:
                                size_bonus = -cluster["size"] * 50.0
                                break
                        
                        score = distance + conflict_penalty + size_bonus
                        scored_targets.append({"point": p, "score": score})

                    # Tri par score (le plus petit est le meilleur)
                    scored_targets.sort(key=lambda x: x["score"])

                    # BOUCLE DE TENTATIVE (REPLI)
                    found_path = False
                    for target_info in scored_targets:
                        path = self.creer_chemin(self.current_pose[:2], target_info["point"])
                        if path:
                            self.target_point = target_info["point"]
                            self.path = path
                            found_path = True
                            break # Cible trouvée, on sort de la boucle
                    
                    if not found_path:
                        self.go_to_return_area(lidar_data)
                else:
                    self.go_to_return_area(lidar_data)


        # Generate movement commands based on current state
        if self.state == self.Activity.EXPLORING:
            gps_pos = self.measured_gps_position()
            has_gps = gps_pos is not None and not np.isnan(gps_pos[0])
            
            if not has_gps:
                command = self.follow_path(lidar_data)
            else:
                if self.path:   
                    command = self.follow_path(lidar_data)
                else:
                    self.go_to_return_area(lidar_data)
                

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

        elif self.state == self.Activity.GOING_TO_RETURN_AREA:
            # CONDITION DE SORTIE : Re-tenter l'exploration régulièrement
            if self.iteration % 30 == 0:  # Toutes les 3 secondes
                local_frontiers = self.find_safe_frontier_points()
                shared_clusters = getattr(self, "shared_frontier_barycenters", [])
                
                # S'il y a de nouveau des zones à explorer (les nôtres ou celles des autres)
                if local_frontiers or shared_clusters:
                    print(f"[{self.identifier}] Nouvelles frontières détectées ! Reprise de l'exploration.")
                    self.state = self.Activity.EXPLORING
                    self.path = [] # Forcer la planification au prochain cycle
                else:
                    # Si le chemin est fini ou invalide, on retente de cibler la zone
                    self.go_to_return_area(lidar_data)
            else:
                # Si le chemin est fini ou invalide, on retente de cibler la zone
                self.go_to_return_area(lidar_data)



        # GRASPER LOGIC
        # Grasper is ONLY active when going to wounded or rescue center
        if self.state == self.Activity.GOING_TO_WOUNDED or self.state == self.Activity.GOING_TO_RESCUE_CENTER:
            command["grasper"] = 1
        else:
            self.grasper._release_grasping()


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


        # Apply Drone Repulsion (Safety against other agents)
        # This will override/modify the command to push us away from collisions
        command = self.drone_repulsion(command)

        if self.iteration % 5 == 0:
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")

        return command

    def detect_semantic_entities(self):
        """Detects wounded and rescue centers using the semantic sensor."""
        try:
            detections = self.semantic_values()
        except Exception:
            detections = None

        if not detections:
            return

        dedup_radius = 60.0
        alpha_update = 0.3

        px, py, ptheta = float(self.current_pose[0]), float(self.current_pose[1]), float(self.current_pose[2])

        if not hasattr(self, 'other_grasped_wounded'):
            self.other_grasped_wounded = set()

        newly_seen_wounded = []
        newly_seen_rescue = []

        for data in detections:
            if not hasattr(data, "entity_type"):
                continue

            # Use the enum for robust type checking
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                global_angle = normalize_angle(ptheta + data.angle)
                xw = px + data.distance * math.cos(global_angle)
                yw = py + data.distance * math.sin(global_angle)
                is_grasped = any(
                    math.hypot(xw - gx, yw - gy) < dedup_radius
                    for (gx, gy) in self.other_grasped_wounded
                )
                if not is_grasped:
                    newly_seen_wounded.append((xw, yw))

            elif data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                global_angle = normalize_angle(ptheta + data.angle)
                xr = px + data.distance * math.cos(global_angle)
                yr = py + data.distance * math.sin(global_angle)
                newly_seen_rescue.append((xr, yr))

        # Merge newly seen wounded
        for nx, ny in newly_seen_wounded:
            merged = False
            for i, (wx, wy) in enumerate(self.wounded_to_rescue):
                if math.hypot(wx - nx, wy - ny) < dedup_radius:
                    newx = (1.0 - alpha_update) * wx + alpha_update * nx
                    newy = (1.0 - alpha_update) * wy + alpha_update * ny
                    self.wounded_to_rescue[i] = (newx, newy)
                    merged = True
                    break
            if not merged:
                self.wounded_to_rescue.append((nx, ny))

        # Merge newly seen rescue centers
        for nx, ny in newly_seen_rescue:
            self._add_or_merge_rescue_point((nx, ny))
            
        # DETECT DEAD DRONES
        self.detect_dead_drones(detections, px, py, ptheta)

    def detect_dead_drones(self, detections, px, py, ptheta):
        """
        Identify drones that are visible but not communicating and not moving.
        """
        # Collect positions of communicating "alive" drones
        alive_drones_positions = []
        for msg in getattr(self.communicator, "received_messages", []):
            if isinstance(msg, dict):
                 data = msg
            elif isinstance(msg, tuple):
                 data = msg[1]
            else:
                 continue
            
            p = data.get("drone_pose")
            if p is not None:
                alive_drones_positions.append(np.array(p[:2]))

        # Process Semantic Detections for Drones
        visible_drones = []
        for data in detections:
             if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                global_angle = normalize_angle(ptheta + data.angle)
                xd = px + data.distance * math.cos(global_angle)
                yd = py + data.distance * math.sin(global_angle)
                visible_drones.append(np.array([xd, yd]))
        
        # Filter visible drones
        for v_drone_pos in visible_drones:
            # Check if it corresponds to a communicating drone
            min_dist_alive = float('inf')
            if alive_drones_positions:
                dists = [np.linalg.norm(v_drone_pos - ap) for ap in alive_drones_positions]
                min_dist_alive = min(dists)

            if min_dist_alive < self.dead_drone_radius:
                continue

            # Check if it corresponds to an already known DEAD drone
            is_known_dead = False
            for dead_pos in self.dead_drones:
                if np.linalg.norm(v_drone_pos - np.array(dead_pos)) < self.dead_drone_radius:
                    is_known_dead = True
                    break
            
            if is_known_dead:
                continue
            
            # It is silent and unknown. Check suspects.
            matched_suspect = None
            for suspect in self.suspected_dead_drones:
                if np.linalg.norm(v_drone_pos - np.array(suspect['pos'])) < self.dead_drone_radius:
                    matched_suspect = suspect
                    break
            
            if matched_suspect:
                # Update suspect
                # Check for movement (if it moved too much from start, reset/remove)
                # But here we just update 'last_seen_pos' effectively by confirming it is still there.
                # Actually, we should check if the NEW position is close to the START position of the suspect.
                # If it drifted significantly, maybe it's moving slowly?
                # For now, we update last_seen_iter.
                
                dist_from_start = np.linalg.norm(v_drone_pos - np.array(matched_suspect['start_pos']))
                if dist_from_start > 50.0: # It moved > 50px since first seen
                    # It is moving, so not dead in the "static" sense. Remove from suspects.
                    print(f"[{self.identifier}] Suspect moved {dist_from_start:.1f}px - REMOVING")
                    self.suspected_dead_drones.remove(matched_suspect)
                else:
                    matched_suspect['last_seen_iter'] = self.iteration
                    matched_suspect['pos'] = (v_drone_pos[0], v_drone_pos[1]) # Update current pos estimate
                    
                    # Check confirmation condition
                    if (self.iteration - matched_suspect['first_seen_iter']) > self.dead_confirm_iterations:
                        # CONFIRMED DEAD
                        print(f"[{self.identifier}] *** CONFIRMED DEAD DRONE at ({matched_suspect['pos'][0]:.1f}, {matched_suspect['pos'][1]:.1f}) ***")
                        self.dead_drones.append(matched_suspect['pos'])
                        self.suspected_dead_drones.remove(matched_suspect)

                        # FORCE REPLANNING to avoid the newly discovered kill zone
                        print(f"[{self.identifier}] -> Clearing path to force replanning around kill zone.")
                        self.path = []
            
            else:
                # Create new suspect
                print(f"[{self.identifier}] VISIBLE DRONE WITHOUT RADIO SIGNAL at ({v_drone_pos[0]:.1f}, {v_drone_pos[1]:.1f})")
                print(f"[{self.identifier}] (Nearest radio signal: {min_dist_alive:.1f})")
                print(f"[{self.identifier}] ??? SUSPECTED DEAD DRONE initialized ???")
                self.suspected_dead_drones.append({
                    'pos': (v_drone_pos[0], v_drone_pos[1]),
                    'start_pos': (v_drone_pos[0], v_drone_pos[1]),
                    'first_seen_iter': self.iteration,
                    'last_seen_iter': self.iteration
                })
        
        # Cleanup suspects not seen for a while (e.g. they moved out of view)
        # We can keep them or drop them. If we lost visual, we can't confirm they are dead static.
        self.suspected_dead_drones = [
            s for s in self.suspected_dead_drones 
            if (self.iteration - s['last_seen_iter']) < 10
        ]

        # Cleanup Dead Drones that disappeared (False Positives correction)
        drones_to_remove = []
        for dead_pos in self.dead_drones:
            # Check if we are close enough to theoretically see it (approx range check)
            if np.linalg.norm(np.array(dead_pos) - np.array([px, py])) < 150.0:
                # Check if we actually see a drone at that position
                is_still_there = False
                for v_pos in visible_drones:
                    if np.linalg.norm(v_pos - np.array(dead_pos)) < self.dead_drone_radius:
                        is_still_there = True
                        break
                
                if not is_still_there:
                    print(f"[{self.identifier}] Dead drone at ({dead_pos[0]:.1f}, {dead_pos[1]:.1f}) disappeared - REMOVING KILL ZONE")
                    drones_to_remove.append(dead_pos)
        
        for d in drones_to_remove:
            self.dead_drones.remove(d)

        # Apply Virtual Walls for Confirmed Dead Drones (The Bubble)
        if self.dead_drones:
            grid_h, grid_w = self.grid.grid.shape
            radius_cells = int(90.0 / self.grid.resolution)
            val_wall = 100.0 # Very high value for virtual wall

            y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
            mask = x**2 + y**2 <= radius_cells**2

            for dx_world, dy_world in self.dead_drones:
                # Get grid indices
                res = self.grid._conv_world_to_grid(dx_world, dy_world)
                
                # Handling return type (can be float array or tuple)
                r_idx = int(res[0])
                c_idx = int(res[1])

                # Calculate bounds
                r_min = max(0, r_idx - radius_cells)
                r_max = min(grid_h, r_idx + radius_cells + 1)
                c_min = max(0, c_idx - radius_cells)
                c_max = min(grid_w, c_idx + radius_cells + 1)

                # Adjust mask for boundary clipping
                mask_r_min = r_min - (r_idx - radius_cells)
                mask_r_max = mask_r_min + (r_max - r_min)
                mask_c_min = c_min - (c_idx - radius_cells)
                mask_c_max = mask_c_min + (c_max - c_min)

                if r_max > r_min and c_max > c_min:
                    # Apply wall to grid
                    region = self.grid.grid[r_min:r_max, c_min:c_max]
                    region_mask = mask[mask_r_min:mask_r_max, mask_c_min:mask_c_max]
                    region[region_mask] = val_wall



    # --------------------------------------------------------------------------
    # FONCTION DE DÉTECTION DES FRONTIÈRES SÛRES 
    # --------------------------------------------------------------------------
    def find_safe_frontier_points(self) -> list:

            grid_map = self.grid.grid 
            
            # CALCUL DU GRADIENT OPTIMISÉ
            dx = ndimage.sobel(grid_map, axis=1)
            dy = ndimage.sobel(grid_map, axis=0)

            mag_sq = dx**2 + dy**2
            frontier_mask = (mag_sq > 25.0)

            # FILTRAGE VECTORISÉ (PAS DE BOUCLE) 
 
            is_wall = (grid_map >= 4.0) 
            danger_zone = binary_dilation(is_wall, iterations=2) # 2 itérations = ~20px
            
            is_unknown = (grid_map >= -2.0) & (grid_map <= 2.0)
            
            # Intersection : Gradient fort et zone sécurisée et proche de l'inconnu

            near_unknown = binary_dilation(is_unknown, iterations=1)
            frontier_mask &= (~danger_zone) & near_unknown

            # CLUSTERING EFFICACE

            labeled, num_features = ndimage.label(frontier_mask)
            
            if num_features == 0:
                return []

            # Utilisation de find_objects pour extraire les clusters sans itérer sur toute la grille
            slices = ndimage.find_objects(labeled)
            
            self.frontier_clusters = []
            min_cluster_size = 5
            
            # On itère seulement sur les bounding boxes des clusters trouvés
            for i, sl in enumerate(slices):
                if sl is None: continue
                
                # Extraction rapide du cluster
                cluster_mask = (labeled[sl] == (i + 1))
                size = np.sum(cluster_mask)
                
                if size >= min_cluster_size:
                    # Calcul direct des coordonnées moyennes dans la slice
                    coords = np.argwhere(cluster_mask)
                    mean_y = coords[:, 0].mean() + sl[0].start
                    mean_x = coords[:, 1].mean() + sl[1].start
                    
                    # Conversion monde
                    x_world, y_world = self.grid._conv_grid_to_world(mean_y, mean_x)
                    
                    self.frontier_clusters.append({
                        "barycenter": np.array([x_world, y_world]),
                        "size": int(size)
                    })

            # Retourne les barycentres pour la stratégie d'exploration
            return [c["barycenter"] for c in self.frontier_clusters]

    # --------------------------------------------------------------------------
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
                    arcade.draw_text("RZ", pt[0] + 12, pt[1] + 12, (0,120,0), 10)
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

    
        

        # DRAW THIS DRONE'S POSITION (for reference)
        my_screen_pos = self.current_pose[:2] + self._half_size_array
        arcade.draw_circle_filled(my_screen_pos[0], my_screen_pos[1], 
                                radius=18, color=detection_color)
        
    

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
            
        # Draw return area points
        try:
            if hasattr(self, 'return_area_points') and self.return_area_points:
                for (rx, ry) in self.return_area_points:
                    pt = np.array([rx, ry]) + self._half_size_array
                    arcade.draw_rectangle_outline(pt[0], pt[1], width=30, height=30, color=(0, 160, 255), border_width=2)
                    arcade.draw_text("RA", pt[0] + 12, pt[1] + 12, (0, 120, 200), 10)
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

        # TARGET SELECTION (Pure Pursuit style)
        lookahead_idx = 0
        for i, wp in enumerate(self.path):
            if np.linalg.norm(wp - self.current_pose[:2]) > lookahead_dist:
                lookahead_idx = i
                break
        target_pos = self.path[min(lookahead_idx, len(self.path)-1)]

        #  ERROR COMPUTATION (Body Frame)
        delta_pos = target_pos - self.current_pose[:2]
        heading = self.current_pose[2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        # Project world error into robot's local frame
        x_err = cos_h * delta_pos[0] + sin_h * delta_pos[1]   # Longitudinal (Forward/Back)
        y_err = -sin_h * delta_pos[0] + cos_h * delta_pos[1]  # Lateral (Left/Right)

        # ROTATION CONTROL (With Damping & Load Sensitivity)
        angle_error = normalize_angle(target_angle - heading)
        deriv_angle = angle_error - self.prev_angle_error
        
        # Base gains
        Kp_rot = self.Kp
        Kd_rot = self.Kd

        if abs(angle_error) < math.radians(10):
            Kp_rot *= 0.6
            Kd_rot *= 0.8

        rotation_speed = Kp_rot * angle_error + Kd_rot * deriv_angle
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error

        # LATERAL CONTROL (Cross-track Correction)
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
        # Calcul de la vitesse cible agressive
        max_speed = 22.0 
        # On utilise une accélération plus forte (0.25 au lieu de 0.15)
        target_speed = max(0.0, min(max_speed, x_err * 0.25 + 0.5))


        measured_vel = self.measured_velocity()
        if measured_vel is None:
            # No velocity data available (no GPS zone) - use target speed directly
            measured_speed = 0.0
        else:
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

        # Ralentissement en zone inconnue
        grid_pos = self.grid._conv_world_to_grid(*self.current_pose[:2])
        if grid_pos is not None:
            gy, gx = int(grid_pos[0]), int(grid_pos[1])
            if 0 <= gy < self.grid.grid.shape[0] and 0 <= gx < self.grid.grid.shape[1]:
                grid_val = self.grid.grid[gy, gx]
                # Valeur entre -4.99 et 2.99 = Inexploré
                if -4.99 <= grid_val <= 2.99:
                    forward_cmd *= 0.5  # On limite la poussée
                    lateral_cmd *= 0.6  # On stabilise les côtés

        forward_cmd = float(np.clip(forward_cmd, -1.0, 1.0))
        self.prev_speed_error = speed_error

        # WAYPOINT MANAGEMENT (Multi-drop logic)
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
        measured_vel = self.measured_velocity()
        lidar_data = self.lidar_values()
        
        # Calculate dt for Kalman filter
        current_time = self.iteration * 0.1  # Assuming 10 Hz
        if self.kf_last_time > 0:
            self.kf_dt = current_time - self.kf_last_time
        else:
            self.kf_dt = 0.1  # Default for first iteration
        self.kf_last_time = current_time

        # UPDATE HEADING
        if compass_angle is not None:
            self.current_pose[2] = compass_angle
        
        # GPS AVAILABLE: Use Kalman filter
        if gps_pos is not None and not np.isnan(gps_pos[0]):
            # Initialize filter on first GPS measurement
            if not self.kf_initialized:
                self.kf_state[0] = gps_pos[0]
                self.kf_state[1] = gps_pos[1]
                self.kf_state[2] = 0.0  # Initial velocity
                self.kf_state[3] = 0.0
                self.kf_initialized = True
            
            # Kalman Filter Prediction Step
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

            # Update velocity estimate if available
            if measured_vel is not None:
                self.kf_state[2] = 0.7 * measured_vel[0] + 0.3 * self.kf_state[2]
                self.kf_state[3] = 0.7 * measured_vel[1] + 0.3 * self.kf_state[3]

            
            # Kalman Filter Update Step
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

        # NO GPS ZONE: Use odometry for dead reckoning
        else:
            odom_data = self.odometer_values()
            if odom_data is None:
                return
            
            dist_travel = odom_data[0]  # Distance traveled
            alpha = odom_data[1]        # Relative angle of travel direction
            theta = odom_data[2]        # Change in orientation
            
            # Update heading first (using odometry rotation if compass unavailable)
            if compass_angle is None:
                self.current_pose[2] += theta
                self.current_pose[2] = normalize_angle(self.current_pose[2])
            
            # Get current heading (either from compass or updated odometry)
            heading = self.current_pose[2]
            
            # Calculate displacement in WORLD frame
            # The drone traveled 'dist_travel' distance at angle (heading + alpha)
            # alpha is the direction of travel relative to drone's orientation
            travel_direction = heading + alpha
            
            dx_world = dist_travel * math.cos(travel_direction)
            dy_world = dist_travel * math.sin(travel_direction)
            
            # Update position using odometry integration
            if self.kf_initialized:
                # Use Kalman prediction with odometry-based velocity estimate
                if self.kf_dt > 0:
                    vx_odom = dx_world / self.kf_dt
                    vy_odom = dy_world / self.kf_dt
                    
                    # Update velocity estimate (trust odometry heavily in no-GPS zones)
                    self.kf_state[2] = 0.8 * vx_odom + 0.2 * self.kf_state[2]
                    self.kf_state[3] = 0.8 * vy_odom + 0.2 * self.kf_state[3]
                
                # Predict position using velocity
                F = np.array([
                    [1, 0, self.kf_dt, 0],
                    [0, 1, 0, self.kf_dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                self.kf_state = F @ self.kf_state
                
                # Increase uncertainty significantly in no-GPS zones
                self.kf_P = F @ self.kf_P @ F.T + self.kf_Q * 1.2
                
                # Update pose
                self.current_pose[0] = self.kf_state[0]
                self.current_pose[1] = self.kf_state[1]
            else:
                # Kalman not initialized - pure odometry dead reckoning
                self.current_pose[0] += dx_world
                self.current_pose[1] += dy_world
                
                if self.iteration % 10 == 0:  # Print every 10 iterations to reduce spam
                    print(f"[{self.identifier}] Dead reckoning: dist={dist_travel:.1f}, alpha={math.degrees(alpha):.1f}°, "
                        f"theta={math.degrees(theta):.1f}°, heading={math.degrees(heading):.1f}°")

        # SLAM (SCAN MATCHING)

        gps_missing = (gps_pos is None or np.isnan(gps_pos[0]))
        has_lidar = (lidar_data is not None)
        # On ne fait le SLAM que tous les 5 pas (ex: toutes les 0.5 secondes)
        do_slam_now = (self.iteration % 15 == 0)

        # Condition : Pas de GPS, Lidar dispo, on a une carte (itération > 50) et c'est le moment
        if gps_missing and has_lidar and self.iteration > 50 and do_slam_now:
            
            # OPTIMISATION 1 : TRACKING MODE
            # Si le score précédent était bon (> 300 par exemple, dépend de la carte),
            # on réduit la zone de recherche pour aller vite.
            # Sinon, on cherche large pour se retrouver.
            if self.last_slam_score > 300: 
                current_radius = 20.0  # Tracking (2 pixels)
                angle_limit = 0.1      # ~6 deg
            else:
                current_radius = 40.0 # Recalage (4 pixels)
                angle_limit = 0.2     # ~12 deg

            current_guess = np.array([self.current_pose[0], self.current_pose[1], self.current_pose[2]])
            
            # Appel avec le rayon dynamique
            corrected_pose = self.run_scan_matching(current_guess, lidar_data, 
                                                  search_radius=current_radius, 
                                                  angle_search=angle_limit)
            
            # Calcul du nouveau score pour le prochain tour
            self.last_slam_score = self.calculate_scan_score(corrected_pose, lidar_data)

            # SECURITE 2 : GATING (ANTI-SAUT)
            # On calcule la distance entre la prédiction (Odométrie) et la correction (SLAM)
            dist_correction = math.sqrt((corrected_pose[0] - current_guess[0])**2 + 
                                        (corrected_pose[1] - current_guess[1])**2)
            
            # SEUIL DE REJET : Si le SLAM veut bouger le drone de plus de 20cm d'un coup,
            # c'est probablement une erreur (faux positif). On rejette.
            MAX_JUMP = 25.0 
            
            if dist_correction < MAX_JUMP:
                # La correction est crédible, on l'applique
                self.current_pose[0] = corrected_pose[0]
                self.current_pose[1] = corrected_pose[1]
                self.current_pose[2] = corrected_pose[2]
                
                # Feedback vers Kalman
                self.kf_state[0] = corrected_pose[0]
                self.kf_state[1] = corrected_pose[1]

    def calculate_scan_score(self, candidate_pose, lidar_distances):
        """
        Calcule la cohérence entre les mesures lidar et la carte pour une pose donnée.
        """
        score = 0
        lidar_angles = self.lidar().ray_angles
        distances = lidar_distances[::5]
        angles = lidar_angles[::5]
        
        x_r, y_r = candidate_pose[0], candidate_pose[1]
        theta_r = candidate_pose[2]
        
        global_angles = angles + theta_r
        x_points = x_r + distances * np.cos(global_angles)
        y_points = y_r + distances * np.sin(global_angles)
        
        max_range = MAX_RANGE_LIDAR_SENSOR
        
        for i in range(len(x_points)):
            dist = distances[i]
            if np.isnan(dist) or dist >= max_range: 
                continue 
            
            grid_pos = self.grid._conv_world_to_grid(x_points[i], y_points[i])
            gy, gx = int(grid_pos[0]), int(grid_pos[1])
            
            if 0 <= gy < self.grid.grid.shape[0] and 0 <= gx < self.grid.grid.shape[1]:
                score += self.grid.grid[gy, gx]
                
        return score

    def run_scan_matching(self, initial_pose, lidar_data, search_radius=10.0, angle_search=0.05):
        best_pose = np.copy(initial_pose)
        best_score = self.calculate_scan_score(best_pose, lidar_data)
        step_size = 5.0       # 0.5 pixel
        angle_step = 0.02     # ~1 deg
        for dx in np.arange(-search_radius, search_radius + 0.1, step_size):
            for dy in np.arange(-search_radius, search_radius + 0.1, step_size):
                for dtheta in np.arange(-angle_search, angle_search + 0.001, angle_step):
                    if dx == 0 and dy == 0 and dtheta == 0: continue
                    candidate = np.array([initial_pose[0] + dx, initial_pose[1] + dy, normalize_angle(initial_pose[2] + dtheta)])
                    score = self.calculate_scan_score(candidate, lidar_data)
                    if score > best_score:
                        best_score = score
                        best_pose = candidate
        return best_pose


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
        
            # Grid fusion (inside process_communication_sensor)
            if "grid_data" in other_message:
                # other_grid is the incoming data, self.grid.grid is our current data
                other_grid = np.array(other_message["grid_data"])
                
                # Fusion logic: Keep the value with the higher absolute confidence
                # This ensures that strong evidence (large |value|) overwrites weak evidence
                mask_update = np.abs(other_grid) > np.abs(self.grid.grid)
                self.grid.grid[mask_update] = other_grid[mask_update]

        # Store drone positions immediately (needed for avoidance)
        self.other_drones_positions = other_drones_positions
    
        # Add own frontier clusters for deduplication
        all_frontier_clusters.extend([
            {"barycenter": cluster["barycenter"].tolist()}
            for cluster in getattr(self, "frontier_clusters", [])
        ])
    
        # WOUNDED MANAGEMENT
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
    
        # ASSIGNMENTS
        for w, drone_id in all_assignments.items():
            if w not in self.wounded_assignments:
                self.wounded_assignments[w] = drone_id
    
        # GRASPED WOUNDED
        if not hasattr(self, "other_grasped_wounded"):
            self.other_grasped_wounded = set()
        self.other_grasped_wounded = all_grasped
    
        # RESCUE ZONES
        for r in all_rescue_zones:
            if r not in self.rescue_zone_points:
                self.rescue_zone_points.append(r)
    
        # FRONTIER DEDUPLICATION (ALL clusters, no limiting)
        if all_frontier_clusters:
            deduped_barycenters = []
            dedup_radius_frontier = 60.0
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
        struct = np.ones((3, 3), dtype=bool)
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

            # Check against ALL existing points
            for (rx, ry) in self.rescue_zone_points:
                dist = math.hypot(rx - nx, ry - ny)
                if dist < DEDUP_RADIUS:
                    # We already have a point roughly here. 
                    # Do NOT update it. Do NOT add a new one.
                    # Just return to keep the existing point stable.
                    return

            # If we are here, it is a completely NEW area.
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
            SAFE_DIST = 120.0  # Start pushing away at 70 pixels (approx 0.7 meter)
            GAIN = 3.5      # Strong push (Stronger than walls to prevent tangling)

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
                    # Magnitude: Stronger as we get closer (0.0 to 1.0)
                    force = (SAFE_DIST - dist) / SAFE_DIST
                    
                    # Direction: Global angle of the push
                    angle_global = math.atan2(dy, dx)
                    
                    # Local Projection: Convert to Forward/Lateral relative to head
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
    

    def go_to_wounded(self, lidar_data) -> CommandsDict:
        """
        Navigation vers le blessé.
        - Si distance > 20 pixels : utilise follow_path (suivi de chemin A*)
        - Si distance <= 20 pixels : s'oriente vers le blessé et avance rapidement (pas de ralentissement).
        """
        if self.current_target_wounded is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        dist_to_wounded = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])

        # Get closer before switching to direct approach
        if dist_to_wounded > 80.0:
            return self.follow_path(lidar_data)
        
        else:
            # Store wounded orientation just before grasping
            self.grasped_wounded_angle = self.get_wounded_orientation()


    # Direct approach: orient and move fast
        delta_pos = np.array(self.current_target_wounded) - self.current_pose[:2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])
        heading = self.current_pose[2]
        angle_error = normalize_angle(target_angle - heading)

        # PID Rotation
        deriv_error = angle_error - getattr(self, "prev_angle_error", 0.0)
        rotation_speed = self.Kp * angle_error + self.Kd * deriv_error
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error

        # Align more loosely if you want to be more aggressive
        ALIGNMENT_THRESHOLD = math.radians(5.0)

        if abs(angle_error) > ALIGNMENT_THRESHOLD:
            # Phase 1: Turn in place
            return {"forward": 0.0, "lateral": 0.0, "rotation": rotation_speed}
        else:
            # Phase 2: Go straight, fast
            return {"forward": 1.0, "lateral": 0.0, "rotation": rotation_speed}


    def get_wounded_orientation(self):
        try:
            detections = self.semantic_values()
            if not detections:
                return None

            for data in detections:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    angle = float(getattr(data, 'angle', 0.0))
                    return angle
                
        except Exception as e:
            print(f"[{self.identifier}] Error processing semantic data: {e}")
        return None
        



    def go_to_rescue_center_oriented(self, lidar_data) -> CommandsDict:
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        if not self.rescue_zone_points:
            return command

        # Vecteur vers le centre de secours
        target_point = np.array(self.rescue_zone_points[0])
        target_vector = target_point - self.current_pose[:2]
        dist_to_center = np.linalg.norm(target_vector)
        
        # ANGLE DE PRÉCISION
        # Angle global vers le centre
        angle_to_center = math.atan2(target_vector[1], target_vector[0])
        
        # On récupère l'angle stocké. S'il est None (cas d'erreur), on assume l'arrière (pi)
        grasp_angle = getattr(self, "grasped_wounded_angle", None)
        if grasp_angle is None:
            grasp_angle = math.pi
            
        # target_orientation est l'angle du drone tel que : 
        # drone_orientation + grasp_angle = angle_to_center
        target_orientation = normalize_angle(angle_to_center - grasp_angle)
        
        # Calcul de l'erreur d'angle
        angle_error = normalize_angle(target_orientation - self.current_pose[2])

        # LOGIQUE DE COMMANDE
        
        # Rotation : S'aligner sur l'axe du blessé
        if abs(angle_error) > 0.03:  # Plus de précision (2 degrés)
            command["rotation"] = np.clip(self.Kp * angle_error, -0.6, 0.6)
            command["forward"] = 1.0
        else:
            command["rotation"] = 0.0
            
            # Translation : Pousser le blessé vers le centre
            # On n'avance/recule que si l'alignement est quasi parfait
            if dist_to_center > 12.0:  # Distance d'arrêt ajustée
                # Si le blessé est plutôt devant (grasp_angle ~ 0), forward positif
                # Si le blessé est plutôt derrière (grasp_angle ~ pi), forward négatif
                direction = 1.0 if abs(grasp_angle) < math.pi/2 else -0.3
                command["forward"] = direction
            else:
                command["forward"] = 1.0

        return command
    

    def add_return_area_point(self, new_point):
        nx, ny = new_point
        # Radius to consider a point "already known"
        DEDUP_RADIUS = 50.0 
        # make distance from rescue center to avoid traffic jam near rescue center
        MIN_DIST_FROM_RESCUE = 170.0


        if hasattr(self, 'rescue_zone_points') and self.rescue_zone_points:
                    for (xr, yr) in self.rescue_zone_points:
                        dist_to_rescue = math.hypot(nx - xr, ny - yr)
                        # Si le point est trop proche d'un centre de secours, on l'ignore
                        if dist_to_rescue < MIN_DIST_FROM_RESCUE:
                            return
                        
        # Check against ALL existing points
        for (rx, ry) in self.return_area_points:
            dist = math.hypot(rx - nx, ry - ny)
            if dist < DEDUP_RADIUS:
                return
            

        # If we are here, it is a completely NEW area.
        self.return_area_points.append((nx, ny))


    def local_exploration_fallback(self):
        """
        Fallback: Move towards the nearest unexplored cell, or random walk if none found.
        """
        grid_map = self.grid.grid
        SEUIL_UNEXPLORED_MIN = -4.99
        SEUIL_UNEXPLORED_MAX = 4.0
        is_unexplored = (grid_map >= SEUIL_UNEXPLORED_MIN) & (grid_map <= SEUIL_UNEXPLORED_MAX)
        is_wall = (grid_map >= 4.01)
        unexplored_mask = is_unexplored & (~is_wall)
        if np.any(unexplored_mask):
            current_grid = self.grid._conv_world_to_grid(self.current_pose[0], self.current_pose[1])
            unexplored_indices = np.argwhere(unexplored_mask)
            distances = np.linalg.norm(unexplored_indices - current_grid, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_unexplored = unexplored_indices[nearest_idx]
            target_world = self.grid._conv_grid_to_world(nearest_unexplored[0], nearest_unexplored[1])
            path = self.creer_chemin(self.current_pose[:2], target_world)
            if path:
                self.path = path
                self.target_point = np.array(target_world)
                print(f"[{self.identifier}] [FALLBACK] Moving to nearest unexplored cell at {target_world}")
                return
            

    def go_to_return_area(self, lidar_data) -> CommandsDict:
        """
        Helper: Set target and path to go to the return area.
        """
        if self.return_area_points:
            target_index = int(self.identifier) % len(self.return_area_points)
            target_zone = self.return_area_points[target_index]
            self.target_point = target_zone
            self.path = self.creer_chemin(self.current_pose[:2], target_zone, explored_only=True)
            self.state = self.Activity.GOING_TO_RETURN_AREA

            return self.follow_path(lidar_data) if lidar_data is not None else {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        else:
            print(f"[{self.identifier}] No return area points available!")
            