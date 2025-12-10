import math
import numpy as np
from enum import Enum
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy import ndimage
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle, circular_mean
from swarm_rescue.simulation.drone.controller import CommandsDict
import arcade
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

    def creer_chemin(self, start_world, goal_world, explored_only=False):
        """
        Calcule un chemin entre start_world et goal_world en évitant les murs et les zones proches des murs (moins de 4 pixels) avec A*.
        Si explored_only=True, le chemin ne passera que par des zones explorées.
        Retourne une liste de points (en coordonnées monde).
        """
        import heapq
        from scipy.ndimage import binary_dilation
        grid = self.grid.grid
        # Conversion monde -> grille
        start = self.grid._conv_world_to_grid(*start_world)
        goal = self.grid._conv_world_to_grid(*goal_world)
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))

        # Masque des murs
        SEUIL_MUR = 5.0
        is_wall = (grid >= SEUIL_MUR)
        
        # Masque des zones explorées (valeurs différentes de 0)
        SEUIL_FREE = -10.0
        is_explored = (grid < SEUIL_FREE)  # zones explorées et libres
        
        # Dilate les murs pour éviter les zones proches (moins de 2 pixels)
        struct = np.ones((5, 5), dtype=bool)  # carré 5x5 ~ rayon 2
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)
        
        # Si explored_only=True, ajouter les zones non explorées à danger_zone
        if explored_only:
            danger_zone = danger_zone | (~is_explored)
        
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
                # Reconstruire le chemin (liste de tuples grille)
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                # Compresser le chemin: ne garder que les points où la direction change
                if len(path) <= 2:
                    compressed = path
                else:
                    compressed = [path[0]]
                    for i in range(1, len(path) - 1):
                        prev_v = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                        next_v = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                        if prev_v != next_v:
                            compressed.append(path[i])
                    compressed.append(path[-1])
                # Conversion grille -> monde et retour
                return [np.array(self.grid._conv_grid_to_world(*pt)) for pt in compressed]
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                # bounds check
                if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    continue

                # don't enter danger zone
                if danger_zone[neighbor]:
                    continue

                # Prevent corner-cutting: if moving diagonally, ensure adjacent orthogonal
                # cells are not blocked (i.e. allow diagonal only if there's space)
                if abs(dx) == 1 and abs(dy) == 1:
                    neigh1 = (current[0] + dx, current[1])
                    neigh2 = (current[0], current[1] + dy)
                    if (0 <= neigh1[0] < grid.shape[0] and 0 <= neigh1[1] < grid.shape[1]):
                        if danger_zone[neigh1]:
                            continue
                    if (0 <= neigh2[0] < grid.shape[0] and 0 <= neigh2[1] < grid.shape[1]):
                        if danger_zone[neigh2]:
                            continue

                # movement cost: diagonal sqrt(2), straight 1
                move_cost = math.hypot(dx, dy)
                tentative_g_score = gscore[current] + move_cost

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        # Pas de chemin trouvé
        return []
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
        # Le rayon d'inflation utilise maintenant la résolution de self.grid
        self.inflation_radius_cells = int(self.robot_radius_pixels / self.grid.resolution)
        if self.inflation_radius_cells < 1:
            self.inflation_radius_cells = 1
        
        # --- OBJECTIF FIXE (La Croix Rouge en bas à gauche) ---
        self.goal_position = np.array([-300.0, -200.0])

        # parametre PID rotation

        self.prev_angle_error = 0.0
        self.Kp = 3
        self.Kd = 2

        # PID translation
        self.prev_diff_position = np.zeros(2)  # dérivée pour translation
        self.Kp_pos = 1.6
        self.Kd_pos = 11.0
        # Pour le PID de vitesse
        self.prev_speed_error = 0.0

        self.path = []
        self.frontiers_world = []

        # `wounded_to_rescue`: list of (x,y) tuples for detected wounded persons
        # `rescue_zone_points`: list of (x,y) tuples representing detected rescue area points
        self.wounded_to_rescue = [(0,0)]
        self.rescue_zone_points = []

        # internal metadata to remember when a detection was last seen
        # keys are rounded tuples (x,y) -> last seen iteration
        self._wounded_memory_meta = {}
        self._rescue_memory_meta = {}
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

        #deblocage blessé face au rescue center
        self.stuck_timer = 0
        self.last_pos = None

        if not hasattr(self, "rescued_wounded"):
            self.rescued_wounded = []

        self.wounded_assignments = {}  # {wounded_pos: drone_id}

        self.received_messages = []

    def define_message_for_all(self) -> None:
        self.message = {
            "drone_id": self.identifier,  
            "drone_pose": self.current_pose.tolist(),
            "wounded_list": self.wounded_to_rescue,
            "rescue_list": self.rescue_zone_points,
            "state": self.state.name,
            "rescued_wounded": getattr(self, "rescued_wounded", []),
            "wounded_assignments": self.wounded_assignments,
            "grasped_wounded": [
                (w.position[0], w.position[1]) if hasattr(w, "position") else None
                for w in getattr(self.grasper, "grasped_wounded_persons", [])
            ]
        }

    def control(self) -> CommandsDict:
        """
        Cerveau : Logique de test simplifiée.
        """

        # increment the iteration counter
        self.iteration += 1

        # Process received messages from other drones
        for msg in self.received_messages:
            print(f"[DEBUG] Drone {self.identifier} received message: {msg}")  
            self.merge_wounded_list(msg)

        # --- 1. PERCEPTION ---
        self.update_pose()
        
        # Mise à jour de la grille probabiliste self.grid.grid (utilisée pour l'exploration)
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())
        self.grid.update_grid(pose=self.estimated_pose) # Mise à jour de la carte utilisée!
        #print("grid :", self.grid.grid > 0.0)

        lidar_data = self.lidar_values()
        if lidar_data is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # Also populate the simpler public lists requested by the user
        try:
            self.detect_semantic_entities()
        except Exception:
            pass


        # STATE MACHINE LOGIC
        # Transitions
        if self.state == self.Activity.EXPLORING:
            # Only consider wounded not already assigned or grasped
            grasped = getattr(self, "other_grasped_wounded", set())
            available_wounded = [
                w for w in self.wounded_to_rescue
                if w not in self.wounded_assignments and w not in grasped
            ]
            if available_wounded:
                # Choose closest available wounded
                distances = [np.linalg.norm(np.array(w) - self.current_pose[:2]) for w in available_wounded]
                closest_idx = int(np.argmin(distances))
                self.current_target_wounded = available_wounded[closest_idx]
                # Assign this wounded to self (first-come, first-served)
                self.wounded_assignments[self.current_target_wounded] = self.identifier
                self.state = self.Activity.GOING_TO_WOUNDED
                self.path = self.creer_chemin(self.current_pose[:2], self.current_target_wounded)
                self.last_replan_iteration = self.iteration

        elif self.state == self.Activity.GOING_TO_WOUNDED:
            if self.grasper.grasped_wounded_persons:
                # Successfully grasped, go to rescue center
                # Mark as rescued
                if self.current_target_wounded is not None:
                    self.rescued_wounded.append(self.current_target_wounded)
                self.state = self.Activity.GOING_TO_RESCUE_CENTER
                
                # Remove the grasped wounded from the list
                if self.current_target_wounded is not None:
                    # Find and remove the wounded within check_radius
                    check_radius = 50.0
                    self.wounded_to_rescue = [
                        (wx, wy) for (wx, wy) in self.wounded_to_rescue
                        if math.hypot(self.current_target_wounded[0] - wx, self.current_target_wounded[1] - wy) >= check_radius
                    ]
                    # Also clean up metadata
                    def _key_of(pt):
                        return (round(float(pt[0]), 1), round(float(pt[1]), 1))
                    if self.current_target_wounded:
                        key = _key_of(self.current_target_wounded)
                        self._wounded_memory_meta.pop(key, None)
                
                if self.rescue_zone_points:
                    self.path = self.creer_chemin(self.current_pose[:2], self.rescue_zone_points[0], explored_only=True)
                    self.last_replan_iteration = self.iteration


            elif self.current_target_wounded is not None:
                    # Check if we've arrived at the target location
                    distance_to_target = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])
                    
                    if distance_to_target < 30.0:  # Within 30 units of target
                        
                        # Check if wounded is currently detected by semantic sensor
                        wounded_detected = False
                        detection_radius = 50.0
                        
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
                        
                        print(f"Wounded detected: {wounded_detected}")
                        
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
                            
                            # Clean up metadata
                            def _key_of(pt):
                                return (round(float(pt[0]), 1), round(float(pt[1]), 1))
                            key = _key_of(self.current_target_wounded)
                            self._wounded_memory_meta.pop(key, None)
                            
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
                self.stuck_timer = 0


            else:
                distance_to_rescue = np.linalg.norm(np.array(self.rescue_zone_points[0]) - self.current_pose[:2])
                self.last_pos = self.current_pose[:2].copy()
                # Near rescue center - check if stuck
                if distance_to_rescue < 10.0:
                    if self.last_pos is not None:
                        movement = np.linalg.norm(self.current_pose[:2] - self.last_pos)
                        if movement < 3.0:
                            self.stuck_timer += 1
                        else:
                            self.stuck_timer = 0
                    self.last_pos = self.current_pose[:2].copy()
                    
                    # Rotate if stuck
                    while self.stuck_timer > 15 and self.stuck_timer <25:
                        command = {"forward": -1.0, "lateral": 0.0, "rotation": 0.5}
                        command["grasper"] = 1
                        self.stuck_timer += 1
                        return command 

        # --- 2. STRATÉGIE ---
        # Replanification for exploration (only when in EXPLORING state)
        if self.state == self.Activity.EXPLORING:
            need_replan = False
            if not self.path or len(self.path) < 1:
                need_replan = True
            # Met à jour le chemin tous les 10 itérations (si cible existante)
            if self.path and hasattr(self, 'target_point') and self.iteration % 10 == 0:
                need_replan = True

            if need_replan:
                self.frontiers_world = self.find_safe_frontier_points()
                if self.frontiers_world:
                    # Scoring simplifié : choisir le point de frontière le plus proche
                    distances = [np.linalg.norm(f - self.current_pose[:2]) for f in self.frontiers_world]
                    closest_index = np.argmin(distances)
                    target_point = self.frontiers_world[closest_index]
                    self.target_point = target_point
                    self.path = self.creer_chemin(self.current_pose[:2], target_point)
        
        # --- 3. ACTION (Le "Pilote") ---
        # Execute commands based on state
        if self.state == self.Activity.EXPLORING:
            # Continue exploration
            if self.path:
                command = self.follow_path(lidar_data)
            else:
                print("Exploration terminée ou pas de frontieres sûres trouvées.")
                command = {"forward": 0.5, "lateral": 0.0, "rotation": 0.2}

        elif self.state == self.Activity.GOING_TO_WOUNDED:
            # Navigate to wounded
            if self.current_target_wounded:
                # Only replan if path is empty AND enough iterations have passed (avoid constant replanning)
                # OR if this is the first time (last_replan_iteration == 0)
                should_replan = False
                if not self.path or len(self.path) == 0:
                    iterations_since_replan = self.iteration - self.last_replan_iteration
                    if iterations_since_replan >= 30 or self.last_replan_iteration == 0:
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
            # Navigate to rescue center
            if self.rescue_zone_points:
                # Only replan if path is empty AND enough iterations have passed
                should_replan = False
                if not self.path or len(self.path) == 0:
                    iterations_since_replan = self.iteration - self.last_replan_iteration
                    if iterations_since_replan >= 30 or self.last_replan_iteration == 0:
                        should_replan = True
                
                if should_replan:
                    self.path = self.creer_chemin(self.current_pose[:2], self.rescue_zone_points[0], explored_only=True)
                    self.last_replan_iteration = self.iteration
                
                if self.path:
                    command = self.follow_path(lidar_data)
                else:
                    command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
            else:
                # No rescue center known, explore
                if self.path:
                    command = self.follow_path(lidar_data)
                else:
                    command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        else:
            command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        command["grasper"] = 1

        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid,
                              self.estimated_pose,
                              title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")

        return command

    def detect_semantic_entities(self):
        """Populate `self.wounded_to_rescue` and `self.rescue_zone_points`.

        Converts semantic detections into world coordinates and fills two
        de-duplicated lists for user convenience. Keeps detections in memory
        for a while when they go out of view.
        """
        try:
            detections = self.semantic_values()
        except Exception:
            detections = None

        # parameters
        dedup_radius = 40.0         # close detections considered same
        alpha_update = 0.3          # mixing factor when updating an existing entry
        memory_max_age = 50000        # iterations before forgetting an unseen entry

        # helper to round keys to reduce float-key issues
        def _key_of(pt):
            return (round(float(pt[0]), 1), round(float(pt[1]), 1))

        # Ensure memory dicts exist
        if not hasattr(self, '_wounded_memory_meta'):
            self._wounded_memory_meta = {}
        if not hasattr(self, '_rescue_memory_meta'):
            self._rescue_memory_meta = {}

        # Use local lists to accumulate new detections
        newly_seen_wounded = []
        newly_seen_rescue = []

        if detections:
            px = float(self.current_pose[0])
            py = float(self.current_pose[1])
            ptheta = float(self.current_pose[2])

            for data in detections:
                try:
                    etype = getattr(data, 'entity_type', None)
                    angle = float(getattr(data, 'angle', 0.0))
                    dist = float(getattr(data, 'distance', 0.0))
                except Exception:
                    continue

                # Convert detection to world coordinates
                global_angle = normalize_angle(ptheta + angle)
                xw = px + dist * math.cos(global_angle)
                yw = py + dist * math.sin(global_angle)

                try:
                    name = etype.name if hasattr(etype, 'name') else str(etype)
                except Exception:
                    name = str(etype)

                # This is where wounded persons are detected and added
                if 'WOUNDED' in name.upper():
                    newly_seen_wounded.append((xw, yw))
                elif 'RESCUE' in name.upper():
                    newly_seen_rescue.append((xw, yw))

        # Initialize lists if missing
        if not hasattr(self, 'wounded_to_rescue'):
            self.wounded_to_rescue = []
        if not hasattr(self, 'rescue_zone_points'):
            self.rescue_zone_points = []

        # Merge newly seen wounded into memory (update existing if near, else append)
        for nx, ny in newly_seen_wounded:
            merged = False
            for i, (wx, wy) in enumerate(self.wounded_to_rescue):
                if math.hypot(wx - nx, wy - ny) < dedup_radius:
                    # weighted update
                    newx = (1.0 - alpha_update) * wx + alpha_update * nx
                    newy = (1.0 - alpha_update) * wy + alpha_update * ny
                    self.wounded_to_rescue[i] = (newx, newy)
                    self._wounded_memory_meta[_key_of(self.wounded_to_rescue[i])] = self.iteration
                    merged = True
                    break
            if not merged:
                pt = (nx, ny)
                self.wounded_to_rescue.append(pt)
                self._wounded_memory_meta[_key_of(pt)] = self.iteration

        # Merge newly seen rescue points
        for nx, ny in newly_seen_rescue:
            merged = False
            for i, (rx, ry) in enumerate(self.rescue_zone_points):
                if math.hypot(rx - nx, ry - ny) < dedup_radius:
                    newx = (1.0 - alpha_update) * rx + alpha_update * nx
                    newy = (1.0 - alpha_update) * ry + alpha_update * ny
                    self.rescue_zone_points[i] = (newx, newy)
                    self._rescue_memory_meta[_key_of(self.rescue_zone_points[i])] = self.iteration
                    merged = True
                    break
            if not merged:
                pt = (nx, ny)
                self.rescue_zone_points.append(pt)
                self._rescue_memory_meta[_key_of(pt)] = self.iteration

        # If no fresh detections, optionally refresh from semantic_tracks (stable tracks)
        if not newly_seen_wounded:
            for tr in getattr(self, 'semantic_tracks', []):
                if tr.get('type') == 'WOUNDED' and tr.get('count', 0) >= 2:
                    pt = tuple(tr['pos'].tolist())
                    # merge as above
                    merged = False
                    for i, (wx, wy) in enumerate(self.wounded_to_rescue):
                        if math.hypot(wx - pt[0], wy - pt[1]) < dedup_radius:
                            newx = (1.0 - alpha_update) * wx + alpha_update * pt[0]
                            newy = (1.0 - alpha_update) * wy + alpha_update * pt[1]
                            self.wounded_to_rescue[i] = (newx, newy)
                            self._wounded_memory_meta[_key_of(self.wounded_to_rescue[i])] = tr.get('last_seen', self.iteration)
                            merged = True
                            break
                    if not merged:
                        self.wounded_to_rescue.append(pt)
                        self._wounded_memory_meta[_key_of(pt)] = tr.get('last_seen', self.iteration)

        if not newly_seen_rescue:
            rescue_tracks = [t for t in getattr(self, 'semantic_tracks', []) if t.get('type') == 'RESCUE']
            for tr in rescue_tracks:
                pt = tuple(tr['pos'].tolist())
                merged = False
                for i, (rx, ry) in enumerate(self.rescue_zone_points):
                    if math.hypot(rx - pt[0], ry - pt[1]) < dedup_radius:
                        newx = (1.0 - alpha_update) * rx + alpha_update * pt[0]
                        newy = (1.0 - alpha_update) * ry + alpha_update * pt[1]
                        self.rescue_zone_points[i] = (newx, newy)
                        self._rescue_memory_meta[_key_of(self.rescue_zone_points[i])] = tr.get('last_seen', self.iteration)
                        merged = True
                        break
                if not merged:
                    self.rescue_zone_points.append(pt)
                    self._rescue_memory_meta[_key_of(pt)] = tr.get('last_seen', self.iteration)

        # Prune old memory entries
        def _prune_memory(lst, meta):
            to_keep = []
            new_meta = {}
            for pt in lst:
                k = _key_of(pt)
                last = meta.get(k, None)
                if last is None:
                    # keep if recently added (use current iteration)
                    last = self.iteration
                age = self.iteration - last
                if age <= memory_max_age:
                    to_keep.append(pt)
                    new_meta[k] = last
            return to_keep, new_meta

        self.wounded_to_rescue, self._wounded_memory_meta = _prune_memory(self.wounded_to_rescue, self._wounded_memory_meta)
        self.rescue_zone_points, self._rescue_memory_meta = _prune_memory(self.rescue_zone_points, self._rescue_memory_meta)

        # Reduce rescue_zone_points to a single barycenter (if any)
        if self.rescue_zone_points:
            xs = [p[0] for p in self.rescue_zone_points]
            ys = [p[1] for p in self.rescue_zone_points]
            bx = float(sum(xs) / len(xs))
            by = float(sum(ys) / len(ys))
            self.rescue_zone_points = [(bx, by)]
            # update metadata to mark barycenter as last seen now
            self._rescue_memory_meta = {_key_of((bx, by)): self.iteration}

    # --------------------------------------------------------------------------
    # FONCTION DE DÉTECTION DES FRONTIÈRES SÛRES (Mise à jour pour self.frontiers_world)
    # --------------------------------------------------------------------------

    def find_safe_frontier_points(self) -> list:
        
        grid_map = self.grid.grid 
        
        # --- NOUVEAUX SEUILS ADAPTÉS À L'INITIALISATION À ZÉRO ---
        # Si grid.grid est initialisée à 0.0:
        
        SEUIL_FREE = -10.0
        SEUIL_MUR = 5.0        
        
        frontiers_world_temp = []
        frontiers = []

        # 1. Identifier les masques
        # is_unknown: Les cellules dont la valeur est très proche de zéro (état initial)
        #is_unknown = (grid_map < SEUIL_MUR) & (grid_map > SEUIL_FREE)
        is_unknown = (grid_map <= 10) & (grid_map >= -10)
        #print("u :", is_unknown)
        
        # is_wall: Les cellules où la probabilité d'occupation est élevée
        is_wall = (grid_map >= SEUIL_MUR)  
        
        # (grid diagnostics removed)
        
        # is_free: Les cellules qui ont été balayées et ne sont pas des murs (0.0 < valeur < 0.6)
        is_free = (grid_map < SEUIL_FREE) 
        #print("f :", is_free) 

        # ---------------------------------------------------------
        # ÉTAPE A : DÉTECTION DES FRONTIÈRES BRUTES (Voisinage 4)
        # ---------------------------------------------------------
        frontier_mask = np.zeros_like(is_free, dtype=bool)

        # connectivity structure
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]], dtype=bool)

        # Dilate is_unknown to mark neighbors
        unknown_neighbors = binary_dilation(is_unknown, structure=structure)
        frontier_mask = is_free & unknown_neighbors

        # frontier mask computed

        # Structure 8-connectée pour la dilatation
        struct = np.ones((5, 5), dtype=bool)
        # Dilate is_wall by 2 pixels
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=2)
        # Remove frontiers too close to a wall
        frontier_mask = frontier_mask & (~danger_zone)
        safe_frontiers_mask = frontier_mask

        # safe_frontiers_mask ready

        # Regrouper les cellules frontier contiguës en composants connexes (voisinage 4)
        # et retourner le barycentre (en coordonnées monde) de chaque composant.
        # safe_frontiers_mask est un tableau booléen de forme (nx, ny)
        structure = generate_binary_structure(2, 2)  # 8-connectivité
        labeled, num_features = ndimage.label(safe_frontiers_mask, structure=structure)

        self.frontier_clusters = []
        min_cluster_size = 4
        for label_idx in range(1, num_features + 1):
            ys, xs = np.where(labeled == label_idx)
            size = ys.size
            if size == 0:
                continue
            if size < min_cluster_size:
                continue
            # barycentre en indices grille (float)
            mean_x = float(np.mean(ys))
            mean_y = float(np.mean(xs))
            # conversion grille -> monde (barycentre)
            x_world, y_world = self.grid._conv_grid_to_world(mean_x, mean_y)
            # conversion des cellules du cluster en coordonnées monde
            x_cells_world, y_cells_world = self.grid._conv_grid_to_world(ys, xs)
            cells_world = np.vstack((x_cells_world, y_cells_world)).T
            frontiers.append(np.array([x_world, y_world]))
            self.frontier_clusters.append({
                "cells_world": cells_world,
                "barycenter": np.array([x_world, y_world]),
                "size": int(size)
            })

        return frontiers

    # FONCTION DE DESSIN
    # --------------------------------------------------------------------------
    
    def draw_bottom_layer(self):
        """ Dessine le chemin calculé (tous les points) """
        # --- DEBUG VISUALISATION DES FRONTIERES (START) ---
        # Ces lignes de dessin servent uniquement à visualiser les clusters
        # de frontières et leurs barycentres. Elles sont clairement marquées
        # pour pouvoir être retirées facilement.
        if hasattr(self, 'frontier_clusters') and self.frontier_clusters:
            # dessin des cellules de chaque cluster (petits points rouges)
            for cluster in self.frontier_clusters:
                cells = cluster.get('cells_world')
                if cells is None or len(cells) == 0:
                    continue
                for c in cells:
                    pt = c + self._half_size_array
                    arcade.draw_circle_filled(pt[0], pt[1], radius=3, color=(255,0,0))
                # barycentre (jaune)
                bc = cluster.get('barycenter')
                ptb = bc + self._half_size_array
                arcade.draw_circle_filled(ptb[0], ptb[1], radius=6, color=(255,255,0))
        # --- DEBUG VISUALISATION DES FRONTIERES (END) ---

        # Draw wounded detected via new simple API (wounded_to_rescue)
        try:
            if hasattr(self, 'wounded_to_rescue') and self.wounded_to_rescue:
                for (xw, yw) in self.wounded_to_rescue:
                    pt = np.array([xw, yw]) + self._half_size_array
                    # visible marker (outline + small filled center)
                    arcade.draw_circle_outline(pt[0], pt[1], radius=18, color=(200,60,60), border_width=2)
                    arcade.draw_circle_filled(pt[0], pt[1], radius=4, color=(255,80,80))
                    arcade.draw_text("det", pt[0] + 14, pt[1] + 14, (200, 60, 60), 10)
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

        self.display_pose()
        
        # Display state machine status above drone
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
            print("No path to follow.")
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        target_pos = self.path[0]
        delta_pos = target_pos - self.current_pose[:2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])

        # --- PID sur la rotation (inchangé) ---
        angle_error = normalize_angle(target_angle - self.current_pose[2])
        deriv_error = angle_error - self.prev_angle_error
        rotation_speed = self.Kp * angle_error + self.Kd * deriv_error
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error

        distance_to_target = np.linalg.norm(delta_pos)

        # --- Profil de vitesse souhaitée (tunable) ---
        # Augmenter max_speed pour rendre le drone plus rapide
        max_speed = 10.0
        # profil proportionnel à la distance, coef augmenté pour plus de vitesse
        target_speed = max(0.0, min(max_speed, distance_to_target * 0.12 + 0.3))

        measured_vel = self.measured_velocity()
        measured_speed = math.sqrt(measured_vel[0] ** 2 + measured_vel[1] ** 2)

        # --- PID sur la vitesse (Kp_pos, Kd_pos) ---
        # erreur = consigne - mesuré
        speed_error = target_speed - measured_speed
        deriv_speed = speed_error - self.prev_speed_error
        # commande continue - peut être négative si il faut freiner
        forward_cmd = self.Kp_pos * speed_error + self.Kd_pos * deriv_speed
        # si on est en train de fortement tourner, ne pas avancer
        if abs(angle_error) > 0.2:
            forward_cmd = 0.0
        # normaliser la commande dans [-1, 1]
        forward_cmd = float(np.clip(forward_cmd, -1.0, 1.0))
        self.prev_speed_error = speed_error

        # Si proche du waypoint → passer au suivant (seuil réduit pour précision)
        if distance_to_target < 30:
        # on arrête le mouvement frontal pour éviter dépassement
            forward_cmd = 0.0
            self.path.pop(0)

        # Si proche du but final
        dist_to_goal = np.linalg.norm(self.goal_position - self.current_pose[:2])

        return {"forward": forward_cmd, "lateral": 0.0, "rotation": rotation_speed}

    def display_pose(self) :
        radius = 10
        red = (0,0,255)
        green = (0,255,0)

        current_pose = self.current_pose[:2] + self._half_size_array
        arcade.draw_circle_filled(current_pose[0],
                              current_pose[1],
                              radius=radius,
                              color=red)

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

    def merge_wounded_list(self, other_message):
        """
        Merge wounded list from another drone's message.
        Adds new wounded positions not already in self.wounded_to_rescue.
        """
        if "wounded_list" not in other_message:
            return
        dedup_radius = 60
        # Add new wounded
        for w in other_message["wounded_list"]:
            if all(math.hypot(w[0] - wx, w[1] - wy) > dedup_radius for (wx, wy) in self.wounded_to_rescue):
                self.wounded_to_rescue.append(tuple(w))
        # Remove rescued wounded shared by other drone
        for rw in other_message.get("rescued_wounded", []):
            self.wounded_to_rescue = [
                (wx, wy) for (wx, wy) in self.wounded_to_rescue
                if math.hypot(rw[0] - wx, rw[1] - wy) > dedup_radius
            ]
        # Merge assignments (first-come, first-served: keep existing assignment)
        for w, drone_id in other_message.get("wounded_assignments", {}).items():
            if w not in self.wounded_assignments:
                self.wounded_assignments[w] = drone_id
        # Collect grasped wounded from other drones
        if not hasattr(self, "other_grasped_wounded"):
            self.other_grasped_wounded = set()
        for w in other_message.get("grasped_wounded", []):
            if w is not None:
                self.other_grasped_wounded.add(tuple(w))

        # Deduplicate wounded list
        deduped = []
        for w in self.wounded_to_rescue:
            if all(math.hypot(w[0] - wx, w[1] - wy) > dedup_radius for (wx, wy) in deduped):
                deduped.append(w)
        self.wounded_to_rescue = deduped

        # Add new rescue zones
        if "rescue_list" in other_message:
            for r in other_message["rescue_list"]:
                if r not in self.rescue_zone_points:
                    self.rescue_zone_points.append(r)

        
        print(f"[DEBUG] Drone {self.identifier} messages at start: {self.message}")

