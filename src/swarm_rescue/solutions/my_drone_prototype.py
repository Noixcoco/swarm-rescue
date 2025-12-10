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
        self.wounded_to_rescue = []
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

    def define_message_for_all(self) -> None:
        pass 

    def control(self) -> CommandsDict:
        """
        Cerveau : Logique de test simplifiée.
        """

        # increment the iteration counter
        self.iteration += 1
        
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
            if self.wounded_to_rescue:
                # Choose closest wounded
                distances = [np.linalg.norm(np.array(w) - self.current_pose[:2]) for w in self.wounded_to_rescue]
                closest_idx = int(np.argmin(distances))
                self.current_target_wounded = self.wounded_to_rescue[closest_idx]
                self.state = self.Activity.GOING_TO_WOUNDED
                self.path = self.creer_chemin(self.current_pose[:2], self.current_target_wounded)
                self.last_replan_iteration = self.iteration

        elif self.state == self.Activity.GOING_TO_WOUNDED:
            if self.grasper.grasped_wounded_persons:
                # Successfully grasped, go to rescue center
                self.state = self.Activity.GOING_TO_RESCUE_CENTER
                
                # Remove the grasped wounded from the list
                if self.current_target_wounded is not None:
                    # Find and remove the wounded within check_radius
                    check_radius = 30.0
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
                # Check distance to target
                distance_to_target = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])
                
                # If we're close enough, verify the wounded is actually here via semantic sensor
                if distance_to_target < 40.0:
                    wounded_detected = False
                    detection_radius = 60.0
                    
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
                                        
                                        dist_to_detected = math.hypot(self.current_target_wounded[0] - xw, 
                                                                      self.current_target_wounded[1] - yw)
                                        
                                        if dist_to_detected < detection_radius:
                                            wounded_detected = True
                                            break
                                except Exception:
                                    continue
                    except Exception:
                        pass
                    
                    # If wounded not detected on site, remove from list and return to exploring
                    if not wounded_detected:
                        print(f"WARNING: Wounded not found at expected location {self.current_target_wounded}. Removing from list.")
                        check_radius = 50.0
                        self.wounded_to_rescue = [
                            (wx, wy) for (wx, wy) in self.wounded_to_rescue
                            if math.hypot(self.current_target_wounded[0] - wx, 
                                        self.current_target_wounded[1] - wy) > check_radius
                        ]
                        
                        # Clean up metadata
                        def _key_of(pt):
                            return (round(float(pt[0]), 1), round(float(pt[1]), 1))
                        key = _key_of(self.current_target_wounded)
                        self._wounded_memory_meta.pop(key, None)
                        
                        # Return to exploring
                        self.state = self.Activity.EXPLORING
                        self.current_target_wounded = None
                        self.path = []
                elif self.current_target_wounded is not None:
                    # Still far from target, check if target still exists in wounded list (in case another drone rescued it)
                    target_still_exists = False
                    check_radius = 30.0  # tolerance radius for target identification
                    for (wx, wy) in self.wounded_to_rescue:
                        if math.hypot(self.current_target_wounded[0] - wx, self.current_target_wounded[1] - wy) < check_radius:
                            target_still_exists = True
                            break
                    if not target_still_exists:
                        # Target truly lost from list, return to exploring
                        self.state = self.Activity.EXPLORING
                        self.current_target_wounded = None
            else:
                # No target defined, return to exploring
                self.state = self.Activity.EXPLORING
                self.current_target_wounded = None

        elif self.state == self.Activity.GOING_TO_RESCUE_CENTER:
            if not self.grasper.grasped_wounded_persons:
                # Dropped wounded, return to exploring
                self.state = self.Activity.EXPLORING
                self.current_target_wounded = None

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
                command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        elif self.state == self.Activity.GOING_TO_WOUNDED:
            # Navigate to wounded
            if self.current_target_wounded:
                # Replan regularly (every 10 iterations)
                should_replan = False
                if not self.path or len(self.path) == 0:
                    should_replan = True
                elif self.iteration % 10 == 0:
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
            # Navigate to rescue center with regular replanning
            if self.rescue_zone_points:
                # Replan regularly (every 10 iterations, like exploration)
                should_replan = False
                if not self.path or len(self.path) == 0:
                    should_replan = True
                elif self.iteration % 10 == 0:
                    should_replan = True
                
                if should_replan:
                    self.path = self.creer_chemin(self.current_pose[:2], self.rescue_zone_points[0], explored_only=True)
                    self.last_replan_iteration = self.iteration
                
                if self.path:
                    command = self.go_to_point(lidar_data)
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
        
        # Store last command for display
        self.last_command = command.copy()

        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid,
                              self.estimated_pose,
                              title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")

        return command

    def filter_position(self, old_pos, new_pos, alpha=0.3):
        """
        Apply Exponential Moving Average (EMA) filtering to smooth positions.
        alpha: smoothing factor (0 < alpha < 1). 
               Lower alpha = more smoothing (slower response).
               Higher alpha = less smoothing (faster response).
        """
        x = (1.0 - alpha) * old_pos[0] + alpha * new_pos[0]
        y = (1.0 - alpha) * old_pos[1] + alpha * new_pos[1]
        return (x, y)

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
        dedup_radius = 50.0         # close detections considered same
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

                global_angle = normalize_angle(ptheta + angle)
                xw = px + dist * math.cos(global_angle)
                yw = py + dist * math.sin(global_angle)

                try:
                    name = etype.name if hasattr(etype, 'name') else str(etype)
                except Exception:
                    name = str(etype)

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
                    # Use filter to smooth position instead of direct replacement
                    self.wounded_to_rescue[i] = self.filter_position(self.wounded_to_rescue[i], (nx, ny), alpha=alpha_update)
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
                    self.rescue_zone_points[i] = self.filter_position(self.rescue_zone_points[i], (nx, ny), alpha=alpha_update)
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
                            self.wounded_to_rescue[i] = self.filter_position(self.wounded_to_rescue[i], pt, alpha=0.5)
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
                        self.rescue_zone_points[i] = self.filter_position(self.rescue_zone_points[i], pt, alpha=0.3)
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
            
            # Display commands below drone
            if hasattr(self, 'last_command'):
                cmd = self.last_command
                cmd_text = f"F:{cmd.get('forward', 0):.2f} L:{cmd.get('lateral', 0):.2f} R:{cmd.get('rotation', 0):.2f}"
                arcade.draw_text(cmd_text, 
                               current_pose_screen[0] - 50, 
                               current_pose_screen[1] - 40, 
                               (0, 0, 0), 
                               10, 
                               bold=False)
        except Exception:
            pass


    # --------------------------------------------------------------------------
    # FONCTIONS DE PILOTAGE
    # --------------------------------------------------------------------------
 
    def follow_path(self, lidar_data) -> CommandsDict:
        if not self.path:
            print("No path to follow.")
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        # Pure pursuit with lookahead and lateral control
        lookahead_idx = 0
        lookahead_dist = 40.0
        # choose a lookahead waypoint ahead along the path
        for i, wp in enumerate(self.path):
            if np.linalg.norm(wp - self.current_pose[:2]) > lookahead_dist:
                lookahead_idx = i
                break
        target_pos = self.path[min(lookahead_idx, len(self.path)-1)]

        delta_pos = target_pos - self.current_pose[:2]
        heading = self.current_pose[2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])

        # Decompose error into longitudinal and lateral components in body frame
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        # body-frame projection
        x_err = cos_h * delta_pos[0] + sin_h * delta_pos[1]   # forward error
        y_err = -sin_h * delta_pos[0] + cos_h * delta_pos[1]  # lateral (cross-track) error

        distance_to_target = math.hypot(delta_pos[0], delta_pos[1])

        # --- Rotation control (reduced when carrying wounded) ---
        angle_error = normalize_angle(target_angle - heading)
        deriv_error = angle_error - self.prev_angle_error
        # Gains adapt if carrying a wounded to reduce oscillation
        carrying = bool(self.grasper.grasped_wounded_persons)
        Kp_rot = self.Kp * (0.7 if carrying else 1.0)
        Kd_rot = self.Kd * (0.7 if carrying else 1.0)
        rotation_speed = Kp_rot * angle_error + Kd_rot * deriv_error
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error

        # --- Lateral control (new) ---
        # Cross-track PID to counter lateral deviations, stronger when carrying
        Kp_lat = 0.05 if not carrying else 0.09
        Kd_lat = 0.02 if not carrying else 0.04
        if not hasattr(self, 'prev_lat_error'):
            self.prev_lat_error = 0.0
        lat_deriv = y_err - self.prev_lat_error
        lateral_cmd = Kp_lat * y_err + Kd_lat * lat_deriv
        lateral_cmd = float(np.clip(lateral_cmd, -1.0, 1.0))
        self.prev_lat_error = y_err

        # --- Forward speed profile ---
        max_speed = 12.0 if not carrying else 9.0
        target_speed = max(0.0, min(max_speed, x_err * 0.15 + 0.3))

        measured_vel = self.measured_velocity()
        measured_speed = math.sqrt(measured_vel[0] ** 2 + measured_vel[1] ** 2)
        speed_error = target_speed - measured_speed
        deriv_speed = speed_error - self.prev_speed_error
        Kp_f = self.Kp_pos
        Kd_f = self.Kd_pos * (0.7 if carrying else 1.0)
        forward_cmd = Kp_f * speed_error + Kd_f * deriv_speed
        # reduce forward when sharp turns to avoid overshoot
        if abs(angle_error) > 0.4:
            forward_cmd *= 0.5
        forward_cmd = float(np.clip(forward_cmd, -1.0, 1.0))
        self.prev_speed_error = speed_error

        # Advance waypoints when close to any early segment
        close_thresh = 30.0 if not carrying else 25.0
        if len(self.path) > 0:
            # compute distances to first few waypoints and drop those already reached
            drop_until = -1
            max_check = min(5, len(self.path))
            for i in range(max_check):
                d = np.linalg.norm(self.path[i] - self.current_pose[:2])
                if d < close_thresh:
                    drop_until = i
                else:
                    break
            if drop_until >= 0:
                # prevent overshoot by easing commands when switching waypoints
                forward_cmd *= 0.5
                lateral_cmd *= 0.5
                # remove all reached waypoints up to drop_until
                self.path = self.path[drop_until+1:]

        # Optional damping on lateral during very small angles
        if abs(angle_error) < 0.1:
            lateral_cmd *= 0.7

        return {"forward": forward_cmd, "lateral": lateral_cmd, "rotation": rotation_speed}

    def go_to_point(self, lidar_data) -> CommandsDict:
        """Pure pursuit navigation with path densification for smoother following.
        Densifies waypoints to provide more intermediate targets for better tracking.
        """
        if not self.path:
            print("No path to follow in go_to_point.")
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        # --- Path densification: interpolate between waypoints ---
        # Add intermediate points between waypoints for smoother pursuit
        if not hasattr(self, '_densified_path') or not hasattr(self, '_last_path_length') or self._last_path_length != len(self.path):
            densified = []
            density_spacing = 15.0  # Add point every 15 units
            
            for i in range(len(self.path)):
                densified.append(self.path[i])
                if i < len(self.path) - 1:
                    # Interpolate between current and next waypoint
                    start = self.path[i]
                    end = self.path[i + 1]
                    segment_vec = end - start
                    segment_len = np.linalg.norm(segment_vec)
                    
                    if segment_len > density_spacing:
                        # Add intermediate points
                        num_points = int(segment_len / density_spacing)
                        for j in range(1, num_points):
                            alpha = j / num_points
                            interp_point = start + alpha * segment_vec
                            densified.append(interp_point)
            
            self._densified_path = densified
            self._last_path_length = len(self.path)
        
        # Use densified path for pure pursuit
        path_to_follow = self._densified_path if hasattr(self, '_densified_path') else self.path
        
        # --- Pure pursuit lookahead ---
        lookahead_dist = 35.0
        lookahead_idx = 0
        
        for i, wp in enumerate(path_to_follow):
            if np.linalg.norm(wp - self.current_pose[:2]) > lookahead_dist:
                lookahead_idx = i
                break
        
        target_pos = path_to_follow[min(lookahead_idx, len(path_to_follow)-1)]
        delta_pos = target_pos - self.current_pose[:2]
        heading = self.current_pose[2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])
        distance_to_target = np.linalg.norm(delta_pos)
        
        # --- Rotation control ---
        angle_error = normalize_angle(target_angle - heading)
        deriv_error = angle_error - self.prev_angle_error
        
        Kp_rot = self.Kp
        Kd_rot = self.Kd
        rotation_speed = Kp_rot * angle_error + Kd_rot * deriv_error
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error
        
        # --- Forward speed control (proportional to distance and angle alignment) ---
        # Speed profile: faster when aligned, slower on sharp turns
        max_speed = 12.0
        
        # Distance-based speed
        if distance_to_target < 30:
            speed_factor = distance_to_target / 30.0
        else:
            speed_factor = 1.0
        
        # Angle-based speed reduction
        if abs(angle_error) > 0.5:
            angle_factor = 0.4
        elif abs(angle_error) > 0.3:
            angle_factor = 0.7
        else:
            angle_factor = 1.0
        
        target_speed = max_speed * speed_factor * angle_factor
        target_speed = max(0.3, target_speed)  # Minimum speed
        
        # PID on speed
        measured_vel = self.measured_velocity()
        measured_speed = math.sqrt(measured_vel[0] ** 2 + measured_vel[1] ** 2)
        speed_error = target_speed - measured_speed
        deriv_speed = speed_error - self.prev_speed_error
        
        forward_cmd = self.Kp_pos * speed_error + self.Kd_pos * deriv_speed
        forward_cmd = float(np.clip(forward_cmd, -1.0, 1.0))
        self.prev_speed_error = speed_error
        
        # --- Waypoint advancement ---
        # Remove waypoints from original path when passed
        close_thresh = 25.0
        if len(self.path) > 0:
            drop_until = -1
            max_check = min(3, len(self.path))
            for i in range(max_check):
                d = np.linalg.norm(self.path[i] - self.current_pose[:2])
                if d < close_thresh:
                    drop_until = i
                else:
                    break
            if drop_until >= 0:
                self.path = self.path[drop_until+1:]
                # Force re-densification on next iteration
                if hasattr(self, '_last_path_length'):
                    delattr(self, '_last_path_length')
        
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
        
        # Draw orientation arrow
        arrow_length = 25
        heading = self.current_pose[2]
        end_x = current_pose[0] + arrow_length * math.cos(heading)
        end_y = current_pose[1] + arrow_length * math.sin(heading)
        arcade.draw_line(current_pose[0], current_pose[1], 
                        end_x, end_y, 
                        (255, 255, 0), 3)  # Yellow arrow
        # Draw arrowhead
        arrow_size = 8
        left_angle = heading + 2.6
        right_angle = heading - 2.6
        left_x = end_x + arrow_size * math.cos(left_angle)
        left_y = end_y + arrow_size * math.sin(left_angle)
        right_x = end_x + arrow_size * math.cos(right_angle)
        right_y = end_y + arrow_size * math.sin(right_angle)
        arcade.draw_line(end_x, end_y, left_x, left_y, (255, 255, 0), 3)
        arcade.draw_line(end_x, end_y, right_x, right_y, (255, 255, 0), 3)

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

    