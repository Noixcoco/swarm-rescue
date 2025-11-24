import math
import numpy as np
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle 
from swarm_rescue.simulation.drone.controller import CommandsDict
from typing import Tuple, Dict, Any

# Requis pour la cartographie et le dessin
from skimage.draw import line as draw_line
from scipy import ndimage 
import cv2
import arcade


class MyDronePrototype(DroneAbstract):
    """
    IMPLEMENTATION PURE PURSUIT (3 COMMANDES)
    Suivi de chemin fixe et densifié en utilisant la distance de regard (Ld) 
    et la correction latérale (strafe), avec des contrôles de direction robustes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.mission_done = False

        # --- Configuration de la Carte ---
        self.map_size_pixels = 1000 
        self.map_resolution = 5 
        self.grid_size = self.map_size_pixels // self.map_resolution
        self.map_max_index = self.grid_size - 1 
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8) 
        self.map_origin_offset = np.array([self.grid_size // 2, self.grid_size // 2]) 
        self.robot_radius_pixels = 30 
        self.inflation_radius_cells = int(self.robot_radius_pixels / self.map_resolution)
        if self.inflation_radius_cells < 1:
            self.inflation_radius_cells = 1
        
        # --- OBJECTIF FIXE ---
        self.goal_position = np.array([-300.0, -200.0])

        # --- CHEMIN STRATÉGIQUE (Waypoints de virage) ---
        strategic_waypoints = [
            np.array([300.0, -200.0]),
            np.array([-50, -200.0]),
            np.array([-50, 200]),
            np.array([-300, 200.0]),
            np.array([-300, -200]),
        ]
        
        # Densification
        self.densification_step = 20.0
        self.path = self.densify_path(strategic_waypoints, step_distance=self.densification_step)
        self.global_path_for_drawing = [p.copy() for p in self.path] 
        
        # --- Paramètres de CONTRÔLE PURE PURSUIT ---
        self.lookahead_distance = 70.0  # Ld: Distance de regard
        self.current_waypoint_index = 0  # Index du dernier point du chemin passé
        self.waypoint_threshold = 50.0   # Seuil de progression d'index
        self.arrival_threshold = 50.0    # Seuil de fin de mission

        # PID Rotation (Yaw)
        self.prev_angle_error = 0.0
        self.Kp_rot = 3.0
        self.Kd_rot = 2.0
        
        # P Lateral (Strafe)
        self.Kp_lat = 0.05               # Gain pour la correction latérale
        
        # P Forward
        self.Kp_forward = 0.008 
        self.max_forward_speed = 0.5

    # --------------------------------------------------------------------------
    # Contrôleur Principal
    # --------------------------------------------------------------------------

    def define_message_for_all(self) -> None:
        pass 

    def control(self) -> CommandsDict:
        """
        Logique de contrôle principale.
        """
        
        # 1. PERCEPTION
        self.update_pose()
        self.update_map()
        self._display_map() 
        
        lidar_data = self.lidar_values()
        if lidar_data is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # 2. STRATÉGIE
        if self.mission_done:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        
        # 3. ACTION (Pure Pursuit)
        if self.path and self.current_waypoint_index < len(self.path):
            command = self.follow_path_pure_pursuit(lidar_data) 
        else:
            command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        command["grasper"] = 0
        return command

    # --------------------------------------------------------------------------
    # Fonctions de Pilotage Pure Pursuit (CORRIGÉES)
    # --------------------------------------------------------------------------

    def find_lookahead_point(self) -> Tuple[np.ndarray, bool]:
        """
        Recherche le point P_look situé à Ld du drone sur le chemin restant,
        en s'assurant qu'il est DEVANT le drone.
        """
        current_pos = self.current_pose[:2]
        current_heading = self.current_pose[2]

        # On parcourt le chemin à partir de l'index actuel
        for i in range(self.current_waypoint_index, len(self.path)):
            point = self.path[i]
            dist = np.linalg.norm(point - current_pos)
            
            # VÉRIFICATION DE LA DIRECTION : Le point doit être dans le cône avant (± 90 degrés)
            vector_to_point = point - current_pos
            angle_to_point = math.atan2(vector_to_point[1], vector_to_point[0])
            angle_diff = normalize_angle(angle_to_point - current_heading)
            
            if abs(angle_diff) > math.pi / 2:
                continue # Ignorer ce point, il est derrière

            # Recherche du P_look (point à distance >= Ld)
            if dist >= self.lookahead_distance:
                return point, False  # P_look trouvé

        # Si le chemin est terminé ou si tous les points restants sont trop proches, 
        # viser le dernier point.
        if len(self.path) > 0:
            return self.path[-1], True 
            
        return None, False

    def find_nearest_index(self) -> int:
        """
        Trouve l'index du point du chemin densifié le plus proche du drone.
        """
        current_pos = self.current_pose[:2]
        min_dist = float('inf')
        nearest_index = self.current_waypoint_index # Commence la recherche à partir de l'index actuel

        # On parcourt uniquement les points à partir de l'index actuel pour ne pas revenir en arrière
        for i in range(self.current_waypoint_index, len(self.path)):
            point = self.path[i]
            dist = np.linalg.norm(point - current_pos)
            
            if dist < min_dist:
                min_dist = dist
                nearest_index = i
        
        # Le nouvel index de départ pour le Pure Pursuit est juste après le point le plus proche.
        return nearest_index

    def follow_path_pure_pursuit(self, lidar_data) -> CommandsDict:
        
        # --- 1. Progression du Chemin (Avancer l'Index) ---
        
        # Mettre à jour l'index de progression pour qu'il soit le point le plus proche.
        # Cela garantit que la recherche de P_look commencera devant la position actuelle.
        nearest_index = self.find_nearest_index()
        self.current_waypoint_index = nearest_index
        
        # --- 2. Trouver le P_look ---
        target_pos, is_final_target = self.find_lookahead_point()
        
        if target_pos is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        delta_pos = target_pos - self.current_pose[:2]
        distance_to_target = np.linalg.norm(delta_pos)
        
        # --- 3. Contrôle de Rotation (PD) vers P_look ---
        
        target_angle = math.atan2(delta_pos[1], delta_pos[0])
        angle_error = normalize_angle(target_angle - self.current_pose[2])
        
        deriv_error = angle_error - self.prev_angle_error
        rotation_speed = self.Kp_rot * angle_error + self.Kd_rot * deriv_error
        rotation_speed = np.clip(rotation_speed, -1.0, 1.0)
        self.prev_angle_error = angle_error
        
        # --- 4. Contrôle de Translation Latérale (P-Control sur l'Erreur Latérale) ---
        
        heading = self.current_pose[2]
        # Matrice de rotation inverse (Monde -> Drone)
        R_inv = np.array([
            [math.cos(heading), math.sin(heading)],
            [-math.sin(heading), math.cos(heading)]
        ])
        
        # Erreur latérale dans le repère du drone (composante Y du vecteur au P_look)
        delta_pos_drone_frame = R_inv @ delta_pos
        lateral_error = delta_pos_drone_frame[1] 
        
        # Kp_lat doit être ajusté pour une correction efficace (e.g., 0.15)
        lateral_speed = self.Kp_lat * lateral_error
        lateral_speed = np.clip(lateral_speed, -1.0, 1.0)
        
        # --- 5. Contrôle d'Avancement (P-Control sur Vitesse) ---
        
        if abs(angle_error) > 0.15: # Ralentir si l'angle de correction est trop grand
            forward_speed = 0.0
        else:
            forward_speed_raw = self.Kp_forward * distance_to_target
            forward_speed = np.clip(forward_speed_raw, 0.0, self.max_forward_speed)
            
            if distance_to_target < 20: 
                 forward_speed = 0.0

        # --- 6. Gestion de la fin de mission ---
        
        if is_final_target and distance_to_target < self.arrival_threshold:
            print("✅ Mission terminée : le drone est arrivé au but.")
            self.mission_done = True
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
            
        return {"forward": forward_speed, "lateral": lateral_speed, "rotation": rotation_speed}

    # --------------------------------------------------------------------------
    # Fonctions de Cartographie et Utilitaires
    # --------------------------------------------------------------------------

    def world_to_grid(self, world_pos_xy): 
        """ Convertit les coordonnées du monde (m) en coordonnées de grille (pixels). """
        grid_pos_xy = (world_pos_xy / self.map_resolution + self.map_origin_offset).astype(int)
        grid_pos_xy[0] = np.clip(grid_pos_xy[0], 0, self.map_max_index)
        grid_pos_xy[1] = np.clip(grid_pos_xy[1], 0, self.map_max_index)
        return grid_pos_xy

    def grid_to_world(self, grid_pos_xy):
        """ Convertit les coordonnées de grille (pixels) en coordonnées du monde (m). """
        return (grid_pos_xy - self.map_origin_offset) * self.map_resolution

    def densify_path(self, original_path, step_distance=20.0):
        """ Crée une trajectoire dense en insérant des points à intervalle régulier. """
        if not original_path or len(original_path) < 2:
            return original_path
            
        dense_path = []
        for i in range(len(original_path) - 1):
            p_start = original_path[i]
            p_end = original_path[i+1]
            vector = p_end - p_start
            distance = np.linalg.norm(vector)
            
            if distance < step_distance:
                dense_path.append(p_start.copy())
                continue
                
            unit_vector = vector / distance
            num_steps = int(distance // step_distance)
            
            for j in range(num_steps):
                intermediate_point = p_start + unit_vector * (j * step_distance)
                dense_path.append(intermediate_point.copy())
                
        dense_path.append(original_path[-1].copy())
        return dense_path


    # --------------------------------------------------------------------------
    # Fonctions de Localisation et Perception
    # --------------------------------------------------------------------------

    def update_pose(self):
        """ Met à jour la position et l'orientation (self.current_pose) du drone. """
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()

        if not np.isnan(gps_pos[0]):
            self.current_pose[0] = gps_pos[0]
            self.current_pose[1] = gps_pos[1]
            self.current_pose[2] = compass_angle
        else:
            odom_data = self.odometer_values() 
            if odom_data is None: return
            dist_traveled = odom_data[0]
            rotation_change = odom_data[2]
            
            self.current_pose[2] += rotation_change
            self.current_pose[2] = normalize_angle(self.current_pose[2])
            
            dx = dist_traveled * math.cos(self.current_pose[2])
            dy = dist_traveled * math.sin(self.current_pose[2])
            self.current_pose[0] += dx
            self.current_pose[1] += dy

    def update_map(self):
        """ Met à jour la grille d'occupation (occupancy_grid) à partir du LIDAR et de la pose. """
        grid_pos_xy = self.world_to_grid(self.current_pose[:2])
        gx, gy = grid_pos_xy[0], grid_pos_xy[1]
        
        if 0 <= gy < self.grid_size and 0 <= gx < self.grid_size:
            self.occupancy_grid[gy, gx] = 1 
        
        lidar_values = self.lidar_values()
        if lidar_values is None: return 
            
        lidar_angles = self.lidar().ray_angles
        max_range = self.lidar().max_range * 0.95 

        for angle, dist in zip(lidar_angles, lidar_values):
            global_angle = normalize_angle(self.current_pose[2] + angle)
            
            if dist < max_range:
                end_point_world = self.current_pose[:2] + np.array([dist * math.cos(global_angle), dist * math.sin(global_angle)])
                is_obstacle = True
            else:
                end_point_world = self.current_pose[:2] + np.array([max_range * math.cos(global_angle), max_range * math.sin(global_angle)])
                is_obstacle = False
            
            grid_end_point_xy = self.world_to_grid(end_point_world) 
            gx_end, gy_end = grid_end_point_xy[0], grid_end_point_xy[1]
            
            rr, cc = draw_line(gy, gx, gy_end, gx_end)
            
            rr = np.clip(rr, 0, self.map_max_index)
            cc = np.clip(cc, 0, self.map_max_index)

            self.occupancy_grid[rr, cc] = 1 
            
            if is_obstacle:
                self.occupancy_grid[gy_end, gx_end] = -1


    # --------------------------------------------------------------------------
    # Fonctions de Dessin (Affichage visuel)
    # --------------------------------------------------------------------------

    def draw_bottom_layer(self):
        """ Dessine le chemin dense global (fixe) en vert et la ligne de visée jaune (P_look) pour Arcade. """
        path_to_draw = self.global_path_for_drawing
        
        if path_to_draw:
            points_list = []
            # self._half_size_array est défini dans la classe parente (DroneAbstract)
            for point_world in path_to_draw:
                point_arcade = point_world + self._half_size_array
                points_list.append((point_arcade[0], point_arcade[1]))

            if len(points_list) > 1:
                arcade.draw_line_strip(points_list, arcade.color.GREEN, 5)
                
            # Dessiner la ligne vers P_look
            target_pos, _ = self.find_lookahead_point()
            if target_pos is not None:
                target_arcade = target_pos + self._half_size_array
                drone_arcade = self.current_pose[:2] + self._half_size_array
                
                arcade.draw_line(drone_arcade[0], drone_arcade[1], 
                                 target_arcade[0], target_arcade[1], 
                                 arcade.color.YELLOW, 3)

    def _display_map(self):
        """ Affiche la carte de l'environnement du drone via OpenCV. """
        vis_map = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        obstacle_mask = (self.occupancy_grid <= 0)
        inflated_map = ndimage.binary_dilation(obstacle_mask, iterations=self.inflation_radius_cells)

        vis_map[self.occupancy_grid == 0] = [128, 128, 128] # Gris (Inconnu)
        vis_map[inflated_map == True] = [50, 50, 50] # Gris foncé (Inflated obstacles)
        vis_map[self.occupancy_grid == -1] = [0, 0, 0] # Noir (Obstacles confirmés)
        vis_map[self.occupancy_grid == 1] = [255, 255, 255] # Blanc (Exploré/Libre)
        
        if self.path:
            # Dessiner le waypoint actuel en jaune
            if self.current_waypoint_index < len(self.path):
                point_world = self.path[self.current_waypoint_index]
                point_grid = self.world_to_grid(point_world)
                cv2.circle(vis_map, (point_grid[0], point_grid[1]), 
                           radius=2, color=(0, 255, 255), thickness=-1) 

        if self.goal_position is not None:
            goal_grid_pos_xy = self.world_to_grid(self.goal_position)
            cv2.circle(vis_map, (goal_grid_pos_xy[0], goal_grid_pos_xy[1]),
                       radius=3, color=(255, 0, 0), thickness=-1) 
                            
        drone_grid_pos_xy = self.world_to_grid(self.current_pose[:2])
        cv2.circle(vis_map, (drone_grid_pos_xy[0], drone_grid_pos_xy[1]), 
                   radius=2, color=(0, 0, 255), thickness=-1) 
                            
        vis_map_large = cv2.resize(vis_map, (400, 400), interpolation=cv2.INTER_NEAREST)
        vis_map_flipped = cv2.flip(vis_map_large, 0) 

        cv2.imshow("Carte Mentale du Drone (Pure Pursuit)", vis_map_flipped)
        cv2.waitKey(1)