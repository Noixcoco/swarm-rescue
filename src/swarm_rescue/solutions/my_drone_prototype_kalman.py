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
# A* n'est pas utilisé dans ce test, mais il est dans le chemin
try:
    from . import a_star
    pass
except ImportError:
    pass


class MyDroneProton(DroneAbstract):
    """
    HARNAIS DE TEST (V-Test) - CHEMIN CORRIGÉ
    Objectif : Tester UNIQUEMENT le pilotage (la fonction 'follow_path').
    Le chemin est codé en dur pour suivre la ligne verte.

    --- AJOUTS POUR LE TEST EKF ---
    - Logging des poses : self.estimated_poses et self.true_poses
    - Visualisation de l'erreur : draw_bottom_layer dessine l'erreur
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.current_pose = np.array([0.0, 0.0, 0.0])

        # Mission accomplie ?
        self.mission_done = False

        # --- Configuration de la Carte (Binaire) ---
        self.map_size_pixels = 1000
        self.map_resolution = 5
        self.grid_size = self.map_size_pixels // self.map_resolution
        self.map_max_index = self.grid_size - 1
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.map_origin_offset = np.array([self.grid_size // 2, self.grid_size // 2])

        # --- Sécurité et Inflation ---
        self.robot_radius_pixels = 30
        self.inflation_radius_cells = int(self.robot_radius_pixels / self.map_resolution)
        if self.inflation_radius_cells < 1:
            self.inflation_radius_cells = 1

        # --- OBJECTIF FIXE (La Croix Rouge en bas à gauche) ---
        self.goal_position = np.array([-300.0, -200.0])

        # parametre PID rotation
        self.prev_angle_error = 0.0
        self.Kp = 3
        self.Kd = 2

        # NOTE: Les paramètres PID de translation suivants ne sont pas utilisés
        # dans la fonction 'follow_path' actuelle.
        # self.prev_diff_position = np.zeros(2)  # dérivée pour translation
        # self.Kp_pos = 1.6
        # self.Kd_pos = 11.0

        # --- CHEMIN CORRIGÉ (Waypoints pour suivre la ligne verte) ---
        self.path = [
            np.array([300.0, -200.0]),
            np.array([-50, -200.0]),
            np.array([-50, 200]),
            np.array([-300, 200.0]),
            np.array([-300, -200])
        ]


        # --- EKF TEST LOGGING ---
        # Listes pour stocker les données à chaque étape
        self.estimated_poses = []
        self.true_poses = [] # AJOUTÉ: Pour stocker la trajectoire vraie
        
        # ------------------------

        ####### KALMAN #######

        # --- Configuration du Filtre de Kalman Étendu (EKF) ---
        # P = Matrice de Covariance (Notre incertitude)
        self.P = np.eye(3) * 500.0

        # Q = Bruit de Processus (Incertitude de l'odométrie)
        self.Q = np.diag([
            0.1,    # Incertitude sur x
            0.1,    # Incertitude sur y
            0.01    # Incertitude sur l'angle
        ])

        # R = Bruit de Mesure (Incertitude du GPS/Compas)
        self.R = np.diag([
            10.0,   # Incertitude du GPS en X
            10.0,   # Incertitude du GPS en Y
            0.5     # Incertitude du Compas
        ])

        # H = Matrice de Mesure
        self.H = np.eye(3)

        # I = Matrice Identité (pour les calculs)
        self.I = np.eye(3)

        # --- EKF TEST LOGGING ---
        # Listes pour stocker les données à chaque étape
        self.estimated_poses = []
        
        # ------------------------

        # self.state est inutile dans ce test

    def define_message_for_all(self) -> None:
        pass

    def control(self) -> CommandsDict:
        """
        Cerveau : Logique de test simplifiée.
        """


        # --- 1. PERCEPTION ---
        self.update_pose()
        self.update_map()
        self._display_map()

        lidar_data = self.lidar_values()
        if lidar_data is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}



        # --- EKF TEST: Log data ---
        # On log les données AVANT que 'update_pose' ne soit appelé
        # NOTE: 'self.true_pose' est fourni par la classe parent 'DroneAbstract'
        self.estimated_poses.append(np.copy(self.current_pose))


        # AJOUTÉ: Logger aussi la position vraie
        true_pos_current = self.true_position()
        if true_pos_current is not None:
            self.true_poses.append(np.copy(true_pos_current))
    
        # --------------------------
        # --- 2. STRATÉGIE (DÉSACTIVÉE) ---
        # Toute la logique FBE et A* est désactivée.

        # Si le chemin est terminé (liste vide)
        if not self.path:
            # Recharger le chemin si le test est fini (pour relancer le test)
            self.path = [
                np.array([400.0, -200.0]),
                np.array([400.0, -300.0]),
                np.array([100.0, -300.0]),
                np.array([100.0, 300.0]),
                np.array([-300.0, 300.0]),
                np.array([-400.0, -300.0])
            ]

        # Si la mission est déjà terminée, le drone reste immobile
        if self.mission_done:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # --- 3. ACTION (Le "Pilote") ---

        if self.path:
            command = self.follow_path(lidar_data)
        else:
            # Si le chemin est terminé, on s'arrête
            command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        command["grasper"] = 0
        return command

    # --------------------------------------------------------------------------
    # FONCTION DE DESSIN (Modifiée pour le test EKF)
    # --------------------------------------------------------------------------

    def draw_bottom_layer(self):
        """
        MODIFIÉ: Dessine le chemin (vert), les trajectoires (rouge/bleu), 
        les positions instantanées (cercles rouge/bleu), et la ligne d'erreur (jaune).
        """
        # Assumes self._half_size_array is defined in the parent class (e.g., DroneAbstract)

        # --- 1. Dessine le chemin A* (self.path) en vert ---
        if self.path:
            points_list_path = [] # Renommé pour éviter confusion
            start_point_path = self.current_pose[:2] + self._half_size_array
            points_list_path.append((start_point_path[0], start_point_path[1]))
            
            for point_world in self.path:
                point_arcade = point_world + self._half_size_array
                points_list_path.append((point_arcade[0], point_arcade[1]))
            
            if len(points_list_path) > 1:
                arcade.draw_line_strip(points_list_path, arcade.color.GREEN, 5)

        # --- 2. Dessine les trajectoires historiques ---

        # Trajectoire Estimée (Rouge)
        if len(self.estimated_poses) > 1:
            estimated_path_points = []
            for pose in self.estimated_poses:
                screen_pos = pose[:2] + self._half_size_array
                estimated_path_points.append((screen_pos[0], screen_pos[1]))
            arcade.draw_line_strip(estimated_path_points, arcade.color.RED, 2)

        # Trajectoire Vraie (Bleue)
        if len(self.true_poses) > 1:
            true_path_points = []
            for pose in self.true_poses:
                screen_pos = pose[:2] + self._half_size_array
                true_path_points.append((screen_pos[0], screen_pos[1]))
            arcade.draw_line_strip(true_path_points, arcade.color.BLUE, 2)


        # --- 3. Dessine la Vérité Terrain vs Estimation EKF INSTANTANÉE ---
        true_pos = self.true_position()
        
        if true_pos is not None:
            true_pos_arcade = true_pos[:2] + self._half_size_array
            est_pos_arcade = self.current_pose[:2] + self._half_size_array


            # E. Cercle Estimé (Rouge) - Dessiné par-dessus la trajectoire
            arcade.draw_circle_filled(
                est_pos_arcade[0], est_pos_arcade[1],
                10, arcade.color.RED
            )
            
            
    #------------------------------------------------------------------------
    # FONCTIONS DE PILOTAGE
    # --------------------------------------------------------------------------
    def follow_path(self, lidar_data) -> CommandsDict:
        if not self.path:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        target_pos = self.path[0]
        delta_pos = target_pos - self.current_pose[:2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])

        # --- PID sur la rotation ---
        angle_error = normalize_angle(target_angle - self.current_pose[2])
        deriv_error = angle_error - self.prev_angle_error
        rotation_speed = self.Kp * angle_error + self.Kd * deriv_error
        rotation_speed = np.clip(rotation_speed, -1.0, 1.0)
        self.prev_angle_error = angle_error

        distance_to_target = np.linalg.norm(delta_pos)

        # --- Gestion de la vitesse ---
        if angle_error > 0.1 or angle_error < -0.1 : # Si on est en train de tourner, alors on met forward_speed à 0         
            forward_speed = 0
            #print("on tourne")

        else :
            target_speed = min(4.0, distance_to_target*0.05 - 0.03)
            measured_speed = math.sqrt(self.measured_velocity()[0]**2 + self.measured_velocity()[1]**2)

            if measured_speed > target_speed :
                forward_speed = -1 # On ralentit
            elif measured_speed < target_speed :
                forward_speed = 1 # On accélère
            else :
                forward_speed = 0 # On garde la même vitesse
            
            #print("target speed : ", target_speed)
            #print("measured speed : ", measured_speed)

        # Si proche du waypoint → passer au suivant
        if distance_to_target < 50:
            self.path.pop(0)

        # Si proche du but final
        dist_to_goal = np.linalg.norm(self.goal_position - self.current_pose[:2])
        if dist_to_goal < 50:
            print("✅ Mission terminée : le drone est arrivé au but.")
            self.mission_done = True
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        return {"forward": forward_speed, "lateral": 0.0, "rotation": rotation_speed}

   
    # --------------------------------------------------------------------------
    # FONCTIONS DE CONVERSION & AFFICHAGE
    # --------------------------------------------------------------------------

    def world_to_grid(self, world_pos_xy):
        grid_pos_xy = (world_pos_xy / self.map_resolution + self.map_origin_offset).astype(int)
        grid_pos_xy[0] = np.clip(grid_pos_xy[0], 0, self.map_max_index)
        grid_pos_xy[1] = np.clip(grid_pos_xy[1], 0, self.map_max_index)
        return grid_pos_xy

    def grid_to_world(self, grid_pos_xy):
        return (grid_pos_xy - self.map_origin_offset) * self.map_resolution

    def _display_map(self):
        vis_map = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        obstacle_mask = (self.occupancy_grid <= 0)
        inflated_map = ndimage.binary_dilation(obstacle_mask, iterations=self.inflation_radius_cells)

        vis_map[self.occupancy_grid == 0] = [128, 128, 128] # Gris (Inconnu)
        vis_map[inflated_map == True] = [50, 50, 50]        # Gris foncé (Inflé)
        vis_map[self.occupancy_grid == -1] = [0, 0, 0]      # Noir (Obstacle)
        vis_map[self.occupancy_grid == 1] = [255, 255, 255] # Blanc (Libre)

        if self.path:
            for point_world in self.path:
                point_grid = self.world_to_grid(point_world)
                # NETTOYÉ: radius=1 pour dessiner un pixel (radius=0 est ambigu)
                cv2.circle(vis_map, (point_grid[0], point_grid[1]),
                           radius=1, color=(0, 255, 0), thickness=-1)

        if self.goal_position is not None:
            goal_grid_pos_xy = self.world_to_grid(self.goal_position)
            cv2.circle(vis_map, (goal_grid_pos_xy[0], goal_grid_pos_xy[1]),
                           radius=3, color=(255, 0, 0), thickness=-1) # Bleu

        drone_grid_pos_xy = self.world_to_grid(self.current_pose[:2])
        cv2.circle(vis_map, (drone_grid_pos_xy[0], drone_grid_pos_xy[1]),
                       radius=2, color=(0, 0, 255), thickness=-1) # Rouge

        vis_map_large = cv2.resize(vis_map, (400, 400), interpolation=cv2.INTER_NEAREST)
        vis_map_flipped = cv2.flip(vis_map_large, 0)

        cv2.imshow("Carte Mentale du Drone (B&W)", vis_map_flipped)
        cv2.waitKey(1)

    # --------------------------------------------------------------------------
    # FONCTIONS PRINCIPALES (Localisation, Cartographie Binaire)
    # --------------------------------------------------------------------------

    def update_pose(self):

        # --- 1. ÉTAPE DE PRÉDICTION (Odométrie) ---
        odom_data = self.odometer_values()
        if odom_data is None:
            dist_traveled = 0.0
            rotation_change = 0.0
        else:
            dist_traveled = odom_data[0]
            rotation_change = odom_data[2]

        prev_x, prev_y, prev_angle = self.current_pose

        # (a) Prédire le nouvel état (x_pred)
        new_angle = normalize_angle(prev_angle + rotation_change)
        new_x = prev_x + dist_traveled * math.cos(new_angle)
        new_y = prev_y + dist_traveled * math.sin(new_angle)
        self.current_pose = np.array([new_x, new_y, new_angle])

        # (b) Calculer le Jacobien (F)
        F = np.eye(3)
        if dist_traveled != 0:
            F[0, 2] = -dist_traveled * math.sin(new_angle)
            F[1, 2] = dist_traveled * math.cos(new_angle)

        # (c) Prédire la nouvelle covariance (P_pred)
        self.P = F @ self.P @ F.T + self.Q

        # --- 2. ÉTAPE DE CORRECTION (GPS/Compas) ---
        gps_pos = self.measured_gps_position()
        if not np.isnan(gps_pos[0]):
            compass_angle = self.measured_compass_angle()
            if not np.isnan(compass_angle):
                # Vecteur de mesure (z)
                z = np.array([gps_pos[0], gps_pos[1], compass_angle])

                # (a) Innovation (y)
                y = z - self.H @ self.current_pose
                y[2] = normalize_angle(y[2]) # Normaliser l'erreur d'angle

                # (b) Covariance de l'Innovation (S)
                S = self.H @ self.P @ self.H.T + self.R

                # (c) Gain de Kalman (K)
                try:
                    K = self.P @ self.H.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    return

                # (d) Corriger l'état (x_new)
                self.current_pose = self.current_pose + K @ y

                # (e) Corriger la covariance (P_new)
                self.P = (self.I - K @ self.H) @ self.P

                # Re-normaliser l'angle après la correction
                self.current_pose[2] = normalize_angle(self.current_pose[2])


    def update_map(self):
        grid_pos_xy = self.world_to_grid(self.current_pose[:2])
        gx, gy = grid_pos_xy[0], grid_pos_xy[1]

        # NETTOYÉ: La variable 'gyx' n'était pas utilisée
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