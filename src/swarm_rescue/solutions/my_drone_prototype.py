import math
import numpy as np
# ... (rest of the original imports)
from scipy.ndimage import binary_dilation
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.drone.controller import CommandsDict
from typing import Tuple, Dict, Any

# Requis pour la cartographie et le dessin
from skimage.draw import line as draw_line
from scipy.ndimage import generate_binary_structure, binary_dilation
from scipy import ndimage 
import cv2
import arcade
# A* n'est pas utilisé dans ce test, mais il est dans le chemin
try:
    from . import a_star
except ImportError:
    pass

import sys
from pathlib import Path

# Ajoute le chemin du dossier parent au sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from examples.example_mapping import OccupancyGrid
from swarm_rescue.simulation.utils.pose import Pose


class MyDronePrototype(DroneAbstract):
    # ... (Keep creer_chemin as is)
    def creer_chemin(self, start_world, goal_world):
        """
        Calcule un chemin entre start_world et goal_world en évitant les murs et les zones proches des murs (moins de 4 pixels) avec A*.
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
        # Dilate les murs pour éviter les zones proches (moins de 2 pixels)
        struct = np.ones((5, 5), dtype=bool)  # carré 5x5 ~ rayon 2
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)
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
            y0 = max(0, gy - radius_clear)
            y1 = min(grid.shape[0], gy + radius_clear + 1)
            x0 = max(0, gx - radius_clear)
            x1 = min(grid.shape[1], gx + radius_clear + 1)
            danger_zone[y0:y1, x0:x1] = False
        except Exception:
            # en cas de problème d'indices, on ignore et laisse danger_zone inchangé
            pass

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]  # 4-connecté

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
                if (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    if danger_zone[neighbor]:
                        continue
                    tentative_g_score = gscore[current] + 1
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                        continue
                    if tentative_g_score < gscore.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
        # Pas de chemin trouvé
        return []

    """
    HARNAIS DE TEST (V-Test) - CHEMIN CORRIGÉ
    Objectif : Tester UNIQUEMENT le pilotage (la fonction 'follow_path').
    Le chemin est codé en dur pour suivre la ligne verte.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Le drone doit utiliser self.current_pose pour le pilotage (follow_path)
        # Mais pour le KF, on utilise un état à 3 dimensions (x, y, theta)
        self.current_pose = np.array([0.0, 0.0, 0.0]) # [x, y, theta]

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
        
        # ----------------------------------------------------------------------
        # --- INITIALISATION DU FILTRE DE KALMAN (EKF dans ce cas simple) ---
        # ----------------------------------------------------------------------
        # État initial: [x, y, theta]^T
        self.state_estimate = np.array([[0.0], [0.0], [0.0]]) 
        
        # Covariance de l'état (P). Grandes valeurs initiales pour l'incertitude.
        self.P = np.diag([10.0, 10.0, 10.0])
        
        # Covariance du bruit du processus (Q). Petite incertitude pour le modèle de mouvement (odométrie).
        # Q = np.diag([var_x, var_y, var_theta])
        self.Q = np.diag([0.01**2, 0.01**2, np.deg2rad(0.5)**2]) 
        
        # Covariance du bruit de mesure (R). Incertitude des capteurs (GPS/Compass).
        # GPS est généralement plus précis en position que l'odométrie à long terme.
        # Compass est une mesure directe de theta.
        # R = np.diag([var_gps_x, var_gps_y, var_compass_theta])
        self.R = np.diag([0.5**2, 0.5**2, np.deg2rad(1.0)**2])
        
        # Variable pour stocker la pose odométrique précédente
        # Utilisé pour calculer le déplacement du drone pour le modèle de mouvement
        self.prev_odom_pose = np.array([0.0, 0.0, 0.0])


    def define_message_for_all(self) -> None:
        pass 

    def control(self) -> CommandsDict:
        """
        Cerveau : Logique de test simplifiée.
        """

        # increment the iteration counter
        self.iteration += 1
        
        # --- 1. PERCEPTION ---
        self.update_pose() # *** Mise à jour via le KF ***
        
        # Mise à jour de la grille probabiliste self.grid.grid (utilisée pour l'exploration)
        # On utilise l'état estimé par le KF
        self.estimated_pose = Pose(np.asarray([self.state_estimate[0,0], self.state_estimate[1,0]]),
                                   self.state_estimate[2,0])
        self.grid.update_grid(pose=self.estimated_pose) # Mise à jour de la carte utilisée!

        lidar_data = self.lidar_values()
        if lidar_data is None:
            print("lidar_data is None")
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # --- 2. STRATÉGIE (DÉSACTIVÉE) ---
        # Replanification si le chemin est vide, si le waypoint est atteint, ou tous les 10 itérations
        need_replan = False
        if not self.path or len(self.path) < 1:
            need_replan = True
        # Met à jour le chemin tous les 10 itérations (si cible existante)
        # Note: self.current_pose est maintenant mise à jour à partir du KF
        if self.path and hasattr(self, 'target_point') and self.iteration % 10 == 0:
            need_replan = True

        if need_replan:
            self.frontiers_world = self.find_safe_frontier_points() # Met à jour la liste des frontières
            #print("frontieres : ",self.frontiers_world)
            if self.frontiers_world:
                # Scoring simplifié : choisir le point de frontière le plus proche
                # Utiliser la pose estimée pour le calcul de distance
                distances = [np.linalg.norm(f - self.estimated_pose.position) for f in self.frontiers_world]
                closest_index = np.argmin(distances)
                target_point = self.frontiers_world[closest_index]
                self.target_point = target_point
                # Le chemin devient ce point de frontière
                self.path = self.creer_chemin(self.estimated_pose.position, target_point)
        
        # --- 3. ACTION (Le "Pilote") ---
        

        if self.path:
            command = self.follow_path(lidar_data)
        else:
            # Si le chemin est terminé, on s'arrête
            print("Exploration terminée ou pas de frontieres sûres trouvées.")
            command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        command["grasper"] = 0

        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid,
                                 self.estimated_pose,
                                 title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                                 self.estimated_pose,
                                 title="zoomed occupancy grid")

        return command

    # ... (Keep find_safe_frontier_points, draw_bottom_layer, follow_path, display_pose as is,
    # but ensure follow_path uses self.current_pose which is updated by KF)
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
        
        """for l in grid_map.T :
            line_str = ""
            for e in l :
                if e <= SEUIL_FREE :
                    line_str += "x"
                elif e >= SEUIL_MUR :
                    line_str += "I"
                else :
                    line_str += "o"
            print("map : ", line_str)"""
        #print("w :", is_wall)                      
        
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

        for l in frontier_mask.T :
            line_str = ""
            for e in l :
                if e :
                    line_str += "S"
                else :
                    line_str += "o"
            #print("frontières : ", line_str)

        # Structure 8-connectée pour la dilatation
        struct = np.ones((5, 5), dtype=bool)
        # Dilate is_wall by 2 pixels
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=2)
        # Remove frontiers too close to a wall
        frontier_mask = frontier_mask & (~danger_zone)
        safe_frontiers_mask = frontier_mask

        for l in safe_frontiers_mask.T :
            line_str = ""
            for e in l :
                if e :
                    line_str += "S"
                else :
                    line_str += "o"
            #print("sans murs : ", line_str)

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

    # --------------------------------------------------------------------------
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


    # --------------------------------------------------------------------------
    # FONCTIONS DE PILOTAGE
    # --------------------------------------------------------------------------
    
    def follow_path(self, lidar_data) -> CommandsDict:
        if not self.path:
            print("No path to follow.")
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        
        # Utilisez l'état estimé par le KF comme pose actuelle pour le contrôle
        current_x = self.state_estimate[0, 0]
        current_y = self.state_estimate[1, 0]
        current_theta = self.state_estimate[2, 0]
        
        target_pos = self.path[0]
        delta_pos = target_pos - np.array([current_x, current_y])
        target_angle = math.atan2(delta_pos[1], delta_pos[0])

        # --- PID sur la rotation (inchangé) ---
        angle_error = normalize_angle(target_angle - current_theta)
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

        # Si proche du waypoint → passer au suivant
        # seuil réduit pour permettre d'atteindre les waypoints plus précisément
        if distance_to_target < 30:
        # on arrête le mouvement frontal pour éviter dépassement
            forward_cmd = 0.0
            self.path.pop(0)

        # Si proche du but final
        dist_to_goal = np.linalg.norm(self.goal_position - np.array([current_x, current_y]))

        # Mettre à jour self.current_pose pour la visualisation, même si on utilise self.state_estimate
        self.current_pose[0] = current_x
        self.current_pose[1] = current_y
        self.current_pose[2] = current_theta
        
        return {"forward": forward_cmd, "lateral": 0.0, "rotation": rotation_speed}

    def display_pose(self) :
        radius = 10
        red = (0,0,255)
        green = (0,255,0)

        # Utiliser la pose estimée (self.state_estimate) pour la visualisation
        current_pose = self.state_estimate[:2, 0] + self._half_size_array
        arcade.draw_circle_filled(current_pose[0],
                                     current_pose[1],
                                     radius=radius,
                                     color=red)

    # --------------------------------------------------------------------------
    # FONCTIONS PRINCIPALES (Localisation, Cartographie Binaire)
    # --------------------------------------------------------------------------

    def update_pose(self):
        """
        Mise à jour de la pose en utilisant le Filtre de Kalman.
        """
        # 1. PRÉDICTION (Basée sur l'odométrie)
        odom_data = self.odometer_values()
        if odom_data is not None:
            dist_traveled = odom_data[0]
            rotation_change = odom_data[2]
            
            # Modèle de mouvement non-linéaire (utilisé pour EKF)
            # x_k = f(x_{k-1}, u_k)
            # u_k = [dist_traveled, rotation_change]
            
            # État prédit (x_hat_k)
            theta_prev = self.state_estimate[2, 0]
            
            # L'état prédit est une fonction non-linéaire de l'état précédent et de l'odométrie
            pred_x = self.state_estimate[0, 0] + dist_traveled * math.cos(theta_prev)
            pred_y = self.state_estimate[1, 0] + dist_traveled * math.sin(theta_prev)
            pred_theta = normalize_angle(theta_prev + rotation_change)
            
            # Mise à jour de l'état prédit
            x_hat_k = np.array([[pred_x], [pred_y], [pred_theta]])
            
            # Calcul de la matrice Jacobienne (F) du modèle de mouvement
            F = np.array([[1.0, 0.0, -dist_traveled * math.sin(theta_prev)],
                          [0.0, 1.0, dist_traveled * math.cos(theta_prev)],
                          [0.0, 0.0, 1.0]])

            # Covariance prédite (P_k = F P_{k-1} F^T + Q)
            P_k = F @ self.P @ F.T + self.Q
            
            self.state_estimate = x_hat_k
            self.P = P_k
        
        # 2. MISE À JOUR (Basée sur les mesures GPS/Compass)
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        
        # Ne mettre à jour que si les données de mesure sont disponibles
        if gps_pos is not None and not np.isnan(gps_pos[0]):
            
            # Vecteur de mesure (z)
            # z = [gps_x, gps_y, compass_theta]^T
            z = np.array([[gps_pos[0]], [gps_pos[1]], [compass_angle]])
            
            # Fonction d'observation (h). Ici, c'est l'identité car la mesure est l'état.
            # z_k = h(x_k)
            h = self.state_estimate # H est la matrice identité

            # Matrice Jacobienne d'observation (H). Ici, c'est l'identité (3x3).
            H = np.eye(3)
            
            # Calcul de l'innovation (y)
            y = z - h
            # Normaliser l'erreur angulaire dans l'innovation
            y[2, 0] = normalize_angle(y[2, 0]) 
            
            # Calcul de la covariance d'innovation (S)
            S = H @ self.P @ H.T + self.R
            
            # Calcul du gain de Kalman (K)
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Mise à jour de l'état (x)
            # x_new = x_pred + K y
            self.state_estimate = self.state_estimate + K @ y
            # Normaliser l'angle après la mise à jour
            self.state_estimate[2, 0] = normalize_angle(self.state_estimate[2, 0])
            
            # Mise à jour de la covariance (P)
            # P_new = (I - K H) P_pred
            I = np.eye(3)
            self.P = (I - K @ H) @ self.P

        # Mettre à jour self.current_pose pour le reste du code
        self.current_pose[0] = self.state_estimate[0, 0]
        self.current_pose[1] = self.state_estimate[1, 0]
        self.current_pose[2] = self.state_estimate[2, 0]