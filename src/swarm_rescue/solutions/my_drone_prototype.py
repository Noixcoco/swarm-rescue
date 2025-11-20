import math
import numpy as np
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
        SEUIL_MUR = 10.0
        is_wall = (grid >= SEUIL_MUR)
        # Dilate les murs pour éviter les zones proches (moins de 4 pixels)
        struct = np.ones((9, 9), dtype=bool)  # carré 9x9 ~ rayon 4
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)

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
                # Reconstruire le chemin
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                # Conversion grille -> monde
                return [np.array(self.grid._conv_grid_to_world(*pt)) for pt in path]
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
        
        self.current_pose = np.array([0.0, 0.0, 0.0])

        # Mission accomplie ?
        self.mission_done = False

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

        
        # --- CHEMIN CORRIGÉ (Waypoints pour suivre la ligne verte) ---
        # NOTE: Ces points sont choisis pour simuler un chemin A* qui contourne les murs.
        """self.path = [
            # 1. Sortir de la zone de retour (en bas à droite)
            np.array([300.0, -200.0]),
            # 2. Descendre à droite
            np.array([-50, -200.0]),
            # 3. Contourner le mur central par le bas
            np.array([-50,200]),
            # 4. Monter (milieu de l'image)
            np.array([-300, 200.0]),
            # 5. Tourner à gauche (vers le couloir)
            np.array([-300, -200]),
            
        ]"""
        self.path = []
        self.frontiers_world = []
        
        # self.state est inutile dans ce test

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

        # --- 2. STRATÉGIE (DÉSACTIVÉE) ---
        # Replanification si le chemin est vide ou si le waypoint est atteint
        if not self.path or len(self.path) < 1:
            
            self.frontiers_world = self.find_safe_frontier_points() # Met à jour la liste des frontières
            print("frontieres : ",self.frontiers_world)
            if self.frontiers_world:
                # Scoring simplifié : choisir le point de frontière le plus proche
                
                # Calculer la distance de chaque frontière au drone
                distances = [np.linalg.norm(f - self.current_pose[:2]) for f in self.frontiers_world]
                
                # Trouver l'indice de la frontière la plus proche
                closest_index = np.argmin(distances)
                
                target_point = self.frontiers_world[closest_index]
                
                # Le chemin devient ce point de frontière
                self.path = self.creer_chemin(self.current_pose[:2], target_point)

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

        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid,
                              self.estimated_pose,
                              title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")

        return command

    # --------------------------------------------------------------------------
    # FONCTION DE DÉTECTION DES FRONTIÈRES SÛRES (Mise à jour pour self.frontiers_world)
    # --------------------------------------------------------------------------

    def find_safe_frontier_points(self) -> list:
        
        grid_map = self.grid.grid 
        
        # --- NOUVEAUX SEUILS ADAPTÉS À L'INITIALISATION À ZÉRO ---
        # Si grid.grid est initialisée à 0.0:
        
        SEUIL_FREE = -10.0
        SEUIL_MUR = 10.0        
        
        frontiers_world_temp = []
        frontiers = []

        # 1. Identifier les masques
        # is_unknown: Les cellules dont la valeur est très proche de zéro (état initial)
        #is_unknown = (grid_map < SEUIL_MUR) & (grid_map > SEUIL_FREE)
        is_unknown = (grid_map <= 10) & (grid_map >= -10)
        #print("u :", is_unknown)
        
        # is_wall: Les cellules où la probabilité d'occupation est élevée
        is_wall = (grid_map >= SEUIL_MUR)  
        
        for l in grid_map.T :
            line_str = ""
            for e in l :
                if e <= SEUIL_FREE :
                    line_str += "x"
                elif e >= SEUIL_MUR :
                    line_str += "I"
                else :
                    line_str += "o"
            print("map : ", line_str)
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
            print("frontières : ", line_str)

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
            print("sans murs : ", line_str)
        
        """# Le reste du code reste inchangé:
        # 2. Définir la zone dangereuse (murs dilatés)
        struct = generate_binary_structure(2, 2) 
        dangerous_zone = binary_dilation(is_wall, 
                                        structure=struct, 
                                        iterations=3)
        
        # 3. Détecter les frontières brutes (Inconnu adjacent à Libre)
        dilated_free = binary_dilation(is_free, structure=struct, iterations=1)
        raw_frontiers = is_unknown & dilated_free"""
        
        """print("raw_frontiers")
        for index_ligne, ligne in enumerate(raw_frontiers):
            for index_colonne, valeur in enumerate(ligne):
                if valeur:
                    print(f"[Ligne : {index_ligne}, Colonne : {index_colonne}]")"""

        """# 4. Filtrer les frontières par la zone dangereuse
        safe_frontiers_mask = raw_frontiers & (~dangerous_zone)"""

        #print("safe_frontiers_mask")
        for index_ligne, ligne in enumerate(safe_frontiers_mask):
            for index_colonne, valeur in enumerate(ligne):
                if valeur:
                    #print(f"[x : {index_ligne}, y : {index_colonne}]")
                    x_frontier_world, y_frontier_world = self.grid._conv_grid_to_world(index_ligne, index_colonne)
                    new_frontier = np.array([x_frontier_world, y_frontier_world])
                    frontiers.append(new_frontier)
                 
        return frontiers

    # --------------------------------------------------------------------------
    # FONCTION DE DESSIN
    # --------------------------------------------------------------------------
    
    def draw_bottom_layer(self):
        """ Dessine le chemin calculé (tous les points) """
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
        #print("following_path")
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
            target_speed = min(6.0, distance_to_target*0.05 - 0.03)
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
        """if dist_to_goal < 50:
            print("✅ Mission terminée : le drone est arrivé au but.")
            #self.mission_done = True
            #return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}"""

        return {"forward": forward_speed, "lateral": 0.0, "rotation": rotation_speed}

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

        if not np.isnan(gps_pos[0]):
        #if False :
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

    