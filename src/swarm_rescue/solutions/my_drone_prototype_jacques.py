import math
import numpy as np
from scipy.ndimage import binary_dilation
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.drone.controller import CommandsDict
import heapq

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
    """
    HARNAIS DE TEST (V-Test) - CHEMIN CORRIGÉ
    Objectif : Tester UNIQUEMENT le pilotage (la fonction 'follow_path').
    Le chemin est codé en dur pour suivre la ligne verte.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.current_pose = np.array([np.nan, np.nan, np.nan])

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


    #-----------------------------------------
    #-------------------CONTROLE
    #---------------------------------------

    def control(self) -> CommandsDict:
        """
        Cerveau : Intègre maintenant la planification A* pour éviter les murs.
        """
        # Incrément du compteur
        self.iteration += 1
        
        # --- 1. PERCEPTION & LOCALISATION ---
        self.update_pose()
        
        # Mise à jour de la grille probabiliste (Mapping)
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())
        self.grid.update_grid(pose=self.estimated_pose)

        # Acquisition Lidar
        lidar_data = self.lidar_values()
        
        # Sécurité : si pas de lidar ou mission finie, on s'arrête
        if lidar_data is None or self.mission_done:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # --- 2. STRATÉGIE (PLANIFICATION) ---
        
        # Si on n'a plus de chemin à suivre, on doit en calculer un nouveau
        if not self.path or len(self.path) < 1:
            
            # A. Trouver les zones inexplorées (Frontières)
            self.frontiers_world = self.find_safe_frontier_points()
            
            if self.frontiers_world:
                print(f"🔍 {len(self.frontiers_world)} frontières détectées.")

                # B. Choisir la frontière la plus proche (Heuristique simple)
                distances = [np.linalg.norm(f - self.current_pose[:2]) for f in self.frontiers_world]
                closest_index = np.argmin(distances)
                target_point = self.frontiers_world[closest_index]
                
                # C. Préparer la carte pour A* (Définir les zones interdites)
                grid_map = self.grid.grid
                SEUIL_MUR = 10.0 # Valeur arbitraire pour considérer qu'une case est un mur
                
                # Créer un masque binaire : True = Obstacle, False = Libre
                is_obstacle = (grid_map >= SEUIL_MUR)
                
                # D. DILATATION (Marge de sécurité)
                # C'est CRUCIAL : On épaissit les murs pour que le drone ne rase pas les murs 
                # et ne se bloque pas. 
                # 'iterations' dépend de la résolution. Si résolution=8, iterations=3 ~= 24cm de marge.
                struct = ndimage.generate_binary_structure(2, 2)
                danger_zone = binary_dilation(is_obstacle, structure=struct, iterations=3)
                
                # E. Lancer A* (De ma position -> vers la cible -> en évitant danger_zone)
                # Note: Assure-toi d'avoir ajouté la fonction planning_a_star dans ta classe !
                computed_path = self.planning_a_star(
                    start_pos=self.current_pose[:2], 
                    goal_pos=target_point, 
                    mask_grid=danger_zone
                )
                
                if computed_path:
                    print("✅ Chemin A* trouvé !")
                    self.path = computed_path
                else:
                    print("⚠️ A* n'a pas trouvé de chemin. Cible inaccessible temporairement.")
                    # Optionnel : On pourrait supprimer ce point de la liste et réessayer
            
            else:
                # Plus de frontières ? L'exploration est peut-être finie.
                # print("Plus de frontières détectées.")
                pass

        # --- 3. ACTION (PILOTAGE) ---
        
        # Si on a un chemin, on le suit
        if self.path:
            command = self.follow_path(lidar_data)
        else:
            # Sinon, on s'arrête
            command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        command["grasper"] = 0

        # --- 4. VISUALISATION (DEBUG) ---
        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid, self.estimated_pose, title="occupancy grid")
            # Optionnel : Afficher la grille zoomée
            # self.grid.display(self.grid.zoomed_grid, self.estimated_pose, title="zoomed occupancy grid")

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
        """ Dessine les frontières """
        if self.path:
            point_arcade = self.path[0] + self._half_size_array
            #point_arcade = np.array([100.0, 100.0])

            radius = 10
            blue = (0,0,255)

            arcade.draw_circle_filled(point_arcade[0],
                              point_arcade[1],
                              radius=radius,
                              color=blue)


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


# --------------------------------------------------------------------------
    # ALGORITHME A* (A-STAR)
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # ALGORITHME A* (A-STAR) - VERSION ROBUSTE
    # --------------------------------------------------------------------------
    def planning_a_star(self, start_pos, goal_pos, mask_grid):
        """
        Calcule un chemin de start_pos à goal_pos (coordonnées monde)
        mask_grid : Tableau numpy (True = Obstacle, False = Libre)
        """
        h, w = mask_grid.shape
        resolution = self.grid.resolution
        
        # --- 1. CONVERSION ROBUSTE (MONDE -> GRILLE) ---
        # On récupère la taille du monde depuis l'objet grid
        world_w, world_h = self.grid.size_area_world
        
        def world_to_grid_safe(pos_world):
            # Formule : index = (pos + demi_taille) / resolution
            # Attention : En numpy, souvent Lignes (Y) puis Colonnes (X)
            
            # Axe X (Colonnes)
            idx_x = int((pos_world[0] + world_w / 2) / resolution)
            # Axe Y (Lignes)
            idx_y = int((pos_world[1] + world_h / 2) / resolution)
            
            # CLIPPING : On force les indices à rester dans le tableau [0, taille-1]
            idx_x = np.clip(idx_x, 0, w - 1)
            idx_y = np.clip(idx_y, 0, h - 1)
            
            # Retourne (LIGNE, COLONNE) pour respecter numpy[row, col]
            return (idx_y, idx_x)

        start_node = world_to_grid_safe(start_pos)
        goal_node = world_to_grid_safe(goal_pos)

        # Debug info si ça plante encore
        # print(f"A* Start World: {start_pos} -> Grid: {start_node} (Map size: {h}x{w})")

# --- 2. GESTION DE DÉPART DANS UN MUR (Version Améliorée) ---
        if mask_grid[start_node]:
            print("A*: Départ dans zone danger, recherche étendue d'une sortie...")
            found_start = False
            # On cherche dans un rayon plus grand (ex: 10 cases = 80cm)
            search_radius = 10 
            
            # Recherche en spirale (plus efficace que les boucles simples)
            # On trie les voisins par distance au centre pour prendre le plus proche
            potential_starts = []
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    ny, nx = start_node[0] + dy, start_node[1] + dx
                    if 0 <= ny < h and 0 <= nx < w and not mask_grid[ny, nx]:
                        dist = math.sqrt(dx**2 + dy**2)
                        potential_starts.append((dist, (ny, nx)))
            
            if potential_starts:
                # On prend le point libre le plus proche
                potential_starts.sort(key=lambda x: x[0])
                start_node = potential_starts[0][1]
                found_start = True
                # print(f"A*: Nouveau départ trouvé à {potential_starts[0][0]:.1f} cases de distance.")
            
            if not found_start:
                print("A*: Vraiment coincé (ou dilatation trop forte).")
                return []

        # --- 3. ALGORITHME A* ---
        
        open_set = []
        # Heapq stocke : (F-Score, G-Score, Noeud(y,x))
        heapq.heappush(open_set, (0, 0, start_node))
        
        came_from = {}
        g_score = {start_node: 0}
        
        # Directions : 8 voisins (Diagonales incluses)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        path_found = False
        
        # Limite de sécurité pour ne pas figer le pc si la map est géante
        max_iterations = 5000 
        iter_count = 0

        while open_set:
            iter_count += 1
            if iter_count > max_iterations:
                print("A*: Trop d'itérations, abandon.")
                break

            current_f, current_g, current = heapq.heappop(open_set)

            if current == goal_node:
                path_found = True
                break

            for dy, dx in neighbors:
                neighbor = (current[0] + dy, current[1] + dx)

                # Vérifier limites
                if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w:
                    if mask_grid[neighbor]: # Obstacle
                        continue
                    
                    # Coût (1.414 pour diagonale, 1 pour cardinal)
                    weight = 1.414 if dx != 0 and dy != 0 else 1.0
                    tentative_g = current_g + weight

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        # Heuristique : Distance Euclidienne
                        h_score = math.sqrt((neighbor[0] - goal_node[0])**2 + (neighbor[1] - goal_node[1])**2)
                        f_score = tentative_g + h_score
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                        came_from[neighbor] = current

        # --- 4. RECONSTRUCTION ---
        if path_found:
            path_grid = []
            curr = goal_node
            while curr in came_from:
                path_grid.append(curr)
                curr = came_from[curr]
            path_grid.reverse()
            
            # Conversion Grille -> Monde
            path_world = []
            # On prend 1 point sur 2 pour lisser un peu
            for r, c in path_grid[::2]: 
                # r = ligne (y), c = colonne (x)
                wx = (c * resolution) - (world_w / 2)
                wy = (r * resolution) - (world_h / 2)
                path_world.append(np.array([wx, wy]))
            
            # Ajouter le point final exact demandé
            path_world.append(goal_pos)
            return path_world
        else:
            print("A*: Pas de chemin trouvé.")
            return []