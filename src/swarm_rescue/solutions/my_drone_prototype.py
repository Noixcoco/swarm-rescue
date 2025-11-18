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

                # PID translation
        self.prev_diff_position = np.zeros(2)  # dérivée pour translation
        self.Kp_pos = 1.6
        self.Kd_pos = 11.0

        
        # --- CHEMIN CORRIGÉ (Waypoints pour suivre la ligne verte) ---
        # NOTE: Ces points sont choisis pour simuler un chemin A* qui contourne les murs.
        self.path = [
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
            
        ]

        self.iteration: int = 0
        resolution = 8
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
        
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
        
        lidar_data = self.lidar_values()
        if lidar_data is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # --- 2. STRATÉGIE (DÉSACTIVÉE) ---
        # Toute la logique FBE et A* est désactivée.


        
        # Si le chemin n'a jamais été initialisé
        if not self.path and len(self.path) == 0:
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

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())

        self.grid.update_grid(pose=self.estimated_pose)
        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid,
                              self.estimated_pose,
                              title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")

        return command

        if self.current_pose is None or np.any(np.isnan(self.current_pose)):
            print("⚠️ Pose non initialisée, attente de données LIDAR...")
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    # --------------------------------------------------------------------------
    # FONCTION DE DESSIN
    # --------------------------------------------------------------------------
    
    def draw_bottom_layer(self):
        """ Dessine le chemin A* (self.path) en pointillés verts. """
        if self.path:
            points_list = []
            
            # 1. Ajouter la position actuelle du drone au début du chemin
            start_point = self.current_pose[:2] + self._half_size_array
            points_list.append((start_point[0], start_point[1]))

            # 2. Ajouter les waypoints restants
            for point_world in self.path:
                point_arcade = point_world + self._half_size_array
                points_list.append((point_arcade[0], point_arcade[1]))

            if len(points_list) > 1:
                # Utiliser 'draw_line_strip' pour dessiner le chemin complet
                arcade.draw_line_strip(points_list, arcade.color.GREEN, 5)
        self.display_pose()


    # --------------------------------------------------------------------------
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
        if dist_to_goal < 50:
            print("✅ Mission terminée : le drone est arrivé au but.")
            self.mission_done = True
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

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

    