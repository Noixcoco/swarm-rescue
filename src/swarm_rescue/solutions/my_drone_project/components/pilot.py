# my_drone.py
import numpy as np
import math
from enum import Enum
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import normalize_angle

from ..core.constants import *



class Pilot:
    def __init__(self):
        self.prev_angle_error = 0.0
        self.prev_speed_error = 0.0
        self.prev_lat_error = 0.0
        self.general_stuck_counter = 0
        self.last_unstuck_check_pos = None
        self.is_unstucking = False
        self.unstuck_target = None


    def _oriented_approach(self):
        """Approche finale pour déposer le blessé sans collision."""
        target = np.array(self.perception.rescue_zones[0])
        diff = target - self.mapper.pose[:2]
        dist = np.linalg.norm(diff)
        angle_to_center = math.atan2(diff[1], diff[0])
        
        # On s'aligne pour pousser le blessé vers le centre
        angle_err = normalize_angle(angle_to_center - self.mapper.pose[2])
        rot = np.clip(PID_ROT_KP * angle_err, -0.6, 0.6)
        
        fwd = 1.0 if dist > 15 else 0.0 #
        return {"forward": fwd, "lateral": 0.0, "rotation": rot}

    def wall_follower_control(self, lidar_data):
        """Contrôle de secours sans GPS (Wall Follower)."""
        if lidar_data is None: return {"forward": 0, "rotation": 0}
        
        # Logique simplifiée du prototype
        front = min(lidar_data[0:10] + lidar_data[-10:])
        left = min(lidar_data[45:135])
        
        rotation = 0.0
        if front < 50: rotation = 0.5
        elif left < 40: rotation = -0.3
        elif left > 60: rotation = 0.3
            
        return {"forward": 0.4, "lateral": 0.0, "rotation": rotation}

    def wall_avoidance(self, command, lidar_data):
        """Dernière couche de sécurité pour ne pas toucher les murs."""
        if lidar_data is None: return command
        
        # Si un obstacle est trop proche devant, on ralentit/stoppe
        front_dist = min(lidar_data[0:20] + lidar_data[-20:])
        if front_dist < 30:
            command["forward"] = min(command["forward"], 0.0)
            command["rotation"] += 0.2 # On aide à tourner
            
        return command
    
    def follow_path(self, pose, path, lidar_data):
        """
        Version fidèle au prototype avec lookahead adaptatif et gestion de vitesse.
        """
        if not path:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        # 1. LOOKAHEAD ADAPTATIF
        min_lidar_dist = min(lidar_data) if lidar_data is not None and len(lidar_data) > 0 else 999
        
        if min_lidar_dist < 40.0:
            lookahead_dist = 15.0  # Suivi serré près des obstacles
        else:
            lookahead_dist = PATH_LOOKAHEAD_DEFAULT

        # 2. SÉLECTION DE CIBLE (Pure Pursuit)
        lookahead_idx = 0
        current_pos = pose[:2]
        for i, wp in enumerate(path):
            if np.linalg.norm(wp - current_pos) > lookahead_dist:
                lookahead_idx = i
                break
        target_pos = path[min(lookahead_idx, len(path)-1)]

        # 3. CALCUL D'ERREUR (Body Frame)
        delta_pos = target_pos - current_pos
        heading = pose[2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        # Projection dans le repère local du robot
        x_err = cos_h * delta_pos[0] + sin_h * delta_pos[1]   # Longitudinal
        y_err = -sin_h * delta_pos[0] + cos_h * delta_pos[1]  # Lateral

        # 4. CONTRÔLE DE ROTATION (PID avec amortissement)
        angle_error = normalize_angle(target_angle - heading)
        deriv_angle = angle_error - self.prev_angle_error
        
        kp_rot = PID_ROT_KP
        kd_rot = PID_ROT_KD

        # Amortissement pour petites erreurs
        if abs(angle_error) < math.radians(10):
            kp_rot *= 0.6
            kd_rot *= 0.8

        rotation_speed = kp_rot * angle_error + kd_rot * deriv_angle
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error

        # 5. CONTRÔLE LATÉRAL (Correction de trajectoire)
        kp_lat, kd_lat = 0.05, 0.02
        lat_deriv = y_err - self.prev_lat_error
        lateral_cmd = kp_lat * y_err + kd_lat * lat_deriv
        
        if abs(angle_error) < 0.1:
            lateral_cmd *= 0.7  # Réduit l'effet "crabe" en ligne droite
            
        lateral_cmd = float(np.clip(lateral_cmd, -1.0, 1.0))
        self.prev_lat_error = y_err

        # 6. PROFIL DE VITESSE AVANT
        # La vitesse cible scale selon la distance longitudinale (x_err)
        target_speed = max(0.0, min(MAX_SPEED_EXPLORATION, x_err * 0.15 + 0.3))

        # Simulation de la vitesse mesurée (vitesse actuelle du drone)
        # Note: Dans le cadre modulaire, on peut passer measured_speed en argument
        # ou utiliser l'estimation du Mapper. Ici on simplifie.
        measured_speed = 0.0 # À remplacer par mapper.get_velocity_magnitude() si possible
        
        speed_error = target_speed - measured_speed
        deriv_speed = speed_error - self.prev_speed_error
        
        forward_cmd = PID_POS_KP * speed_error + PID_POS_KD * deriv_speed

        # Sécurité : Ralentissement ou arrêt en virage serré
        if abs(angle_error) > 1.0:
            forward_cmd *= 0.0
        elif abs(angle_error) > 0.4:
            forward_cmd *= 0.5

        forward_cmd = float(np.clip(forward_cmd, -1.0, 1.0))
        self.prev_speed_error = speed_error

        # 7. GESTION DES WAYPOINTS (Multi-drop)
        # On supprime du chemin les points déjà "validés" (trop proches)
        drop_until = -1
        max_check = min(5, len(path))
        for i in range(max_check):
            if np.linalg.norm(path[i] - current_pos) < CLOSE_THRESHOLD:
                drop_until = i
            else:
                break
                
        if drop_until >= 0:
            # On renvoie le chemin raccourci pour que le drone mette à jour son état
            new_path = path[drop_until+1:]
        else:
            new_path = path

        return {"command": {"forward": forward_cmd, "lateral": lateral_cmd, "rotation": rotation_speed}, 
                "updated_path": new_path}


    def wall_avoidance(self, command, lidar_data, ray_angles):
        """Réflexe de survie : repousse le drone des murs."""
        if lidar_data is None: return command
        SAFE_DIST, GAIN = 30.0, 2.0
        rep_fwd, rep_lat = 0.0, 0.0
        
        for i, dist in enumerate(lidar_data):
            if dist < SAFE_DIST:
                force = (SAFE_DIST - dist) / SAFE_DIST 
                angle = ray_angles[i]
                rep_fwd -= force * math.cos(angle)
                rep_lat -= force * math.sin(angle)
        
        if abs(rep_fwd) > 0.05 or abs(rep_lat) > 0.05:
            command["forward"] = float(np.clip(command["forward"] + rep_fwd * GAIN, -1.0, 1.0))
            command["lateral"] = float(np.clip(command["lateral"] + rep_lat * GAIN, -1.0, 1.0))
        return command

    def drone_repulsion(self, command, my_pose, other_drones):
        """Repousse le drone des autres drones (Potential Field)."""
        if not other_drones: return command
        SAFE_DIST, GAIN = 80.0, 2.0
        rep_fwd, rep_lat = 0.0, 0.0
        
        for other_pos, _ in other_drones:
            dx, dy = my_pose[0] - other_pos[0], my_pose[1] - other_pos[1]
            dist = math.hypot(dx, dy)
            if 0 < dist < SAFE_DIST:
                force = (SAFE_DIST - dist) / SAFE_DIST
                angle_local = normalize_angle(math.atan2(dy, dx) - my_pose[2])
                rep_fwd += force * math.cos(angle_local)
                rep_lat += force * math.sin(angle_local)

        command["forward"] = float(np.clip(command["forward"] + rep_fwd * GAIN, -1.0, 1.0))
        command["lateral"] = float(np.clip(command["lateral"] + rep_lat * GAIN, -1.0, 1.0))
        return command

    def check_and_handle_stuck(self, current_pose, iteration, grid_obj, planner_func):
        """Vérifie si le drone ne progresse plus et trouve une échappatoire."""
        if iteration % 10 != 0:
            return self.is_unstucking
        
        if self.last_unstuck_check_pos is None:
            self.last_unstuck_check_pos = current_pose[:2].copy()
            return False
        
        movement = np.linalg.norm(current_pose[:2] - self.last_unstuck_check_pos)
        self.last_unstuck_check_pos = current_pose[:2].copy()
        
        if movement > 10.0:
            self.general_stuck_counter = 0
            self.is_unstucking = False
            return False
        
        self.general_stuck_counter += 1
        if self.general_stuck_counter > 5:
            self.unstuck_target = self._find_free_pos(current_pose, grid_obj)
            if self.unstuck_target:
                self.is_unstucking = True
                return True
        return False

    def _find_free_pos(self, pose, grid_obj):
        """Trouve une zone libre à proximité pour se débloquer."""
        from scipy.ndimage import binary_dilation
        grid_map = grid_obj.grid
        is_free = (grid_map < -10.0)
        is_wall = (grid_map >= 5.0)
        danger = binary_dilation(is_wall, structure=np.ones((5,5)), iterations=1)
        safe_free = is_free & (~danger)
        
        if not np.any(safe_free): return None
        current_grid = grid_obj._conv_world_to_grid(pose[0], pose[1])
        safe_pos = np.argwhere(safe_free)
        dists = np.sqrt((safe_pos[:, 0] - current_grid[0])**2 + (safe_pos[:, 1] - current_grid[1])**2)
        
        for idx, d in enumerate(dists):
            if 3 <= d <= 6:
                return grid_obj._conv_grid_to_world(safe_pos[idx][0], safe_pos[idx][1])
        return None