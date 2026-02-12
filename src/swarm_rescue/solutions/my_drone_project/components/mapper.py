# components/mapper.py
import numpy as np
import math
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.simulation.utils.utils import normalize_angle
from ..core.constants import SEUIL_MUR, SEUIL_FREE
from ..algorithms.slam import ScanMatcher
from ..algorithms.filters import find_frontier_clusters

class Mapper:
    def __init__(self, size_area, resolution, lidar_spec, kalman_filter):
        from examples.example_mapping import OccupancyGrid
        self.grid = OccupancyGrid(size_area_world=size_area, resolution=resolution, lidar=lidar_spec)
        self.kf = kalman_filter
        self.pose = np.array([0.0, 0.0, 0.0])
        self.kf_last_time = 0
        self.last_slam_score = 0
        self.frontier_clusters = []
        self.slam = ScanMatcher()
        self.kill_zone_grid = None

    def update(self, gps, compass, odom, vel, iteration):
        dt = 0.1
        self.kf.predict(dt, vel)

        if gps is not None and not np.isnan(gps[0]):
            self.kf.update(gps)
            self.pose[0], self.pose[1] = self.kf.get_pos()
        elif odom is not None:
            # Fallback Odométrie (Intégration du prototype)
            dist_travel, alpha, theta = odom
            travel_direction = self.pose[2] + alpha
            self.pose[0] += dist_travel * math.cos(travel_direction)
            self.pose[1] += dist_travel * math.sin(travel_direction)
            # Update Kalman state to match odom
            self.kf.state[0], self.kf.state[1] = self.pose[0], self.pose[1]

        if compass is not None:
            self.pose[2] = compass
        elif odom is not None:
            self.pose[2] = normalize_angle(self.pose[2] + odom[2])

        est_pose = Pose(np.asarray([self.pose[0], self.pose[1]]), self.pose[2])
        self.grid.update_grid(pose=est_pose)

        if iteration % 20 == 0:
            self.frontier_clusters = find_frontier_clusters(self.grid.grid, self.grid)

    def inject_dead_drone_wall(self, dead_pos):
        # Création d'un mur virtuel autour d'un drone mort
        radius_cells = int(50.0 / self.resolution)
        grid_pos = self.grid._conv_world_to_grid(dead_pos[0], dead_pos[1])
        r_idx, c_idx = int(grid_pos[0]), int(grid_pos[1])
        
        y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
        mask = x**2 + y**2 <= radius_cells**2
        
        r_min, r_max = max(0, r_idx-radius_cells), min(self.grid.grid.shape[0], r_idx+radius_cells+1)
        c_min, c_max = max(0, c_idx-radius_cells), min(self.grid.grid.shape[1], c_idx+radius_cells+1)
        
        # Simplifié pour le mur virtuel
        self.grid.grid[r_min:r_max, c_min:c_max] = 100.0

    def update_return_area(self, current_pose):
        """Gère le point de retour unique par calcul de barycentre."""
        # On stocke les points bruts dans une liste temporaire interne
        if not hasattr(self, '_raw_return_points'):
            self._raw_return_points = []
        
        self._raw_return_points.append(current_pose[:2])
        
        # Calcul du barycentre
        xs = [p[0] for p in self._raw_return_points]
        ys = [p[1] for p in self._raw_return_points]
        bx = float(sum(xs) / len(xs))
        by = float(sum(ys) / len(ys))
        
        # Le point officiel utilisé pour la navigation
        self.return_area_points = [(bx, by)]

    def apply_kill_zones(self):
        """Applique les cercles d'exclusion pour les drones morts."""
        # Cette méthode appelle la logique inject_dead_drone_wall 
        # pour chaque drone confirmé mort dans la perception.
        pass # La logique est déjà définie dans inject_dead_drone_wall

    def update_pose(self, gps_pos, compass_angle, lidar_data, odom_data, iteration):
        """Localisation hybride : Kalman + SLAM."""
        # 1. Calcul du dt
        current_time = iteration * 0.1
        kf_dt = current_time - self.kf_last_time if self.kf_last_time > 0 else 0.1
        self.kf_last_time = current_time

        if compass_angle is not None:
            self.pose[2] = compass_angle

        # 2. GPS Disponible : Kalman Update
        if gps_pos is not None and not np.isnan(gps_pos[0]):
            self.kf.predict(kf_dt)
            self.kf.update(gps_pos)
            self.pose[0], self.pose[1] = self.kf.get_pos()

        # 3. No GPS : Odometry Dead Reckoning
        elif odom_data is not None:
            dist_travel, alpha, theta = odom_data
            if compass_angle is None:
                self.pose[2] = normalize_angle(self.pose[2] + theta)
            
            travel_direction = self.pose[2] + alpha
            dx_world = dist_travel * math.cos(travel_direction)
            dy_world = dist_travel * math.sin(travel_direction)
            
            # Mise à jour Kalman avec augmentation d'incertitude
            if self.kf.initialized:
                # Prediction avec vitesse odom
                vx, vy = dx_world/kf_dt, dy_world/kf_dt
                self.kf.state[2] = 0.8 * vx + 0.2 * self.kf.state[2]
                self.kf.state[3] = 0.8 * vy + 0.2 * self.kf.state[3]
                self.kf.predict(kf_dt)
                self.kf.P += self.kf.Q * 3.0 # Augmenter l'incertitude sans GPS
                self.pose[0], self.pose[1] = self.kf.get_pos()
            else:
                self.pose[0] += dx_world
                self.pose[1] += dy_world

        # 4. SLAM (Scan Matching) toutes les 15 itérations
        gps_missing = (gps_pos is None or np.isnan(gps_pos[0]))
        if gps_missing and lidar_data is not None and iteration > 50 and iteration % 15 == 0:
            # Choix du rayon selon la confiance précédente
            radius = 20.0 if self.last_slam_score > 300 else 40.0
            
            guess = np.array([self.pose[0], self.pose[1], self.pose[2]])
            corrected = self.slam.run_scan_matching(
                guess, lidar_data, self.lidar_spec.ray_angles, self.grid,
                search_radius=radius, angle_search=0.1
            )
            
            self.last_slam_score = self.slam.calculate_scan_score(
                corrected, lidar_data, self.lidar_spec.ray_angles, self.grid
            )

            # Anti-Saut (Gating)
            if np.linalg.norm(corrected[:2] - guess[:2]) < 25.0:
                self.pose = corrected
                self.kf.state[0], self.kf.state[1] = corrected[0], corrected[1]

    def mark_kill_zone(self, death_pos, drone_id):
        """Marque une zone de mort permanente sur une grille dédiée."""
        if self.kill_zone_grid is None:
            self.kill_zone_grid = np.zeros_like(self.grid.grid)
        
        # Taille basée sur ta logique de vitesse/timeout (150px * 1.5)
        square_size = 225.0 
        grid_pos = self.grid._conv_world_to_grid(death_pos[0], death_pos[1])
        size_cells = int(square_size / self.resolution)
        h = size_cells // 2
        
        y0, y1 = max(0, int(grid_pos[0])-h), min(self.grid.grid.shape[0], int(grid_pos[0])+h)
        x0, x1 = max(0, int(grid_pos[1])-h), min(self.grid.grid.shape[1], int(grid_pos[1])+h)
        
        self.kill_zone_grid[y0:y1, x0:x1] = 1.0
        print(f"[{drone_id}] Kill zone marked at {death_pos}")

    def clear_kill_zone(self, pos):
        """Supprime une zone de mort (cas de résurrection)."""
        if self.kill_zone_grid is None: return
        grid_pos = self.grid._conv_world_to_grid(pos[0], pos[1])
        h = int(112 / self.resolution) # Moitié de 225px
        y0, y1 = max(0, int(grid_pos[0])-h), min(self.grid.grid.shape[0], int(grid_pos[0])+h)
        x0, x1 = max(0, int(grid_pos[1])-h), min(self.grid.grid.shape[1], int(grid_pos[1])+h)
        self.kill_zone_grid[y0:y1, x0:x1] = 0.0