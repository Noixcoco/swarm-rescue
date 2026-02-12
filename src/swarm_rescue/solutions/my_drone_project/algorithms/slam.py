# algorithms/slam.py
import numpy as np
import math
from swarm_rescue.simulation.utils.utils import normalize_angle
from ..core.constants import MAX_RANGE_LIDAR_SENSOR

class ScanMatcher:
    def calculate_scan_score(self, candidate_pose, lidar_distances, ray_angles, grid_obj):
        """Calcule la cohérence entre mesures et carte."""
        score = 0
        # Sous-échantillonnage pour la performance (1 rayon sur 5)
        distances = lidar_distances[::5]
        angles = ray_angles[::5]
        
        x_r, y_r, theta_r = candidate_pose
        global_angles = angles + theta_r
        
        # Projection des points lidar dans le monde
        x_points = x_r + distances * np.cos(global_angles)
        y_points = y_r + distances * np.sin(global_angles)
        
        for i in range(len(x_points)):
            if np.isnan(distances[i]) or distances[i] >= MAX_RANGE_LIDAR_SENSOR:
                continue 
            
            grid_pos = grid_obj._conv_world_to_grid(x_points[i], y_points[i])
            gy, gx = int(grid_pos[0]), int(grid_pos[1])
            
            if 0 <= gy < grid_obj.grid.shape[0] and 0 <= gx < grid_obj.grid.shape[1]:
                # On ajoute la valeur de la cellule (plus c'est un mur, plus le score monte)
                score += grid_obj.grid[gy, gx]
        return score

    def run_scan_matching(self, initial_pose, lidar_data, ray_angles, grid_obj, search_radius=10.0, angle_search=0.05):
        """Recherche locale de la meilleure pose."""
        best_pose = np.copy(initial_pose)
        best_score = self.calculate_scan_score(best_pose, lidar_data, ray_angles, grid_obj)
        
        step_size = 5.0      # 0.5 pixel
        angle_step = 0.02    # ~1 deg
        
        for dx in np.arange(-search_radius, search_radius + 0.1, step_size):
            for dy in np.arange(-search_radius, search_radius + 0.1, step_size):
                for dtheta in np.arange(-angle_search, angle_search + 0.001, angle_step):
                    candidate = np.array([
                        initial_pose[0] + dx, 
                        initial_pose[1] + dy, 
                        normalize_angle(initial_pose[2] + dtheta)
                    ])
                    score = self.calculate_scan_score(candidate, lidar_data, ray_angles, grid_obj)
                    if score > best_score:
                        best_score = score
                        best_pose = candidate
        return best_pose