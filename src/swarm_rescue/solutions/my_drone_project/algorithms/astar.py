import heapq
import math
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy import ndimage
from ..core.constants import SEUIL_MUR, SEUIL_FREE, COMFORT_DISTANCE_WORLD

class AStarPlanner:
    def __init__(self, resolution):
        self.resolution = resolution
        self.path_cache = {}
        self.path_cache_max_age = 50
        self.path_cache_max_size = 15
        self.path_smoothing_enabled = True

    def compute_path(self, start_world, goal_world, grid_obj, explored_only=False, iteration=0, kill_zone_grid=None):
        """
        Version fidèle au prototype original avec Soft Constraints et Lissage.
        """
        # --- CACHE SYSTEM ---
        start_key = (round(start_world[0] / 10) * 10, round(start_world[1] / 10) * 10)
        goal_key = (round(goal_world[0] / 10) * 10, round(goal_world[1] / 10) * 10)
        cache_key = (start_key, goal_key, explored_only)
        
        if cache_key in self.path_cache:
            cached_path, cached_iteration = self.path_cache[cache_key]
            if iteration - cached_iteration < self.path_cache_max_age:
                return [np.array(pt) for pt in cached_path]

        grid = grid_obj.grid.copy()

        # Conversion monde -> grille
        try:
            start = tuple(map(int, grid_obj._conv_world_to_grid(*start_world)))
            goal = tuple(map(int, grid_obj._conv_world_to_grid(*goal_world)))
        except Exception:
            return []

        # --- SEUILS ET MASQUES ---
        SEUIL_MUR_VAL = 4.01
        SEUIL_FREE_VAL = -5.0
        SEUIL_UNEXPLORED_MAX = 4.0
        SEUIL_UNEXPLORED_MIN = -4.99
    
        is_wall = (grid >= SEUIL_MUR_VAL)
        
        # Gestion de la kill_zone_grid (drones morts)
        if kill_zone_grid is not None:
            is_wall = is_wall | (kill_zone_grid == 1.0)
        
        is_explored_free = (grid < SEUIL_FREE_VAL)
        
        # Dilatation des murs (Danger Zone)
        struct = np.ones((3, 3), dtype=bool)
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)

        # --- MAP DE DISTANCE (SOFT CONSTRAINTS) ---
        dist_map = distance_transform_edt(~is_wall)
        comfort_dist_cells = COMFORT_DISTANCE_WORLD / self.resolution
        MAX_PENALTY = 100.0

        if explored_only:
            danger_zone = danger_zone | (~is_explored_free)
        
        # Dégagement départ/arrivée pour éviter d'être bloqué par la dilatation
        self._clear_safety_margin(danger_zone, start, goal, grid.shape)

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = [(fscore[start], start)]

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                # --- RECONSTRUCTION ---
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # --- SUBSAMPLING (STEP 7) ---
                STEP = 7 
                if len(path) > STEP:
                    compressed = [path[0]]
                    for i in range(1, len(path) - 1):
                        prev_v = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                        next_v = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                        if prev_v != next_v:
                            compressed.append(path[i])
                    compressed.append(path[-1])
                    path = compressed
            
                # --- SMOOTHING ---
                if len(path) > 2 and self.path_smoothing_enabled:
                    smoothed = self.smooth_path(path, danger_zone)
                else:
                    smoothed = path
                
                world_path = [np.array(grid_obj._conv_grid_to_world(*pt)) for pt in smoothed]
                
                # Mise en cache
                self.path_cache[cache_key] = (world_path, iteration)
                self._cleanup_cache()
                return world_path

            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    continue
                if danger_zone[neighbor]:
                    continue

                # --- CALCUL DU COUT AVEC PENALITE ---
                dist_to_wall = dist_map[neighbor[0], neighbor[1]]
                penalty = 0.0
                if dist_to_wall < comfort_dist_cells:
                    factor = 1.0 - (dist_to_wall / comfort_dist_cells)
                    penalty = MAX_PENALTY * factor
                
                tentative_g_score = gscore[current] + math.hypot(dx, dy) + penalty

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                            
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
                    
        return []

    def smooth_path(self, path_grid, danger_zone):
        """
        Apply Chaikin's corner-cutting algorithm to smooth the path
        """
        if len(path_grid) <= 2:
            return path_grid
        
        smoothed = [path_grid[0]]
        
        for i in range(len(path_grid) - 1):
            p0 = np.array(path_grid[i], dtype=float)
            p1 = np.array(path_grid[i + 1], dtype=float)
            
            # Points de Chaikin au quart et aux trois-quarts du segment
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            
            q_safe = self.is_point_safe(q, danger_zone)
            r_safe = self.is_point_safe(r, danger_zone)
            
            # Si les nouveaux points sont sûrs, on les ajoute, sinon on garde le point original
            if q_safe and r_safe:
                smoothed.append(tuple(q.astype(int)))
                smoothed.append(tuple(r.astype(int)))
            else:
                smoothed.append(path_grid[i + 1])
        
        # Suppression des doublons pour éviter les points de stagnation
        deduplicated = [smoothed[0]]
        for pt in smoothed[1:]:
            if pt != deduplicated[-1]:
                deduplicated.append(pt)
        
        return deduplicated

    def is_point_safe(self, point, danger_zone):
        """
        Vérifie si un point (x, y) de la grille est dans une zone sûre
        """
        x, y = int(round(point[0])), int(round(point[1]))
        
        # Vérification des limites de la grille
        if not (0 <= y < danger_zone.shape[1] and 0 <= x < danger_zone.shape[0]):
            return False
        
        # Retourne True si la cellule n'est pas marquée comme dangereuse
        return not danger_zone[x, y]

    def _clear_safety_margin(self, danger_zone, start, goal, shape):
        for pt, r in [(start, 2), (goal, 5)]:
            x, y = pt
            x0, x1 = max(0, x - r), min(shape[0], x + r + 1)
            y0, y1 = max(0, y - r), min(shape[1], y + r + 1)
            danger_zone[x0:x1, y0:y1] = False

    def _cleanup_cache(self):
        if len(self.path_cache) > self.path_cache_max_size:
            oldest_key = min(self.path_cache.keys(), key=lambda k: self.path_cache[k][1])
            self.path_cache.pop(oldest_key, None)