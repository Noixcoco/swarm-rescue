# components/navigator.py
import numpy as np
from ..algorithms.astar import AStarPlanner #

class Navigator:
    def __init__(self, resolution):
        self.planner = AStarPlanner(resolution) #
        self.current_path = [] #
        self.breadcrumbs = [] #
        self.last_crumb = None #

    def plan(self, start, goal, grid_obj, safe_only=False, iteration=0, kill_zone_grid=None):
        """Appelle le planificateur A* avec les paramètres de sécurité."""
        self.current_path = self.planner.compute_path(
            start, goal, grid_obj, 
            explored_only=safe_only, 
            iteration=iteration, 
            kill_zone_grid=kill_zone_grid
        ) #
        return self.current_path

    def update_breadcrumbs(self, pose):
        """Enregistre la position actuelle si le drone s'est assez déplacé."""
        if self.last_crumb is None or np.linalg.norm(pose[:2] - self.last_crumb) > 100:
            self.breadcrumbs.append(pose[:2]) #
            self.last_crumb = pose[:2] #

    def get_backtrack_path(self, current_pos):
        """Génère un chemin de retour basé sur l'historique (Hansel & Gretel)."""
        if not self.breadcrumbs: return [] #
        arr = np.array(self.breadcrumbs) #
        dists = np.linalg.norm(arr - current_pos, axis=1) #
        idx = np.argmin(dists) #
        # Inverse l'ordre pour revenir en arrière
        path = list(arr[:idx+1])[::-1] #
        return [np.array(p) for p in path] #