# behaviors/exploration.py
import numpy as np
from ..components.mapper import find_frontier_clusters

class ExplorationBehavior:
    def __init__(self, navigator, mapper):
        self.nav = navigator
        self.mapper = mapper
        self.target = None

    def execute(self, pose, iteration, other_assignments, other_drones_pos):
        """Sélection de cible par fonction de coût (Prototype)."""
        shared_clusters = self.mapper.frontier_clusters # Issu de la comm
        if not shared_clusters:
            return

        barycenters = [np.array(c["barycenter"]) for c in shared_clusters]
        scored_targets = []
        min_separation = 300.0

        for bc in barycenters:
            distance = np.linalg.norm(bc - pose[:2])
            
            # 1. Pénalité de Conflit (si un autre drone vise cet endroit)
            conflict_penalty = 0.0
            for other_target in other_assignments.values():
                if np.linalg.norm(bc - np.array(other_target)) < min_separation:
                    conflict_penalty += 5000.0

            # 2. Pénalité de proximité (éviter les drones physiques)
            drone_penalty = 0.0
            for d_pos, _ in other_drones_pos:
                d = np.linalg.norm(bc - d_pos[:2])
                if d < 200.0:
                    drone_penalty += 300.0 / (d + 1.0)

            # 3. Bonus de taille de cluster
            cluster_size = 10
            for cluster in self.mapper.frontier_clusters:
                if np.linalg.norm(cluster["barycenter"] - bc) < 20:
                    cluster_size = cluster["size"]
                    break

            # Fonction de Coût Finale
            cost = distance + conflict_penalty + drone_penalty - (cluster_size * 2.0)
            scored_targets.append((cost, bc))

        # Tri par coût croissant
        scored_targets.sort(key=lambda x: x[0])

        # Tentative de planification sur les meilleures cibles
        for cost, target in scored_targets:
            path = self.nav.plan(pose[:2], target, self.mapper.grid, iteration=iteration)
            if path:
                self.target = target
                return True
        return False