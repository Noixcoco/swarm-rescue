# components/communicator.py
import numpy as np
import math
from ..core.constants import DEATH_TIMEOUT, CONFIRMATION_TIMEOUT, MAX_COMM_RANGE

class Communicator:
    def __init__(self, drone_id):
        self.id = drone_id
        self.drone_last_heard = {}      # {id: {"iteration": int, "position": (x,y)}}
        self.suspected_dead_drones = {} # {id: {"first_timeout_iter": int, "position": (x,y)}}
        self.declared_dead_drones = set()
        self.other_drones_pos = []
        
        # --- FIX: Initialisation des variables de suivi pour build_message ---
        self._last_wounded_list = None
        self._last_rescue_list = None
        self.shared_frontier_barycenters = []

    def build_message(self, iteration, pose, state, target_point, perception, mapper, grasper):
        """
        Version optimisée du prototype pour construire le message de communication.
        """
        # 1. Récupération des positions des blessés actuellement saisis
        grasped_positions = set(
            (w.position[0], w.position[1]) 
            for w in getattr(grasper, "grasped_wounded_persons", []) 
            if hasattr(w, "position")
        )

        # 2. Filtrage : ne diffuser que les blessés non saisis
        wounded_list = [
            w for w in perception.wounded
            if w not in grasped_positions
        ]

        # 3. Message de base (envoyé à chaque itération)
        message = {
            "drone_id": self.id,
            "drone_pose": pose.tolist(),
            "wounded_assignments": perception.assignments,
            "grasped_wounded": list(grasped_positions),
        }

        # 4. Ajout conditionnel de la liste des blessés (si changement ou toutes les 5 itérations)
        if self._last_wounded_list != wounded_list or iteration % 5 == 0:
            message["wounded_list"] = wounded_list
            self._last_wounded_list = wounded_list

        # 5. Ajout conditionnel de la zone de secours (si changement ou toutes les 10 itérations)
        if self._last_rescue_list != perception.rescue_zones or iteration % 10 == 0:
            message["rescue_list"] = perception.rescue_zones
            self._last_rescue_list = perception.rescue_zones

        # 6. Données de grille (toutes les 20 itérations)
        if iteration % 20 == 0:
            message["grid_data"] = mapper.grid.grid.copy()

        # 7. Blessés supprimés
        if perception.removed_wounded:
            message["removed_wounded"] = list(perception.removed_wounded)

        # 8. Clusters de frontières (toutes les 10 itérations)
        if iteration % 10 == 0 and mapper.frontier_clusters:
            message["frontier_clusters"] = [
                {"barycenter": cluster["barycenter"].tolist()}
                for cluster in mapper.frontier_clusters
            ]

        # 9. Barycentres assignés (uniquement en exploration)
        from ..my_drone import MyDrone # Import local pour éviter les cycles
        if state == MyDrone.Activity.EXPLORING and target_point is not None:
            message["assigned_barycenters"] = {
                str(self.id): np.array(target_point).tolist()
            }

        return message

    def process(self, messages, mapper, perception, iteration, my_pose):
        """Traitement complet des messages et détection de mort."""
        self.other_drones_pos = []
        dedup_radius = 50.0
        
        for msg_data in messages:
            msg = msg_data[1] if isinstance(msg_data, tuple) else msg_data
            other_id = msg.get('drone_id')
            if other_id is None or str(other_id) == str(self.id): continue

            pos = msg.get('drone_pose')
            if pos:
                # Anti Faux-Positif : Si on l'entend, il n'est pas mort
                if other_id in self.declared_dead_drones:
                    self.declared_dead_drones.remove(other_id)
                
                self.drone_last_heard[other_id] = {
                    "iteration": iteration, 
                    "position": (pos[0], pos[1])
                }
                self.other_drones_pos.append((np.array(pos[:2]), other_id))

            # Fusion de grille pondérée (70/30)
            if iteration % 20 == 0 and "grid_data" in msg:
                other_grid = np.array(msg["grid_data"])
                mapper.grid.grid = 0.7 * mapper.grid.grid + 0.3 * other_grid

            # Traitement des blessés et zones de secours
            if "wounded_list" in msg:
                for w in msg["wounded_list"]:
                    perception.add_wounded(tuple(w)) # Nécessite une méthode add_wounded dans Perception

            perception.assignments.update(msg.get('wounded_assignments', {}))

            # Synchronisation des blessés retirés
            if "removed_wounded" in msg:
                for rw in msg["removed_wounded"]:
                    perception.removed_wounded.add(tuple(rw))
                    perception.wounded = [w for w in perception.wounded if math.hypot(w[0]-rw[0], w[1]-rw[1]) > dedup_radius]

            # Mise à jour des frontières partagées pour ExplorationBehavior
            if "frontier_clusters" in msg:
                all_bc = [np.array(c["barycenter"]) for c in msg["frontier_clusters"]]
                for bc in all_bc:
                    if all(np.linalg.norm(bc - np.array(b)) > 60.0 for b in self.shared_frontier_barycenters):
                        self.shared_frontier_barycenters.append(bc.tolist())

        # DETECTION DE MORT (Silent Drones)
        for d_id, info in list(self.drone_last_heard.items()):
            silence = iteration - info["iteration"]
            if silence > DEATH_TIMEOUT:
                d_pos = info["position"]
                dist = math.hypot(my_pose[0]-d_pos[0], my_pose[1]-d_pos[1])
                if dist > MAX_COMM_RANGE: continue
                
                if d_id not in self.suspected_dead_drones:
                    self.suspected_dead_drones[d_id] = {"first_timeout_iter": iteration, "position": d_pos}
                else:
                    suspect = self.suspected_dead_drones[d_id]
                    if (iteration - suspect["first_timeout_iter"]) >= CONFIRMATION_TIMEOUT:
                        if d_id not in self.declared_dead_drones:
                            mapper.inject_dead_drone_wall(d_pos)
                            self.declared_dead_drones.add(d_id)