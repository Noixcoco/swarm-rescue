import numpy as np
from ..core.constants import DEATH_TIMEOUT, CONFIRMATION_TIMEOUT, MAX_COMM_RANGE

class Communicator:
    def __init__(self, drone_id):
        self.id = drone_id
        self.drone_last_heard = {}      # {id: {"iteration": int, "position": (x,y)}}
        self.suspected_dead_drones = {} # {id: {"first_timeout_iter": int, "position": (x,y)}}
        self.declared_dead_drones = set()
        self.other_drones_pos = []

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
                str(self.id): target_point.tolist()
            }

        return message

def process(self, messages, mapper, perception, iteration, my_pose):
        """Traitement complet des messages et détection de mort."""
        self.other_drones_pos = []
        
        for msg_data in messages:
            msg = msg_data[1] if isinstance(msg_data, tuple) else msg_data
            other_id = msg.get('drone_id')
            if other_id == self.id: continue

            pos = msg.get('drone_pose')
            if pos:
                # Anti Faux-Positif : Si on l'entend, il n'est pas mort
                if other_id in self.declared_dead_drones:
                    self.declared_dead_drones.remove(other_id)
                    # Mapper doit nettoyer la grille ici
                
                self.drone_last_heard[other_id] = {
                    "iteration": iteration, 
                    "position": (pos[0], pos[1])
                }
                self.other_drones_pos.append((np.array(pos[:2]), other_id))

            # Fusion de grille pondérée (0.7/0.3)
            if iteration % 20 == 0 and "grid_data" in msg:
                other_grid = np.array(msg["grid_data"])
                mapper.grid.grid = 0.7 * mapper.grid.grid + 0.3 * other_grid

            # ... (Traitement wounded/rescue identique)

        # DETECTION DE MORT (Silent Drones)
        for d_id, info in list(self.drone_last_heard.items()):
            silence = iteration - info["iteration"]
            
            if silence > DEATH_TIMEOUT:
                d_pos = info["position"]
                dist = math.hypot(my_pose[0]-d_pos[0], my_pose[1]-d_pos[1])
                
                # On ne juge que si on est à portée radio théorique
                if dist > MAX_COMM_RANGE: continue
                
                # Phase de suspicion
                if d_id not in self.suspected_dead_drones:
                    self.suspected_dead_drones[d_id] = {"first_timeout_iter": iteration, "position": d_pos}
                    continue
                
                # Phase de confirmation
                suspect = self.suspected_dead_drones[d_id]
                if (iteration - suspect["first_timeout_iter"]) >= CONFIRMATION_TIMEOUT:
                    if d_id not in self.declared_dead_drones:
                        mapper.inject_dead_drone_wall(d_pos) # Marquer sur la grille
                        self.declared_dead_drones.add(d_id)