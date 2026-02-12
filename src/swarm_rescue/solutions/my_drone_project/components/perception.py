import math
import numpy as np
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.solutions.my_drone_project.core.constants import DEAD_DRONE_RADIUS, DEAD_CONFIRM_ITERATIONS

class Perception:
    def __init__(self, drone_id):
        self.id = drone_id
        self.wounded = []
        self.rescue_zones = []
        self.dead_drones = []
        self.suspects = [] 
        self.assignments = {}
        self.removed_wounded = set()

    def scan(self, sensors, pose, alive_pos, iteration):
        if not sensors: return
        px, py, pt = pose
        visible_drones = []
        
        for data in sensors:
            ang = normalize_angle(pt + data.angle)
            wx = px + data.distance * math.cos(ang)
            wy = py + data.distance * math.sin(ang)
            pt_w = np.array([wx, wy])

            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                if not any(np.linalg.norm(pt_w - w) < 40 for w in self.wounded):
                    is_removed = False
                    for rw in self.removed_wounded:
                        if np.linalg.norm(pt_w - np.array(rw)) < 50: is_removed = True
                    if not is_removed:
                        self.wounded.append((wx, wy))

            elif data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                if not any(np.linalg.norm(pt_w - r) < 80 for r in self.rescue_zones):
                    self.rescue_zones.append((wx, wy))
            
            elif data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                visible_drones.append(pt_w)
        
        self._detect_dead(visible_drones, alive_pos, iteration)

    def _detect_dead(self, visible, alive, iteration):
        for v in visible:
            if not any(np.linalg.norm(v - a) < DEAD_DRONE_RADIUS for a in alive):
                self._update_suspect(v, iteration)
    
    def _update_suspect(self, pos, iteration):
        matched = False
        for s in self.suspects:
            if np.linalg.norm(np.array(s['pos']) - pos) < 30:
                s['last'] = iteration
                s['pos'] = pos
                if iteration - s['first'] > DEAD_CONFIRM_ITERATIONS:
                    if not any(np.linalg.norm(np.array(d) - pos) < 30 for d in self.dead_drones):
                        self.dead_drones.append(pos)
                matched = True
                break
        if not matched:
            self.suspects.append({'pos': pos, 'first': iteration, 'last': iteration})

    def update(self, detections, pose):
        """
        Version fidèle au prototype : Détection, filtrage et fusion pondérée.
        """
        if not detections:
            return

        # Paramètres du prototype
        dedup_radius = 60.0
        alpha_update = 0.3
        
        px, py, ptheta = float(pose[0]), float(pose[1]), float(pose[2])
        newly_seen_wounded = []
        newly_seen_rescue = []

        for data in detections:
            try:
                etype = getattr(data, 'entity_type', None)
                angle = float(getattr(data, 'angle', 0.0))
                dist = float(getattr(data, 'distance', 0.0))
                
                # Conversion en coordonnées mondiales
                global_angle = normalize_angle(ptheta + angle)
                xw = px + dist * math.cos(global_angle)
                yw = py + dist * math.sin(global_angle)
                
                name = etype.name if hasattr(etype, 'name') else str(etype)
                name_up = name.upper()

                # --- LOGIQUE BLESSÉS ---
                if 'WOUNDED' in name_up:
                    # Filtrage : On ignore si déjà tenu par un autre drone
                    is_grasped = any(
                        math.hypot(xw - gx, yw - gy) < dedup_radius
                        for (gx, gy) in self.other_grasped_wounded
                    )
                    if not is_grasped:
                        newly_seen_wounded.append((xw, yw))

                # --- LOGIQUE ZONE DE SECOURS ---
                elif 'RESCUE' in name_up:
                    newly_seen_rescue.append((xw, yw))
                    
            except Exception:
                continue

        # 1. Fusion pondérée des blessés (Alpha Update)
        for nx, ny in newly_seen_wounded:
            merged = False
            for i, (wx, wy) in enumerate(self.wounded):
                if math.hypot(wx - nx, wy - ny) < dedup_radius:
                    # Mise à jour pondérée pour stabiliser la position (Prototype)
                    newx = (1.0 - alpha_update) * wx + alpha_update * nx
                    newy = (1.0 - alpha_update) * wy + alpha_update * ny
                    self.wounded[i] = (newx, newy)
                    merged = True
                    break
            if not merged:
                # Vérification finale contre les blessés déjà supprimés/sauvés
                if not any(math.hypot(nx - rx, ny - ry) < dedup_radius for rx, ry in self.removed_wounded):
                    self.wounded.append((nx, ny))

        # 2. Gestion des points de la zone de secours
        for nx, ny in newly_seen_rescue:
            self._add_or_merge_rescue_point((nx, ny), dedup_radius)

    def _add_or_merge_rescue_point(self, pt, dedup_radius):
        """Logique de stabilisation des zones de secours du prototype."""
        nx, ny = pt
        for (rx, ry) in self.rescue_zones:
            if math.hypot(rx - nx, ry - ny) < dedup_radius:
                return  # Déjà connu et stable
        
        if len(self.rescue_zones) < 5:
            self.rescue_zones.append(pt)

    def get_target_semantic_angle(self, type_name):
        """Récupère l'angle relatif brut du capteur pour un type d'entité."""
        if not hasattr(self, '_last_detections') or not self._last_detections:
            return 0.0
        for data in self._last_detections:
            etype = getattr(data, 'entity_type', None)
            name = etype.name if hasattr(etype, 'name') else str(etype)
            if type_name in name.upper():
                return float(getattr(data, 'angle', 0.0))
        return 0.0