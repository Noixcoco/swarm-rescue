import numpy as np
from swarm_rescue.simulation.utils.utils import normalize_angle

class RescueBehavior:
    def __init__(self, navigator, mapper, perception, drone_id):
        self.nav = navigator
        self.mapper = mapper
        self.perception = perception
        self.drone_id = drone_id
        self.current_wounded = None

    def assign_target(self, pose):
        available = [w for w in self.perception.wounded if w not in self.perception.assignments]
        if available:
            dists = [np.linalg.norm(np.array(w) - pose[:2]) for w in available]
            idx = np.argmin(dists)
            target = available[idx]
            self.current_wounded = target
            self.perception.assignments[target] = self.drone_id
            return target
        return None

    def go_to_wounded(self, pose):
        if self.current_wounded:
            self.nav.plan(pose[:2], self.current_wounded, self.mapper.grid)

    def go_to_center(self, pose):
        if self.perception.rescue_zones:
            idx = int(self.drone_id) % len(self.perception.rescue_zones)
            zone = self.perception.rescue_zones[idx]
            path = self.nav.plan(pose[:2], zone, self.mapper.grid, safe_only=True)
            if not path:
                self.nav.current_path = self.nav.get_backtrack_path(pose[:2])

    def go_to_wounded(self, pose, target_wounded, pilot_obj, lidar_data):
        """Approche mixte : A* puis charge directe."""
        dist = np.linalg.norm(target_wounded - pose[:2])
        if dist > 40.0:
            return pilot_obj.follow_path(pose, self.nav.current_path, lidar_data)
        
        # Charge directe (Prototype)
        angle_to_w = np.atan2(target_wounded[1]-pose[1], target_wounded[0]-pose[0])
        err = normalize_angle(angle_to_w - pose[2])
        
        if abs(err) > np.radians(1.0):
            return {"forward": 0.0, "lateral": 0.0, "rotation": np.clip(5.0 * err, -1.0, 1.0)}
        return {"forward": 1.0, "lateral": 0.0, "rotation": 0.0}

    def go_to_rescue_oriented(self, pose, center_pos, grasp_angle, pilot_obj, lidar_data):
        """Retour rapide et alignement final < 15px."""
        dist = np.linalg.norm(center_pos - pose[:2])
        if dist > 15.0:
            return pilot_obj.follow_path(pose, self.nav.current_path, lidar_data)
        
        # Alignement final (Fidèle à ton snippet)
        angle_to_c = np.atan2(center_pos[1]-pose[1], center_pos[0]-pose[0])
        desired_h = normalize_angle(angle_to_c - (grasp_angle if grasp_angle else 0))
        err = normalize_angle(desired_h - pose[2])
        
        fwd = 0.6 if abs(err) < np.radians(30) else 0.3
        return {"forward": fwd, "lateral": 0.0, "rotation": np.clip(5.0 * err, -1.0, 1.0)}