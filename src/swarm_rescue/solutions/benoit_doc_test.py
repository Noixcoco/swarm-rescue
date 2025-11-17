from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from pynput import keyboard
import numpy as np


# Variables globales pour le contrôle
key_states = {
    "forward": 0.0,
    "lateral": 0.0,
    "rotation": 0.0,
    "grasper": 0
}

# Fonction appelée quand une touche est pressée
def on_press(key):
    try:
        if key == keyboard.Key.up:
            key_states["forward"] = 1.0
        elif key == keyboard.Key.down:
            key_states["forward"] = -1.0
        elif key == keyboard.Key.left:
            key_states["lateral"] = 1.0
        elif key == keyboard.Key.right:
            key_states["lateral"] = -1.0
        elif key.char == 'a':
            key_states["rotation"] = 1.0
        elif key.char == 'd':
            key_states["rotation"] = -1.0
        elif key.char == 'g':
            key_states["grasper"] = 1  # Exemple : attraper
    except AttributeError:
        pass

# Fonction appelée quand une touche est relâchée
def on_release(key):
    try:
        if key in [keyboard.Key.up, keyboard.Key.down]:
            key_states["forward"] = 0.0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            key_states["lateral"] = 0.0
        elif hasattr(key, "char") and key.char in ['a', 'd']:
            key_states["rotation"] = 0.0
        elif hasattr(key, "char") and key.char == 'g':
            key_states["grasper"] = 0
    except AttributeError:
        pass


# Lancer le listener clavier dans un thread séparé
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

### Définition de la carte du monde ###
MapDict = dict[tuple[int, int], int]
world_map: MapDict = {}

#max_range = self.get_lidar_sensor().max_range
max_range=300



class MyGreatDrone(DroneAbstract):
    def control(self):


        #on met à jour la carte du monde avec les infos du lidar
        drone_position = self.measured_gps_position()
        values = self.lidar_values()
        ray_angles = [n*(np.pi/180)/len(values) for n in range(len(values))]  # angles des rayons du lidar
        for i in range(len(values)):
            if values[i]!=0:
                for j in range(int(values[i])):  # On marque tous les points jusqu'à la distance détectée comme libres
                    distance = j
                    angle = ray_angles[i] + self.measured_compass_angle()
                    world_map[(drone_position[0] + int(distance * np.cos(angle)),
                                drone_position[1] + int(distance * np.sin(angle)))] = 0  # 0 pour libre
                if values[i]<max_range:
                    # Marquer le point à la distance détectée comme un obstacle
                    world_map[(drone_position[0] + int(values[i] * np.cos(angle)),
                                drone_position[1] + int(values[i] * np.sin(angle)))] = 1  # 1 pour obstacle
        
        # on veut maintenant trouver une frontière interessante vers laquelle se diriger
        # le choix de la frontière se fait sur le maximum du score score = α× gain_info(plus grande frontière) − β× distance (entre frontière et drone)
        best_score = float('-inf') ## on a ici supposé qu'on va forcément trouver un point, ce qui peut être faux mais à gérer plus tard
        best_target = None
        alpha = 1.0
        beta = 0.5
        for (x, y), value in world_map.items():
            if value == 0:  # Si c'est un point libre
                # Calculer le gain d'information (taille de la frontière)
                gain_info = 0
                for dx in range(-5, 6): # valeurs arbitraires pour définir une zone autour du point, à revoir
                    for dy in range(-5, 6):
                        if (x + dx, y + dy) not in world_map:
                            gain_info += 1
                # Calculer la distance entre le drone et ce point
                distance = ((drone_position[0] - x) ** 2 + (drone_position[1] - y) ** 2) ** 0.5
                # Calculer le score
                score = alpha * gain_info - beta * distance
                if score > best_score:
                    best_score = score
                    best_target = (x, y)

                




        # Retourne simplement les valeurs en fonction des touches
        return {
            "forward": key_states["forward"],
            "lateral": key_states["lateral"],
            "rotation": key_states["rotation"],
            "grasper": key_states["grasper"]
        }


    def define_message_for_all(self) -> None:
        pass
