import numpy as np
from enum import Enum
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.controller import CommandsDict

# --- IMPORTS MODULAIRES ---
from swarm_rescue.solutions.my_drone_project.core.constants import *
from swarm_rescue.solutions.my_drone_project.components.mapper import Mapper
from swarm_rescue.solutions.my_drone_project.components.pilot import Pilot
from swarm_rescue.solutions.my_drone_project.components.perception import Perception
from swarm_rescue.solutions.my_drone_project.components.communicator import Communicator
from swarm_rescue.solutions.my_drone_project.components.navigator import Navigator
from swarm_rescue.solutions.my_drone_project.components.debug_drawer import DebugDrawer
from swarm_rescue.solutions.my_drone_project.behaviors.exploration import ExplorationBehavior
from swarm_rescue.solutions.my_drone_project.behaviors.rescue import RescueBehavior
import math
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.solutions.my_drone_project.algorithms.kalman import KalmanFilter
from swarm_rescue.solutions.my_drone_project.behaviors.exploration import ExplorationBehavior





class MyDrone(DroneAbstract):
    class Activity(Enum):
        EXPLORING = 1
        GOING_TO_WOUNDED = 2
        GOING_TO_RESCUE = 3
        GOING_TO_RETURN_AREA = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.mapper = Mapper(self.size_area, RESOLUTION, self.lidar(), KalmanFilter([0,0]))
        self.pilot = Pilot()
        self.perception = Perception(self.identifier)
        self.comm = Communicator(self.identifier)
        self.nav = Navigator(RESOLUTION)
        self.drawer = DebugDrawer(self.identifier)
        
        self.behavior_explore = ExplorationBehavior(self.nav, self.mapper)
        self.behavior_rescue = RescueBehavior(self.nav, self.mapper, self.perception, self.identifier)
        
        self.state = self.Activity.EXPLORING
        self.iteration = 0
        self.grasped_wounded_angle = None

    def define_message_for_all(self):
        """
        Appelé par le simulateur à chaque itération.
        Délègue la création du message au composant Communicator.
        """
        # On ne traite l'envoi que toutes les 5 itérations pour économiser de la bande passante
        if self.iteration % 5 != 0:
            return None
            
        # On appelle build_message du composant Communicator
        return self.comm.build_message(
            iteration=self.iteration,
            pose=self.mapper.pose,
            state=self.state,
            target_point=getattr(self.behavior_explore, "target", None),
            perception=self.perception,
            mapper=self.mapper,
            grasper=self.grasper
        )

    def control(self) -> CommandsDict:
        self.iteration += 1
        lidar_data = self.lidar_values()
        
        # 0. SÉCURITÉ : Vérification de destruction du drone
        if lidar_data is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # 1. PERCEPTION & LOCALISATION
        # Mise à jour de la pose (GPS / Kalman / SLAM / Odométrie)
        self.mapper.update_pose(
            self.measured_gps_position(), 
            self.measured_compass_angle(), 
            lidar_data, 
            self.odometer_values(), 
            self.iteration
        )
        
        # Perception sémantique (Blessés, Zones de secours, Drones morts)
        alive_drones_ids = [info[1] for info in self.comm.other_drones_pos]
        self.perception.scan(self.semantic_values(), self.mapper.pose, alive_drones_ids, self.iteration)
        
        # Marquer les zones de mort sur la carte pour les drones immobiles
        for dead_pos in self.perception.dead_drones:
            self.mapper.mark_kill_zone(dead_pos, self.identifier)
        
        # Enregistrement du fil d'Ariane (Breadcrumbs) pour le retour de secours
        if self.state != self.Activity.GOING_TO_RESCUE:
            self.nav.update_breadcrumbs(self.mapper.pose)

        # 2. COMMUNICATION
        self.comm.process(self.communicator.received_messages, self.mapper, self.perception, self.iteration, self.mapper.pose)

        # 3. DÉCISION (Machine à états)
        # Gestion du blocage (Stuck) : si bloqué, on cherche une zone libre
        if self.pilot.check_and_handle_stuck(self.mapper.pose, self.iteration, self.mapper.grid, self.nav.plan):
            if self.pilot.unstuck_target:
                self.nav.plan(self.mapper.pose[:2], self.pilot.unstuck_target, self.mapper.grid, iteration=self.iteration)
            # On suit immédiatement le chemin de dégagement
            pilot_out = self.pilot.follow_path(self.mapper.pose, self.nav.current_path, lidar_data)
            self.nav.current_path = pilot_out["updated_path"]
            return pilot_out["command"]

        # Logique stratégique selon l'état actuel
        if self.state == self.Activity.EXPLORING:
            # Tenter d'assigner un blessé trouvé par soi ou les autres
            target_wounded = self.behavior_rescue.assign_target(self.mapper.pose)
            if target_wounded:
                self.state = self.Activity.GOING_TO_WOUNDED
                self.nav.plan(self.mapper.pose[:2], target_wounded, self.mapper.grid, iteration=self.iteration)
            else:
                # On vérifie si l'exploration est terminée
                frontiers = self.mapper.frontier_clusters # Cibles locales
                shared_frontiers = self.comm.shared_frontier_barycenters # Cibles des autres
                
                # Condition de fin : plus aucune frontière ET plus aucun blessé connu
                other_assignments = {str(w_pos): d_id for w_pos, d_id in self.perception.assignments.items()}
                if not frontiers and not shared_frontiers and not self.perception.wounded or not self.behavior_explore.execute(self.mapper.pose, self.iteration, other_assignments, self.comm.other_drones_pos):
                    if self.mapper.return_area_points:
                        print(f"[{self.identifier}] EXPLORATION 100% - Retour à la base")
                        self.state = self.Activity.GOING_TO_RETURN_AREA
                        # Planification vers le point de retour
                        self.nav.plan(self.mapper.pose[:2], self.mapper.return_area_points[0], self.mapper.grid)
                else:
                    # Continuer l'exploration normale
                    self.behavior_explore.execute(self.mapper.pose, self.iteration, other_assignments, self.comm.other_drones_pos)
            
        elif self.state == self.Activity.GOING_TO_WOUNDED:
            # Capture de l'angle sémantique précis juste avant la saisie (Crucial pour l'approche centre)
            dist_to_w = np.linalg.norm(self.behavior_rescue.current_wounded - self.mapper.pose[:2])
            if dist_to_w < 40.0:
                self.grasped_wounded_angle = self.perception.get_target_semantic_angle('WOUNDED')
            
            if self.grasper.grasped_wounded_persons:
                self.state = self.Activity.GOING_TO_RESCUE
                self.behavior_rescue.go_to_center(self.mapper.pose) # Planifie le retour sécurisé
            elif self.iteration % 30 == 0:
                # Recalculer le chemin vers le blessé s'il a bougé ou toutes les 3s
                self.nav.plan(self.mapper.pose[:2], self.behavior_rescue.current_wounded, self.mapper.grid, iteration=self.iteration)

        elif self.state == self.Activity.GOING_TO_RESCUE:
            if not self.grasper.grasped_wounded_persons:
                self.state = self.Activity.EXPLORING
                self.grasped_wounded_angle = None
            elif self.iteration % 30 == 0:
                self.behavior_rescue.go_to_center(self.mapper.pose)

        elif self.state == self.Activity.GOING_TO_RETURN_AREA:
            dist_base = np.linalg.norm(np.array(self.mapper.return_area_points[0]) - self.mapper.pose[:2])
            
            if dist_base < 10.0:
                # Arrêt complet une fois dans la zone
                command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
            else:
                # Navigation normale vers la base
                pilot_out = self.pilot.follow_path(self.mapper.pose, self.nav.current_path, lidar_data)
                command.update(pilot_out["command"])
                self.nav.current_path = pilot_out["updated_path"]

        # 4. PILOTAGE (Calcul des commandes moteurs)
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        gps_pos = self.measured_gps_position()
        has_gps = gps_pos is not None and not np.isnan(gps_pos[0])

        if self.state == self.Activity.GOING_TO_RESCUE and self.perception.rescue_zones:
            # Approche finale chirurgicale pour le dépôt (< 15px)
            dist_rz = np.linalg.norm(np.array(self.perception.rescue_zones[0]) - self.mapper.pose[:2])
            if dist_rz < 15.0:
                command.update(self._oriented_approach())
            else:
                pilot_out = self.pilot.follow_path(self.mapper.pose, self.nav.current_path, lidar_data)
                command.update(pilot_out["command"])
                self.nav.current_path = pilot_out["updated_path"]
        
        elif self.state == self.Activity.EXPLORING and not has_gps:
            # Comportement de secours en zone "No GPS" : suivi de mur
            command.update(self.pilot.wall_follower_control(lidar_data))
            
        else:
            # Navigation A* normale avec Lookahead adaptatif
            pilot_out = self.pilot.follow_path(self.mapper.pose, self.nav.current_path, lidar_data)
            command.update(pilot_out["command"])
            self.nav.current_path = pilot_out["updated_path"]

        # 5. POST-PROCESS (Actionneurs et Sécurité Physique)
        # Activation du grasper uniquement si nécessaire
        command["grasper"] = 1 if self.state in [self.Activity.GOING_TO_WOUNDED, self.Activity.GOING_TO_RESCUE] else 0
        
        # Réflexes de sécurité (Potential Fields) : évitement des murs et des autres drones
        command = self.pilot.wall_avoidance(command, lidar_data, self.lidar().ray_angles)
        command = self.pilot.drone_repulsion(command, self.mapper.pose, self.comm.other_drones_pos)

        # 6. DESSIN DEBUG
        if self.iteration % 50 == 0:
            self.drawer.draw(self.mapper.pose, self._half_size_array, self.state.name, self.nav, self.mapper, self.perception)

        return command
    
    def _oriented_approach(self):
        """
        Approche orientée fidèle au prototype original.
        Aligne le drone pour que le blessé (même excentré) entre droit dans la zone.
        """
        target_point = np.array(self.perception.rescue_zones[0])
        target_vector = target_point - self.mapper.pose[:2]
        dist_to_center = np.linalg.norm(target_vector)
        
        # 1. Calcul de l'angle global vers le centre
        angle_to_center = math.atan2(target_vector[1], target_vector[0])
        
        # 2. Utilisation de l'angle de saisie mémorisé
        grasp_angle = getattr(self, "grasped_wounded_angle", 0.0)
        if grasp_angle is None: grasp_angle = 0.0
            
        # target_orientation : angle où le drone + blessé sont alignés vers le centre
        target_orientation = normalize_angle(angle_to_center - grasp_angle)
        angle_error = normalize_angle(target_orientation - self.mapper.pose[2])

        # 3. Logique de commande (Turn then Push)
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        
        if abs(angle_error) > 0.05:  # Phase d'alignement
            command["rotation"] = np.clip(PID_ROT_KP * angle_error, -0.6, 0.6)
            # On avance un peu pour ne pas stagner
            command["forward"] = 0.2 
        else: # Phase de poussée
            if dist_to_center > 15.0:
                # Si le blessé est devant (angle < 90°), on avance, sinon on recule
                command["forward"] = 1.0 if abs(grasp_angle) < math.pi/2 else -0.5
            else:
                command["forward"] = 0.0 # On est arrivé au centre
                
        return command