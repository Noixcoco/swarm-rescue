import gc
import math
import pathlib
import random
import sys
from typing import List, Type

import numpy as np

# Insert the parent directory of the current file's directory into sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
from swarm_rescue.simulation.elements.rescue_center import RescueCenter
from swarm_rescue.simulation.elements.return_area import ReturnArea
from swarm_rescue.simulation.elements.sensor_disablers import ZoneType, NoGpsZone
from swarm_rescue.simulation.elements.wounded_person import WoundedPerson
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.reporting.evaluation import ZonesConfig, EvalPlan, EvalConfig
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.pose import Pose

from swarm_rescue.maps.walls_intermediate_map_1 import add_walls, add_boxes


class MapMoving01(MapAbstract):

    def __init__(self, drone_type: Type[DroneAbstract], zones_config: ZonesConfig = ()):
        super().__init__(drone_type, zones_config)
        self._max_timestep_limit = 2700
        self._max_walltime_limit = 270  # In seconds

        # PARAMETERS MAP
        self._size_area = (800, 500)

        self._return_area = ReturnArea(size=(200, 120))
        self._return_area_pos = ((295, 100), 0)

        self._rescue_center = RescueCenter(size=(200, 80))
        self._rescue_center_pos = ((295, 205), 0)

        self._no_gps_zone = NoGpsZone(size=(400, 500))
        self._no_gps_zone_pos = ((-190, 0), 0)

        # --- MODIFIED SECTION: ADDING MOVING WOUNDED PERSONS ---
        
        # 1. Define the starting positions for 3 people
        self._wounded_persons_pos = [
            (-310, -180),   # Person 1 (Original)
            (100, 0),       # Person 2 (New)
            (-200, 150)     # Person 3 (New)
        ]

        # 2. Define the path for each person. 
        # The index in this list matches the index in _wounded_persons_pos.
        self._wounded_persons_path = [
            # Path for Person 1 (Original path)
            [(-260, -170), (-360, -190)], 
            
            # Path for Person 2 (Moves in an L-shape loop)
            [(100, 100), (200, 100), (200, 0), (100, 0)],
            
            # Path for Person 3 (Patrols back and forth)
            [(-100, 150), (-200, 150)]
        ]
        
        # -------------------------------------------------------

        self._number_wounded_persons = len(self._wounded_persons_pos)
        self._wounded_persons: List[WoundedPerson] = []

        orient = random.uniform(-math.pi, math.pi)
        self._drones_pos = [((295, 118), orient)]
        self._number_drones = len(self._drones_pos)
        self._drones: List[DroneAbstract] = []

        self._playground = ClosedPlayground(size=self._size_area)

        self._playground.add(self._return_area, self._return_area_pos)
        self._playground.add(self._rescue_center, self._rescue_center_pos)

        add_walls(self._playground)
        add_boxes(self._playground)

        self._explored_map.initialize_walls(self._playground)

        # DISABLER ZONES
        if ZoneType.NO_GPS_ZONE in self._zones_config:
            self._playground.add(self._no_gps_zone, self._no_gps_zone_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            
            # Set initial position
            init_pos = (self._wounded_persons_pos[i], 0)
            self._playground.add(wounded_person, init_pos)

            # Assign movement path
            list_path = self._wounded_persons_path[i]
            # Add initial pos as first point in path
            wounded_person.add_pose_to_path(Pose(np.array(init_pos[0])))
            # Add subsequent waypoints
            for pt in list_path:
                wounded_person.add_pose_to_path(Pose(np.array(list(pt))))

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones,
                             max_timestep_limit=self._max_timestep_limit,
                             max_walltime_limit=self._max_walltime_limit)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            self._playground.add(drone, self._drones_pos[i])


def main():
    eval_plan = EvalPlan()

    zones_config: ZonesConfig = ()
    eval_config = EvalConfig(map_name="MapMoving01", zones_config=zones_config, nb_rounds=2)
    eval_plan.add(eval_config=eval_config)

    zones_config: ZonesConfig = (ZoneType.NO_GPS_ZONE,)
    eval_config = EvalConfig(map_name="MapMoving01", zones_config=zones_config, nb_rounds=2)
    eval_plan.add(eval_config=eval_config)

    for one_eval in eval_plan.list_eval_config:
        gc.collect()

        # Retrieve the class object from the global namespace using its name
        map_class = globals().get(one_eval.map_name)
        # Instantiate the map class with the provided zones configuration
        the_map = map_class(drone_type=DroneMotionless, zones_config=one_eval.zones_config)

        gui = GuiSR(the_map=the_map,
                    use_mouse_measure=True,
                    )
        gui.run()


if __name__ == '__main__':
    main()