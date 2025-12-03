"""
Autonomous frontier-based exploration with corrected grid coordinates,
correct indexation (grid[row, col] = grid[y, x]),
A* path planning through unknown space, and proper world→grid offsets.
"""

import pathlib
import sys
from typing import Type, List, Tuple, Optional
from heapq import heappush, heappop

import cv2
import numpy as np
from scipy.ndimage import label

# Insert local project src
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.maps.walls_medium_02 import add_walls, add_boxes
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR
from swarm_rescue.simulation.utils.grid import Grid
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.elements.rescue_center import RescueCenter
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract


# =====================================================================
#  OCCUPANCY GRID 
# =====================================================================

class OccupancyGrid(Grid):
    """Occupancy grid with corrected world↔grid conversion and A* exploration."""
    
    def __init__(self, size_area_world, resolution: float, lidar):        
        super().__init__(size_area_world=size_area_world, resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution
        self.lidar = lidar

        # Grid size (rows = Y, cols = X)
        self.x_max_grid = int(size_area_world[0] / resolution + 0.5)
        self.y_max_grid = int(size_area_world[1] / resolution + 0.5)

        # numpy array MUST be (rows, cols) = (Y, X)
        self.grid = np.zeros((self.y_max_grid, self.x_max_grid))
        self.zoomed_grid = np.zeros((self.y_max_grid, self.x_max_grid))

        # We set world origin (0,0) at CENTER of map
        self.world_offset = (
            size_area_world[0] / 2.0,
            size_area_world[1] / 2.0
        )

        # Thresholds
        self.EXPLORED_THRESHOLD = -2.0
        self.OBSTACLE_THRESHOLD = 2.0

    # ------------------------------
    # Coordinates conversion
    # ------------------------------

    def convert_to_grid_coordinates(self, x_world, y_world):
        x_shift = x_world + self.world_offset[0]
        y_shift = y_world + self.world_offset[1]
        x = int(x_shift / self.resolution)
        y = int(y_shift / self.resolution)
        return x, y

    def convert_to_world_coordinates(self, x_grid, y_grid):
        wx = (x_grid + 0.5) * self.resolution - self.world_offset[0]
        wy = (y_grid + 0.5) * self.resolution - self.world_offset[1]
        return wx, wy

    # ------------------------------
    # Local Bresenham implementation
    # ------------------------------
    def line_bresenham(self, x0, y0, x1, y1):
        """Return all grid cells between two world coords using Bresenham."""
        x0g, y0g = self.convert_to_grid_coordinates(x0, y0)
        x1g, y1g = self.convert_to_grid_coordinates(x1, y1)

        points_x = []
        points_y = []

        dx = abs(x1g - x0g)
        dy = -abs(y1g - y0g)
        sx = 1 if x0g < x1g else -1
        sy = 1 if y0g < y1g else -1
        err = dx + dy

        x, y = x0g, y0g
        while True:
            points_x.append(x)
            points_y.append(y)
            if x == x1g and y == y1g:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

        return points_x, points_y

    # ------------------------------
    # Safe add_value_along_line
    # ------------------------------
    def add_value_along_line(self, x0, y0, x1, y1, val):
        xs, ys = self.line_bresenham(x0, y0, x1, y1)
        for gx, gy in zip(xs, ys):
            if 0 <= gx < self.x_max_grid and 0 <= gy < self.y_max_grid:
                self.grid[gy, gx] += val

    # ------------------------------
    # Safe add_points
    # ------------------------------
    def add_points(self, x, y, val):
        xs = np.array(x, ndmin=1)
        ys = np.array(y, ndmin=1)
        for wx, wy in zip(xs, ys):
            gx, gy = self.convert_to_grid_coordinates(wx, wy)
            if 0 <= gx < self.x_max_grid and 0 <= gy < self.y_max_grid:
                self.grid[gy, gx] += val




    # ------------------------------
    # Update grid with lidar
    # ------------------------------

    def update_grid(self, pose: Pose):
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY = -0.602
        OBST = 2.0
        FREE = -4.0

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        cos_r = np.cos(lidar_angles + pose.orientation)
        sin_r = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # Empty zone update
        lidar_clip = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        lidar_clip = np.minimum(lidar_clip, max_range)

        px = pose.position[0] + lidar_clip * cos_r
        py = pose.position[1] + lidar_clip * sin_r

        for x, y in zip(px, py):
            self.add_value_along_line(
                pose.position[0], pose.position[1],
                x, y,
                EMPTY
            )

        # Obstacles
        hits = lidar_dist < max_range
        px = pose.position[0] + lidar_dist[hits] * cos_r[hits]
        py = pose.position[1] + lidar_dist[hits] * sin_r[hits]
        self.add_points(px, py, OBST)

        # Drone location always free
        self.add_points(pose.position[0], pose.position[1], FREE)

        # clip
        self.grid = np.clip(self.grid, -40, 40)

        # zoom (for display)
        new_size = (
            int(self.size_area_world[1] * 0.5),
            int(self.size_area_world[0] * 0.5)
        )
        self.zoomed_grid = cv2.resize(self.grid, new_size, interpolation=cv2.INTER_NEAREST)

    # ------------------------------
    # Frontier detection
    # ------------------------------

    def detect_frontiers(self):
        explored = self.grid < self.EXPLORED_THRESHOLD
        obstacle = self.grid > self.OBSTACLE_THRESHOLD
        free = explored & (~obstacle)
        unknown = (~explored) & (~obstacle)

        kernel = np.ones((3, 3), np.uint8)
        free_dil = cv2.dilate(free.astype(np.uint8), kernel, 1)

        frontier = (free_dil & unknown.astype(np.uint8)).astype(bool)

        labeled, n = label(frontier)
        out = []

        for i in range(1, n + 1):
            cells = np.argwhere(labeled == i)
            if len(cells) < 3:
                continue

            # centroid in grid coords (col=x, row=y)
            cy = int(np.mean(cells[:, 0]))
            cx = int(np.mean(cells[:, 1]))
            out.append((cx, cy, len(cells)))

        return out

    # ------------------------------
    # Traversability
    # ------------------------------

    def is_traversable(self, x, y):
        if x < 0 or x >= self.x_max_grid:
            return False
        if y < 0 or y >= self.y_max_grid:
            return False

        val = self.grid[y, x]  # grid[row=y, col=x]
        return not (val > self.OBSTACLE_THRESHOLD)

    # ------------------------------
    # A* Pathfinding
    # ------------------------------

    def astar_path(self, start_world, goal_world):
        sx, sy = self.convert_to_grid_coordinates(*start_world)
        gx, gy = self.convert_to_grid_coordinates(*goal_world)

        if not self.is_traversable(sx, sy):
            return None

        if not self.is_traversable(gx, gy):
            return None

        def h(a, b, c, d):
            return np.hypot(a - c, b - d)

        open_set = []
        heappush(open_set, (0, 0, (sx, sy)))

        came = {}
        g = {(sx, sy): 0}
        counter = 1

        neigh = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

        while open_set:
            _, _, (x, y) = heappop(open_set)

            if (x, y) == (gx, gy):
                path = [(x, y)]
                while (x, y) in came:
                    x, y = came[(x, y)]
                    path.append((x, y))
                path.reverse()
                return [self.convert_to_world_coordinates(px, py) for px, py in path]

            for dx, dy in neigh:
                nx, ny = x + dx, y + dy
                if not self.is_traversable(nx, ny):
                    continue

                cost = g[(x, y)] + (1.414 if dx and dy else 1.0)

                if (nx, ny) not in g or cost < g[(nx, ny)]:
                    g[(nx, ny)] = cost
                    f = cost + h(nx, ny, gx, gy)
                    came[(nx, ny)] = (x, y)
                    heappush(open_set, (f, counter, (nx, ny)))
                    counter += 1

        return None


# =====================================================================
#  DRONE (frontier-based autonomous exploration)
# =====================================================================

class MyDroneMapping(DroneAbstract):

    def __init__(self, **kw):
        super().__init__(**kw)

        self.iteration = 0
        self.estimated_pose = Pose()

        resolution = 8
        self.grid = OccupancyGrid(self.size_area, resolution, self.lidar())

        self.target = None
        self.path = []
        self.wi = 0  # waypoint index

        self.waypoint_radius = 25.0
        self.max_speed = 0.5
        self.rot_speed = 0.3

        self.SIZE_W = 0.7
        self.DIST_W = 0.3

        self.expl_started = False
        self.min_iter = 20

    def define_message_for_all(self):
        pass

    # ------------------------------
    # Frontier selection
    # ------------------------------

    def select_best_frontier(self, fronts):
        if not fronts:
            return None

        cx = self.estimated_pose.position[0]
        cy = self.estimated_pose.position[1]

        best = None
        best_score = -1e9

        for xg, yg, s in fronts:
            wx, wy = self.grid.convert_to_world_coordinates(xg, yg)
            d = np.hypot(wx - cx, wy - cy)
            if d < 1:
                d = 1
            score = self.SIZE_W * s - self.DIST_W * d
            if score > best_score:
                best_score = score
                best = (wx, wy)

        return best

    # ------------------------------
    # Navigate through path
    # ------------------------------

    def follow_path(self) -> CommandsDict:

        cmd = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0
        }

        if not self.path or self.wi >= len(self.path):
            return cmd

        wx, wy = self.path[self.wi]
        px, py = self.estimated_pose.position

        dx = wx - px
        dy = wy - py
        dist = np.hypot(dx, dy)

        if dist < self.waypoint_radius:
            self.wi += 1
            return cmd

        desired = np.arctan2(dy, dx)
        diff = desired - self.estimated_pose.orientation
        diff = np.arctan2(np.sin(diff), np.cos(diff))

        if abs(diff) > 0.1:
            cmd["rotation"] = np.clip(diff * 2.0, -self.rot_speed, self.rot_speed)

        if abs(diff) < np.pi / 4:
            cmd["forward"] = min(self.max_speed, dist / 100.0)

        return cmd

    # ------------------------------
    # Main drone control
    # ------------------------------

    def control(self) -> CommandsDict:

        self.iteration += 1

        self.estimated_pose = Pose(
            np.asarray(self.measured_gps_position()),
            self.measured_compass_angle()
        )

        self.grid.update_grid(self.estimated_pose)

        if not self.expl_started:
            if self.iteration < self.min_iter:
                return {"forward": 0, "lateral": 0, "rotation": 0, "grasper": 0}
            self.expl_started = True

        # Frontier discovery every 50 frames or when no target
        if self.target is None or self.iteration % 50 == 0:

            fronts = self.grid.detect_frontiers()

            if not fronts:
                print("Exploration complete.")
                return {"forward": 0, "lateral": 0, "rotation": 0, "grasper": 0}

            new_t = self.select_best_frontier(fronts)

            if new_t and new_t != self.target:
                self.target = new_t
                path = self.grid.astar_path(
                    (self.estimated_pose.position[0], self.estimated_pose.position[1]),
                    self.target
                )

                if path:
                    self.path = path
                    self.wi = 0
                else:
                    self.target = None

        return self.follow_path()


# =====================================================================
#  MAP + GUI
# =====================================================================

class MyMapMapping(MapAbstract):

    def __init__(self, drone_type: Type[DroneAbstract]):
        super().__init__(drone_type=drone_type)

        self._size_area = (1113, 750)

        self._rescue_center = RescueCenter(size=(210, 90))
        self._rescue_center_pos = ((440, 315), 0)

        self._number_drones = 1
        self._drones_pos = [((-50, 0), 0)]
        self._drones = []

        self._playground = ClosedPlayground(self._size_area)

        self._playground.add(self._rescue_center, self._rescue_center_pos)
        add_walls(self._playground)
        add_boxes(self._playground)

        misc = MiscData(
            size_area=self._size_area,
            number_drones=self._number_drones,
            max_timestep_limit=self._max_timestep_limit,
            max_walltime_limit=self._max_walltime_limit
        )

        for i in range(self._number_drones):
            d = drone_type(identifier=i, misc_data=misc)
            self._drones.append(d)
            self._playground.add(d, self._drones_pos[i])


def main():
    the_map = MyMapMapping(drone_type=MyDroneMapping)
    gui = GuiSR(the_map=the_map, use_keyboard=False)
    gui.run()


if __name__ == '__main__':
    main()
