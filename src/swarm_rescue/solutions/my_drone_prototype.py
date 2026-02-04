import math
import numpy as np
from enum import Enum
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy import ndimage
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
import arcade
import heapq
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from examples.example_mapping import OccupancyGrid
from swarm_rescue.simulation.utils.pose import Pose


class MyDronePrototype(DroneAbstract):
    class Activity(Enum):
        """
        Enumeration of all drone states in the state machine.
        """
        EXPLORING = 1
        GOING_TO_WOUNDED = 2
        GOING_TO_RESCUE_CENTER = 3
        GOING_TO_RETURN_AREA = 4


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.current_pose = np.array([0.0, 0.0, 0.0])

        self.iteration: int = 0
        resolution = 10
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
        
        self.robot_radius_pixels = 30
        self.inflation_radius_cells = int(self.robot_radius_pixels / self.grid.resolution)
        if self.inflation_radius_cells < 1:
            self.inflation_radius_cells = 1
        
        self.prev_angle_error = 0.0
        self.Kp = 5.0
        self.Kd = 3.0

        self.Kp_pos = 6.0
        self.Kd_pos = 11.0
        self.prev_speed_error = 0.0
        self.prev_lat_error = 0.0

        self.path = []

        self.path_smoothing_enabled = True
        self.path_lookahead_distance = 35.0

        self.wounded_to_rescue = []
        self.rescue_zone_points = []
        self.return_area_points = []

        self.state = self.Activity.EXPLORING
        self.current_target_wounded = None
        self.last_replan_iteration = 0
        
        self.kf_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.kf_P = np.eye(4) * 100.0
        self.kf_Q = np.eye(4) * 8.0
        self.kf_Q[2, 2] = 1.0
        self.kf_Q[3, 3] = 1.0
        self.kf_R = np.eye(2) * 10.0
        self.kf_dt = 0.1
        self.kf_last_time = 0
        self.kf_initialized = False

        self.evaluated_wounded = set()
        
        self.wounded_assignments = {}

        self.removed_wounded = []
        self.removed_wounded_set = set() 
       
        
        self.general_stuck_counter = 0
        self.last_unstuck_check_pos = None
        self.unstuck_target = None
        self.is_unstucking = False

        self._last_drone_positions = None
        self._last_drone_danger_zone = None
        self._last_danger_zone_iter = -100

        self.frontier_clusters = [] 

        self.path_cache = {}
        self.path_cache_max_age = 50
        self.path_cache_max_size = 15

        self.breadcrumbs = []
        self.last_breadcrumb_pos = None
        self.breadcrumb_spacing = 100.0

        self.grasped_wounded_angle = None 

        self.dead_drones = []
        self.suspected_dead_drones = []
        self.dead_drone_radius = 120.0
        self.dead_confirm_iterations = 5
        
        self.fallback_attempts = 0
        self.max_fallback_attempts = 3
        self.fallback_attempt_iteration = -1


    def update_grid_with_gps_check(self, has_gps):
        """
        Update the occupancy grid.
        If no GPS, only allow modifications to unexplored cells to preserve already-mapped areas.
        """
        self.grid.update_grid(pose=self.estimated_pose)
        
        if not has_gps:
            THRESHOLD_UNEXPLORED_MIN = -4.99
            THRESHOLD_UNEXPLORED_MAX = 2.99
            
            is_unexplored = (self.grid.grid >= THRESHOLD_UNEXPLORED_MIN) & (self.grid.grid <= THRESHOLD_UNEXPLORED_MAX)
            
            is_wall = (self.grid.grid >= 3.0)
            is_explored_free = (self.grid.grid < -5.0)
            
            is_explored = is_wall | is_explored_free
            
            self.grid.grid[is_explored] = self.grid.grid[is_explored]

    def create_path(self, start_world, goal_world, explored_only=False):
        """
        Compute a path with smoothing for linear movements.
        """
        start_key = (round(start_world[0] / 10) * 10, round(start_world[1] / 10) * 10)
        goal_key = (round(goal_world[0] / 10) * 10, round(goal_world[1] / 10) * 10)
        cache_key = (start_key, goal_key, explored_only)
        
        if cache_key in self.path_cache:
            cached_path, cached_iteration = self.path_cache[cache_key]
            if self.iteration - cached_iteration < self.path_cache_max_age:
                return [np.array(pt) for pt in cached_path]

        grid = self.grid.grid.copy()

        start = self.grid._conv_world_to_grid(*start_world)
        goal = self.grid._conv_world_to_grid(*goal_world)
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))

        THRESHOLD_WALL = 3.0
        THRESHOLD_FREE = -5.0
        THRESHOLD_UNEXPLORED_MAX = 2.99
        THRESHOLD_UNEXPLORED_MIN = -4.99
    
        is_wall = (grid >= THRESHOLD_WALL)
        is_explored_free = (grid < THRESHOLD_FREE)
        is_unexplored = (grid >= THRESHOLD_UNEXPLORED_MIN) & (grid <= THRESHOLD_UNEXPLORED_MAX)
        
        struct = np.ones((6, 6), dtype=bool)
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)

        dist_map = ndimage.distance_transform_edt(~is_wall)

        COMFORT_DISTANCE_WORLD = 200.0
        comfort_dist_cells = COMFORT_DISTANCE_WORLD / self.grid.resolution
        MAX_PENALTY = 100.0

        if explored_only:
            danger_zone = danger_zone | (~is_explored_free)
        
        try:
            radius_clear = 2
            sx, sy = start
            gx, gy = goal
            x0 = max(0, sx - radius_clear)
            x1 = min(grid.shape[0], sx + radius_clear + 1)
            y0 = max(0, sy - radius_clear)
            y1 = min(grid.shape[1], sy + radius_clear + 1)
            danger_zone[x0:x1, y0:y1] = False
            radius_clear_goal = 5
            y0 = max(0, gy - radius_clear_goal)
            y1 = min(grid.shape[1], gy + radius_clear_goal + 1)
            x0 = max(0, gx - radius_clear_goal)
            x1 = min(grid.shape[0], gx + radius_clear_goal + 1)
            danger_zone[x0:x1, y0:y1] = False
        except Exception:
            pass

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = [(fscore[start], start)]

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                STEP = 7 
                if len(path) > STEP:
                    compressed = [path[0]]
                    for i in range(1, len(path) - 1):
                        prev_v = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                        next_v = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                        if prev_v != next_v:
                            compressed.append(path[i])
                    compressed.append(path[-1])
                    path = compressed
            
                if len(path) > 2 and self.path_smoothing_enabled:
                    smoothed = self.smooth_path(path, danger_zone)
                else:
                    smoothed = path
                
                world_path = [np.array(self.grid._conv_grid_to_world(*pt)) for pt in smoothed]

                self.path_cache[cache_key] = (world_path, self.iteration)
                if len(self.path_cache) > self.path_cache_max_size:
                    oldest_key = min(self.path_cache.keys(), key=lambda k: self.path_cache[k][1])
                    self.path_cache.pop(oldest_key, None)
                return world_path
        

            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    continue

                if danger_zone[neighbor]:
                    continue

                base_cost = math.hypot(dx, dy)
                dist_to_wall_cells = dist_map[neighbor[0], neighbor[1]]
                
                penalty = 0.0
                if dist_to_wall_cells < comfort_dist_cells:
                    proximity = 1.0 - (dist_to_wall_cells / comfort_dist_cells)
                    penalty = MAX_PENALTY * (proximity ** 2)
                
                move_cost = base_cost + penalty

                tentative_g_score = gscore[current] + move_cost

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                            
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
                    
        return []


    def smooth_path(self, path_grid, danger_zone):
        """
        Apply Chaikin's corner-cutting algorithm to smooth the path.
        """
        if len(path_grid) <= 2:
            return path_grid
        
        smoothed = [path_grid[0]]
        
        for i in range(len(path_grid) - 1):
            p0 = np.array(path_grid[i], dtype=float)
            p1 = np.array(path_grid[i + 1], dtype=float)
            
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            
            q_safe = self.is_point_safe(q, danger_zone)
            r_safe = self.is_point_safe(r, danger_zone)
            
            if q_safe and r_safe:
                smoothed.append(tuple(q.astype(int)))
                smoothed.append(tuple(r.astype(int)))
            else:
                smoothed.append(path_grid[i + 1])
        
        deduplicated = [smoothed[0]]
        for pt in smoothed[1:]:
            if pt != deduplicated[-1]:
                deduplicated.append(pt)
        
        return deduplicated
    

    def is_point_safe(self, point, danger_zone):
        """
        Check if a point is in a safe region.
        """
        x, y = int(round(point[0])), int(round(point[1]))
        
        if not (0 <= y < danger_zone.shape[1] and 0 <= x < danger_zone.shape[0]):
            return False
        
        return not danger_zone[x, y]
    

    def define_message_for_all(self):
        """
        Prepare optimized communication message with essential data.
        """
    
        grasped_positions = set(
            (w.position[0], w.position[1]) for w in getattr(self.grasper, "grasped_wounded_persons", []) if hasattr(w, "position")
        )
    
        wounded_list = [
            w for w in self.wounded_to_rescue
            if w not in grasped_positions
        ]
    
        message = {
            "drone_id": self.identifier,
            "status": "alive",
            "drone_pose": self.current_pose.tolist(),
            "wounded_assignments": self.wounded_assignments,
            "grasped_wounded": list(grasped_positions),
        }
    
        if not hasattr(self, '_last_wounded_list') or self._last_wounded_list != wounded_list or self.iteration % 5 == 0:
            message["wounded_list"] = wounded_list
            self._last_wounded_list = wounded_list
    
        if not hasattr(self, '_last_rescue_list') or self._last_rescue_list != self.rescue_zone_points or self.iteration % 10 == 0:
            message["rescue_list"] = self.rescue_zone_points
            self._last_rescue_list = self.rescue_zone_points
    
        if self.iteration % 5 == 0:
            message["grid_data"] = self.grid.grid.copy()
    
        if self.removed_wounded:
            message["removed_wounded"] = self.removed_wounded
    
        if self.iteration % 10 == 0 and self.frontier_clusters:
            message["frontier_clusters"] = [
                {"barycenter": cluster["barycenter"].tolist()}
                for cluster in self.frontier_clusters
            ]
    
        if self.state == self.Activity.EXPLORING and hasattr(self, "target_point"):
            message["assigned_barycenters"] = {
                str(self.identifier): np.array(self.target_point).tolist()
            }
    
        return message

    def control(self) -> CommandsDict:
        """
        Main brain: simplified test logic for drone behavior.
        """

        self.iteration += 1

        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        self.process_communication_sensor()

        self.update_pose()

        self.update_breadcrumbs()

        lidar_data = self.lidar_values()
        if lidar_data is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        

        if getattr(self, 'is_inside_return_area', False):
            if len(self.return_area_points) < 3:
                self.add_return_area_point(self.current_pose[:2])
            
        self.estimated_pose = Pose(np.asarray([self.current_pose[0], self.current_pose[1]]),
                                self.current_pose[2])
        
        has_gps = self.measured_gps_position() is not None
        self.update_grid_with_gps_check(has_gps)

        try:
            self.detect_semantic_entities()
        except Exception:
            pass

        if self.iteration % 20 == 0:
            self.find_safe_frontier_points()

        if self.check_and_handle_general_stuck():
            if self.path:
                command = self.follow_path(lidar_data)
                return command
            
            else:
                print(f"[{self.identifier}] No barycenters available - using simple reverse")
                return {"forward": -0.5, "lateral": 0.3, "rotation": 0.4, "grasper": 1}
               

        if self.state == self.Activity.EXPLORING:

            if self.grasper.grasped_wounded_persons:
                self.state = self.Activity.GOING_TO_RESCUE_CENTER

                if self.rescue_zone_points:
                    target_index = int(self.identifier) % len(self.rescue_zone_points)
                    target_zone = self.rescue_zone_points[target_index]
                    self.path = self.create_path(self.current_pose[:2], target_zone, explored_only=True)
                    self.last_replan_iteration = self.iteration
            
            grasped = getattr(self, "other_grasped_wounded", set())
            exclusion_radius = 50.0

            def is_near_grasped(w):
                return any(math.hypot(w[0] - gx, w[1] - gy) < exclusion_radius for (gx, gy) in grasped)

            available_wounded = [
                w for w in self.wounded_to_rescue
                if w not in self.wounded_assignments and not is_near_grasped(w)
            ]
            
            if available_wounded and (self.iteration % 30 == 0):
                print(f"[{self.identifier}] [DEBUG] Available wounded: {available_wounded}")
                distances = [np.linalg.norm(np.array(w) - self.current_pose[:2]) for w in available_wounded]
                closest_idx = int(np.argmin(distances))
                closest_wounded = available_wounded[closest_idx]
                my_distance = distances[closest_idx]
                
                wounded_key = (round(closest_wounded[0], 1), round(closest_wounded[1], 1))
                
                if wounded_key not in self.evaluated_wounded:
                    self.evaluated_wounded.add(wounded_key)
                    
                    should_assign = True
                    for msg in getattr(self.communicator, "received_messages", []):
                        other = msg[1] if isinstance(msg, tuple) else msg
                        other_id = other.get("drone_id")
                        other_pose = np.array(other.get("drone_pose", [None, None, None]))
                        
                        if other_id != self.identifier and other_pose[0] is not None:
                            other_dist = np.linalg.norm(np.array(closest_wounded) - other_pose[:2])
                            
                            if other_dist < my_distance - 10.0:
                                should_assign = False
                                break
                            
                            if abs(other_dist - my_distance) < 30.0 and other_id < self.identifier:
                                should_assign = False
                                break
                    
               
                    if should_assign:
                        self.current_target_wounded = closest_wounded
                        self.wounded_assignments[self.current_target_wounded] = self.identifier
                        self.state = self.Activity.GOING_TO_WOUNDED
                        self.path = self.create_path(self.current_pose[:2], self.current_target_wounded)
                        self.last_replan_iteration = self.iteration
                        

        elif self.state == self.Activity.GOING_TO_WOUNDED:
            grasped = getattr(self, "other_grasped_wounded", set())
            exclusion_radius = 20.0

            def is_near_grasped(w):
                return any(math.hypot(w[0] - gx, w[1] - gy) < exclusion_radius for (gx, gy) in grasped)

            for (gx, gy) in grasped:
                if math.hypot(self.current_pose[0] - gx, self.current_pose[1] - gy) < exclusion_radius:
                    self.state = self.Activity.EXPLORING
                    self.current_target_wounded = None
                    self.path = []

            if self.current_target_wounded is not None and (self.iteration % 20 == 0 or not self.path):
                self.path = self.create_path(self.current_pose[:2], self.current_target_wounded)
                self.last_replan_iteration = self.iteration

                    

            if self.current_target_wounded is not None:
                if self.iteration % 20 == 0:
                    my_dist = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])
                    should_abandon = False
                    
                    for msg in getattr(self.communicator, "received_messages", []):
                        other = msg[1] if isinstance(msg, tuple) else msg
                        other_id = other.get("drone_id")
                        other_pose = np.array(other.get("drone_pose", [None, None, None]))
                        
                        if other_id != self.identifier and other_pose[0] is not None:
                            other_dist = np.linalg.norm(np.array(self.current_target_wounded) - other_pose[:2])
                            
                            if other_dist < my_dist - 10.0:
                                should_abandon = True
                                winner_id = other_id
                                break
                    
                    if should_abandon:
                        print(f"[{self.identifier}] Abandoning target - drone {winner_id} is closer")
                        self.state = self.Activity.EXPLORING
                        self.wounded_assignments.pop(self.current_target_wounded, None)
                        self.current_target_wounded = None
                        self.path = []
                        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1}
                    
            if self.grasper.grasped_wounded_persons:
        
                self.state = self.Activity.GOING_TO_RESCUE_CENTER
                if self.current_target_wounded is not None:
                    
                    self.removed_wounded.append(self.current_target_wounded)

                    self.wounded_to_rescue = [
                        w for w in self.wounded_to_rescue 
                        if math.hypot(w[0] - self.current_target_wounded[0], 
                                    w[1] - self.current_target_wounded[1]) > 50.0
                    ]

                if self.rescue_zone_points:
                    target_index = int(self.identifier) % len(self.rescue_zone_points)
                    target_zone = self.rescue_zone_points[target_index]
    
                    self.path = self.create_path(self.current_pose[:2], target_zone, explored_only=True)
                    self.last_replan_iteration = self.iteration


            elif self.current_target_wounded is not None:
                distance_to_target = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])
                

                wounded_detected = False
                detection_radius = 50.0
                
                if distance_to_target < 30.0:
                    try:
                        detections = self.semantic_values()
                        if detections:
                            px = float(self.current_pose[0])
                            py = float(self.current_pose[1])
                            ptheta = float(self.current_pose[2])
                            
                            for data in detections:
                                try:
                                    etype = getattr(data, 'entity_type', None)
                                    name = etype.name if hasattr(etype, 'name') else str(etype)
                                    
                                    if 'WOUNDED' in name.upper():
                                        angle = float(getattr(data, 'angle', 0.0))
                                        dist = float(getattr(data, 'distance', 0.0))
                                        
                                        global_angle = normalize_angle(ptheta + angle)
                                        xw = px + dist * math.cos(global_angle)
                                        yw = py + dist * math.sin(global_angle)
                                        
                                        dist_to_target = math.hypot(self.current_target_wounded[0] - xw, 
                                                                   self.current_target_wounded[1] - yw)
                                        
                                        if dist_to_target < detection_radius:
                                            wounded_detected = True
                                            break
                                except Exception as e:
                                    print(f"  Error processing detection: {e}")
                                    continue
                        else:
                            print("No semantic detections available")
                    except Exception as e:
                        print(f"Error reading semantic sensor: {e}")
                    
                    
                  
                  
                    if not wounded_detected:
                        print(f"\n*** WOUNDED NOT FOUND - REMOVING FROM LIST ***")
                        check_radius = 50.0
                        
                        count_before = len(self.wounded_to_rescue)
                        print(f"Wounded list before removal: {self.wounded_to_rescue}")
                        
                        self.wounded_to_rescue = [
                            (wx, wy) for (wx, wy) in self.wounded_to_rescue
                            if math.hypot(self.current_target_wounded[0] - wx, 
                                        self.current_target_wounded[1] - wy) > check_radius
                        ]
                        
                        count_after = len(self.wounded_to_rescue)
                        print(f"Removed {count_before - count_after} wounded from list")
                        print(f"Wounded list after removal: {self.wounded_to_rescue}")
                        self.removed_wounded.append(self.current_target_wounded)
                        

                        
                        self.state = self.Activity.EXPLORING
                        self.current_target_wounded = None
                        self.path = []
                        print(f"Switched to EXPLORING state\n")


            else:
                self.state = self.Activity.EXPLORING
                self.current_target_wounded = None

        elif self.state == self.Activity.GOING_TO_RESCUE_CENTER:

            if not self.grasper.grasped_wounded_persons:
                self.grasped_wounded_angle = None
                if self.current_target_wounded is not None:
                    self.wounded_assignments.pop(self.current_target_wounded, None)
               
                self.state = self.Activity.EXPLORING
                self.current_target_wounded = None

                self.breadcrumbs = []
                self.path = []
                
                
            else:
                if self.rescue_zone_points:
                   
                    should_replan = False
                    
                    if self.iteration % 20 == 0:
                        should_replan = True
                    elif (not self.path or len(self.path) == 0):
                        iterations_since_replan = self.iteration - self.last_replan_iteration
                        if iterations_since_replan >= 30 or self.last_replan_iteration == 0:
                            should_replan = True
                    
                    if should_replan:
                        print(f"[{self.identifier}] Planning path to rescue center at {self.rescue_zone_points[0]}")
                        self.path = self.create_path(
                            self.current_pose[:2], 
                            self.rescue_zone_points[0], 
                            explored_only=False
                        )
                        self.last_replan_iteration = self.iteration
                        
                        if self.path:
                            print(f"[{self.identifier}] SUCCESS: Found path to rescue center (length: {len(self.path)})")
                        else:
                            print(f"[{self.identifier}] FAILED: No path to rescue center found")
                    
                    if not self.path:
                        print(f"[{self.identifier}] No path to rescue center, attempting breadcrumb fallback...")
                        if len(self.breadcrumbs) > 4:
                            try:
                                current_pos = self.current_pose[:2]
                                breadcrumbs_np = np.array(self.breadcrumbs)
                                dists = np.linalg.norm(breadcrumbs_np - current_pos, axis=1)
                                nearest_idx = int(np.argmin(dists))
                                
                                intermediate_waypoints = []
                                for i in range(nearest_idx, max(0, nearest_idx - 10), -2):
                                    intermediate_waypoints.append(self.breadcrumbs[i])
                                
                                self.path = []
                                current_start = self.current_pose[:2]
                                
                                for waypoint in intermediate_waypoints:
                                    segment_path = self.create_path(current_start, waypoint, explored_only=False)
                                    if segment_path:
                                        self.path.extend(segment_path)
                                        current_start = waypoint
                                    else:
                                        print(f"[{self.identifier}] Failed to path to breadcrumb waypoint")
                                        break
                                
                                final_path = self.create_path(current_start, self.rescue_zone_points[0], explored_only=False)
                                if final_path:
                                    self.path.extend(final_path)
                                    print(f"[{self.identifier}] SUCCESS: Breadcrumb path with pathfinding (length: {len(self.path)})")
                                else:
                                    print(f"[{self.identifier}] Failed to path from last breadcrumb to rescue center")
                                    self.path = []
                            except Exception as e:
                                print(f"[{self.identifier}] Breadcrumb fallback failed: {e}")
                        else:
                            print(f"[{self.identifier}] Not enough breadcrumbs, retrying full pathfind...")
                            self.path = self.create_path(
                                self.current_pose[:2], 
                                self.rescue_zone_points[0], 
                                explored_only=False
                            )

        if self.state == self.Activity.EXPLORING:
            need_replan = False
            
            if not self.path or len(self.path) < 1:
                need_replan = True
            elif hasattr(self, 'target_point') and self.target_point is not None:
                dist_to_target = np.linalg.norm(self.target_point - self.current_pose[:2])
                if dist_to_target > 200.0:
                    if self.iteration % 50 == 0:
                        need_replan = True
                else:
                    need_replan = False

            if need_replan:
                local_frontiers = self.find_safe_frontier_points()
                shared_clusters = getattr(self, "shared_frontier_barycenters", [])
                
                all_candidates = []
                
                for bc in shared_clusters:
                    all_candidates.append({"point": np.array(bc), "source": "shared"})
                
                for lf in local_frontiers:
                    if not any(np.linalg.norm(lf - c["point"]) < 40.0 for c in all_candidates):
                        all_candidates.append({"point": lf, "source": "local"})

                if all_candidates:
                    scored_targets = []
                    
                    assigned_targets = {}
                    for msg in getattr(self.communicator, "received_messages", []):
                        other = msg[1] if isinstance(msg, tuple) else msg
                        other_assignments = other.get("assigned_barycenters", {})
                        for drone_id_str, target in other_assignments.items():
                            if int(drone_id_str) != self.identifier:
                                assigned_targets[int(drone_id_str)] = np.array(target)

                    for cand in all_candidates:
                        p = cand["point"]
                        distance = np.linalg.norm(p - self.current_pose[:2])
                        
                        conflict_penalty = 0.0
                        for other_target in assigned_targets.values():
                            if np.linalg.norm(p - other_target) < 300.0:
                                conflict_penalty += 10000.0

                        size_bonus = 0.0
                        for cluster in self.frontier_clusters:
                            if np.linalg.norm(cluster["barycenter"] - p) < 20:
                                size_bonus = -cluster["size"] * 50.0
                                break
                        
                        score = distance + conflict_penalty + size_bonus
                        scored_targets.append({"point": p, "score": score})

                    scored_targets.sort(key=lambda x: x["score"])

                    found_path = False
                    for target_info in scored_targets:
                        path = self.create_path(self.current_pose[:2], target_info["point"])
                        if path:
                            self.target_point = target_info["point"]
                            self.path = path
                            found_path = True
                            break
                    
                    if not found_path:
                        print(f"[{self.identifier}] No path found to any frontier candidate - trying unexplored areas fallback")
                        if self.fallback_attempt_iteration != self.iteration:
                            self.fallback_attempts = 0
                            self.fallback_attempt_iteration = self.iteration
                        
                        if self.fallback_attempts < self.max_fallback_attempts:
                            self.fallback_attempts += 1
                            if self.local_exploration_fallback():
                                print(f"[{self.identifier}] Fallback succeeded")
                            else:
                                print(f"[{self.identifier}] Fallback failed (attempt {self.fallback_attempts}/{self.max_fallback_attempts})")
                                if self.fallback_attempts >= self.max_fallback_attempts:
                                    print(f"[{self.identifier}] Max fallback attempts reached - going to return area")
                                    self.go_to_return_area(lidar_data)
                        else:
                            print(f"[{self.identifier}] Max fallback attempts reached - going to return area")
                            self.go_to_return_area(lidar_data)
                else:
                    print(f"[{self.identifier}] No frontier candidates found - trying unexplored areas fallback")
                    if self.fallback_attempt_iteration != self.iteration:
                        self.fallback_attempts = 0
                        self.fallback_attempt_iteration = self.iteration
                    
                    if self.fallback_attempts < self.max_fallback_attempts:
                        self.fallback_attempts += 1
                        if self.local_exploration_fallback():
                            print(f"[{self.identifier}] Fallback succeeded")
                        else:
                            print(f"[{self.identifier}] Fallback failed (attempt {self.fallback_attempts}/{self.max_fallback_attempts})")
                            if self.fallback_attempts >= self.max_fallback_attempts:
                                print(f"[{self.identifier}] Max fallback attempts reached - going to return area")
                                self.go_to_return_area(lidar_data)
                    else:
                        print(f"[{self.identifier}] Max fallback attempts reached - going to return area")
                        self.go_to_return_area(lidar_data)


        if self.state == self.Activity.EXPLORING:
            if self.path:   
                command = self.follow_path(lidar_data)
            else:
                self.go_to_return_area(lidar_data)
                

        elif self.state == self.Activity.GOING_TO_WOUNDED:

            command = self.go_to_wounded(lidar_data)
            

        elif self.state == self.Activity.GOING_TO_RESCUE_CENTER:

            if self.rescue_zone_points:
                dist_to_rescue = np.linalg.norm(
                    np.array(self.rescue_zone_points[0]) - self.current_pose[:2]
                )
            else:
                dist_to_rescue = 999

            if self.path and dist_to_rescue < 100.0:
                command = self.go_to_rescue_center_oriented(lidar_data)
            elif self.path: 
                command = self.follow_path(lidar_data)
            else:
                command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        elif self.state == self.Activity.GOING_TO_RETURN_AREA:
            local_frontiers = self.find_safe_frontier_points()
            if local_frontiers and len(local_frontiers) > 0:
                print(f"[{self.identifier}] Found new exploration points while returning - switching back to EXPLORING")
                self.state = self.Activity.EXPLORING
                self.path = []
            elif self.path:
                command = self.follow_path(lidar_data)
            else:
                self.go_to_return_area(lidar_data)


        if self.state == self.Activity.GOING_TO_WOUNDED or self.state == self.Activity.GOING_TO_RESCUE_CENTER:
            command["grasper"] = 1
        else:
            self.grasper._release_grasping()


        if self.path and hasattr(self, 'other_drones_positions') and self.other_drones_positions:
            replan_needed = False
            for drone_pos in self.other_drones_positions:
                drone_pos = drone_pos[0]
                for waypoint in self.path[:min(3, len(self.path))]:
                    dist_to_waypoint = math.hypot(waypoint[0] - drone_pos[0], waypoint[1] - drone_pos[1])
                    if dist_to_waypoint < 60.0:
                        replan_needed = True
                        break
                if replan_needed:
                    break
            
            if replan_needed and (self.iteration - self.last_replan_iteration) > 10:
                if self.state == self.Activity.GOING_TO_WOUNDED and self.current_target_wounded:
                    self.path = self.create_path(self.current_pose[:2], self.current_target_wounded)
                    self.last_replan_iteration = self.iteration
                elif self.state == self.Activity.GOING_TO_RESCUE_CENTER and self.rescue_zone_points:
                    self.path = self.create_path(self.current_pose[:2], self.rescue_zone_points[0], explored_only=True)
                    self.last_replan_iteration = self.iteration


        command = self.drone_repulsion(command)

        if self.iteration % 5 == 0:
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")

        return command

    def detect_semantic_entities(self):
        """
        Detect wounded persons and rescue centers using the semantic sensor.
        """
        try:
            detections = self.semantic_values()
        except Exception:
            detections = None

        if not detections:
            return

        dedup_radius = 60.0
        alpha_update = 0.3

        px, py, ptheta = float(self.current_pose[0]), float(self.current_pose[1]), float(self.current_pose[2])

        if not hasattr(self, 'other_grasped_wounded'):
            self.other_grasped_wounded = set()

        newly_seen_wounded = []
        newly_seen_rescue = []

        for data in detections:
            if not hasattr(data, "entity_type"):
                continue

            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                global_angle = normalize_angle(ptheta + data.angle)
                xw = px + data.distance * math.cos(global_angle)
                yw = py + data.distance * math.sin(global_angle)
                is_grasped = any(
                    math.hypot(xw - gx, yw - gy) < dedup_radius
                    for (gx, gy) in self.other_grasped_wounded
                )
                if not is_grasped:
                    newly_seen_wounded.append((xw, yw))

            elif data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                global_angle = normalize_angle(ptheta + data.angle)
                xr = px + data.distance * math.cos(global_angle)
                yr = py + data.distance * math.sin(global_angle)
                newly_seen_rescue.append((xr, yr))

        for nx, ny in newly_seen_wounded:
            merged = False
            for i, (wx, wy) in enumerate(self.wounded_to_rescue):
                if math.hypot(wx - nx, wy - ny) < dedup_radius:
                    newx = (1.0 - alpha_update) * wx + alpha_update * nx
                    newy = (1.0 - alpha_update) * wy + alpha_update * ny
                    self.wounded_to_rescue[i] = (newx, newy)
                    merged = True
                    break
            if not merged:
                self.wounded_to_rescue.append((nx, ny))

        for nx, ny in newly_seen_rescue:
            self._add_or_merge_rescue_point((nx, ny))
            
        self.detect_dead_drones(detections, px, py, ptheta)

    def detect_dead_drones(self, detections, px, py, ptheta):
        """
        Identify drones that are visible but not communicating and not moving.
        """
        alive_drones_positions = []
        for msg in getattr(self.communicator, "received_messages", []):
            if isinstance(msg, dict):
                 data = msg
            elif isinstance(msg, tuple):
                 data = msg[1]
            else:
                 continue
            
            p = data.get("drone_pose")
            if p is not None:
                alive_drones_positions.append(np.array(p[:2]))

        visible_drones = []
        for data in detections:
             if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                global_angle = normalize_angle(ptheta + data.angle)
                xd = px + data.distance * math.cos(global_angle)
                yd = py + data.distance * math.sin(global_angle)
                visible_drones.append(np.array([xd, yd]))
        
        for v_drone_pos in visible_drones:
            min_dist_alive = float('inf')
            if alive_drones_positions:
                dists = [np.linalg.norm(v_drone_pos - ap) for ap in alive_drones_positions]
                min_dist_alive = min(dists)

            if min_dist_alive < self.dead_drone_radius:
                continue

            is_known_dead = False
            for dead_pos in self.dead_drones:
                if np.linalg.norm(v_drone_pos - np.array(dead_pos)) < self.dead_drone_radius:
                    is_known_dead = True
                    break
            
            if is_known_dead:
                continue
            
            matched_suspect = None
            for suspect in self.suspected_dead_drones:
                if np.linalg.norm(v_drone_pos - np.array(suspect['pos'])) < self.dead_drone_radius:
                    matched_suspect = suspect
                    break
            
            if matched_suspect:
                dist_from_start = np.linalg.norm(v_drone_pos - np.array(matched_suspect['start_pos']))
                if dist_from_start > 30.0:
                    print(f"[{self.identifier}] Suspect moved {dist_from_start:.1f}px - REMOVING")
                    self.suspected_dead_drones.remove(matched_suspect)
                else:
                    matched_suspect['last_seen_iter'] = self.iteration
                    matched_suspect['pos'] = (v_drone_pos[0], v_drone_pos[1])
                    
                    if (self.iteration - matched_suspect['first_seen_iter']) > self.dead_confirm_iterations:
                        print(f"[{self.identifier}] *** CONFIRMED DEAD DRONE at ({matched_suspect['pos'][0]:.1f}, {matched_suspect['pos'][1]:.1f}) ***")
                        self.dead_drones.append(matched_suspect['pos'])
                        self.suspected_dead_drones.remove(matched_suspect)

                        print(f"[{self.identifier}] -> Clearing path to force replanning around kill zone.")
                        self.path = []
            
            else:
                print(f"[{self.identifier}] VISIBLE DRONE WITHOUT RADIO SIGNAL at ({v_drone_pos[0]:.1f}, {v_drone_pos[1]:.1f})")
                print(f"[{self.identifier}] (Nearest radio signal: {min_dist_alive:.1f})")
                print(f"[{self.identifier}] ??? SUSPECTED DEAD DRONE initialized ???")
                self.suspected_dead_drones.append({
                    'pos': (v_drone_pos[0], v_drone_pos[1]),
                    'start_pos': (v_drone_pos[0], v_drone_pos[1]),
                    'first_seen_iter': self.iteration,
                    'last_seen_iter': self.iteration
                })
        
        self.suspected_dead_drones = [
            s for s in self.suspected_dead_drones 
            if (self.iteration - s['last_seen_iter']) < 10
        ]

        drones_to_remove = []
        for dead_pos in self.dead_drones:
            if np.linalg.norm(np.array(dead_pos) - np.array([px, py])) < 150.0:
                is_still_there = False
                for v_pos in visible_drones:
                    if np.linalg.norm(v_pos - np.array(dead_pos)) < self.dead_drone_radius:
                        is_still_there = True
                        break
                
                if not is_still_there:
                    print(f"[{self.identifier}] Dead drone at ({dead_pos[0]:.1f}, {dead_pos[1]:.1f}) disappeared - REMOVING KILL ZONE")
                    drones_to_remove.append(dead_pos)
        
        for d in drones_to_remove:
            self.dead_drones.remove(d)

        if self.dead_drones:
            grid_h, grid_w = self.grid.grid.shape
            radius_cells = int(50.0 / self.grid.resolution)
            val_wall = 100.0

            y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
            mask = x**2 + y**2 <= radius_cells**2

            for dx_world, dy_world in self.dead_drones:
                res = self.grid._conv_world_to_grid(dx_world, dy_world)
                
                r_idx = int(res[0])
                c_idx = int(res[1])

                r_min = max(0, r_idx - radius_cells)
                r_max = min(grid_h, r_idx + radius_cells + 1)
                c_min = max(0, c_idx - radius_cells)
                c_max = min(grid_w, c_idx + radius_cells + 1)

                mask_r_min = r_min - (r_idx - radius_cells)
                mask_r_max = mask_r_min + (r_max - r_min)
                mask_c_min = c_min - (c_idx - radius_cells)
                mask_c_max = mask_c_min + (c_max - c_min)

                if r_max > r_min and c_max > c_min:
                    region = self.grid.grid[r_min:r_max, c_min:c_max]
                    region_mask = mask[mask_r_min:mask_r_max, mask_c_min:mask_c_max]
                    region[region_mask] = val_wall


    def find_safe_frontier_points(self) -> list:
        """
        Detect safe frontier points for exploration.
        """
        grid_map = self.grid.grid 
        
        dx = ndimage.sobel(grid_map, axis=1)
        dy = ndimage.sobel(grid_map, axis=0)

        mag_sq = dx**2 + dy**2
        frontier_mask = (mag_sq > 25.0)

        is_wall = (grid_map >= 4.0) 
        danger_zone = binary_dilation(is_wall, iterations=2)
        
        is_unknown = (grid_map >= -2.0) & (grid_map <= 2.0)
        
        near_unknown = binary_dilation(is_unknown, iterations=1)
        frontier_mask &= (~danger_zone) & near_unknown

        labeled, num_features = ndimage.label(frontier_mask)
        
        if num_features == 0:
            return []

        slices = ndimage.find_objects(labeled)
        
        self.frontier_clusters = []
        min_cluster_size = 5
        
        for i, sl in enumerate(slices):
            if sl is None: continue
                
            cluster_mask = (labeled[sl] == (i + 1))
            size = np.sum(cluster_mask)
            
            if size >= min_cluster_size:
                coords = np.argwhere(cluster_mask)
                mean_y = coords[:, 0].mean() + sl[0].start
                mean_x = coords[:, 1].mean() + sl[1].start
                
                x_world, y_world = self.grid._conv_grid_to_world(mean_y, mean_x)
                
                self.frontier_clusters.append({
                    "barycenter": np.array([x_world, y_world]),
                    "size": int(size)
                })

        return [c["barycenter"] for c in self.frontier_clusters]

    
    def draw_bottom_layer(self):
        """
        Draw the computed path and detected entities.
        """

        palette = [
            (200, 60, 60),
            (60, 200, 60),
            (60, 60, 200),
            (200, 200, 60),
            (200, 60, 200),
            (60, 200, 200),
            (255, 128, 0),
            (128, 0, 255),
        ]
        color_idx = int(self.identifier) % len(palette)
        detection_color = palette[color_idx]
    

        if hasattr(self, 'frontier_clusters') and self.frontier_clusters :
            if len(self.frontier_clusters) > 5:
                sorted_clusters = sorted(
                    self.frontier_clusters,
                    key=lambda c: np.linalg.norm(c['barycenter'] - self.current_pose[:2])
                )[:5]
            else:
                sorted_clusters = self.frontier_clusters
            
            for cluster in sorted_clusters:
                bc = cluster.get('barycenter')
                if bc is not None:
                    ptb = bc + self._half_size_array
                    arcade.draw_circle_filled(ptb[0], ptb[1], radius=6, color=detection_color)

        try:
            if hasattr(self, 'wounded_to_rescue') and self.wounded_to_rescue:
                for (xw, yw) in self.wounded_to_rescue:
                    pt = np.array([xw, yw]) + self._half_size_array

                    assigned_drone_id = None
                    
                    for w_pos, drone_id in self.wounded_assignments.items():
                        if isinstance(w_pos, str):
                            try:
                                coords = [float(x) for x in w_pos.strip("()").split(",")]
                                kx, ky = coords[0], coords[1]
                            except: continue
                        else:
                            kx, ky = w_pos[0], w_pos[1]
                        
                        if math.hypot(xw - kx, yw - ky) < 10.0:
                            assigned_drone_id = drone_id
                            break
                    
                    if assigned_drone_id is not None:
                        color = palette[int(assigned_drone_id) % len(palette)]
                        label = f"ASSIGNED: DRONE {assigned_drone_id}"
                    else:
                        color = (255, 255, 255)
                        label = "AVAILABLE"

                    arcade.draw_circle_outline(pt[0], pt[1], 20, color, 2)
                    arcade.draw_text(label, pt[0] + 25, pt[1] - 10, color, 11, bold=True)
        except Exception:
            pass

        try:
            if hasattr(self, 'rescue_zone_points') and self.rescue_zone_points:
                for (xr, yr) in self.rescue_zone_points:
                    pt = np.array([xr, yr]) + self._half_size_array
                    arcade.draw_rectangle_outline(pt[0], pt[1], width=30, height=30, color=(0,160,0), border_width=2)
        except Exception:
            pass

        if self.path and len(self.path) > 0:
            radius = 7
            blue = (0,0,255)
            green = (0,255,0)
            for pt in self.path:
                point_arcade = pt + self._half_size_array
                arcade.draw_circle_filled(point_arcade[0], point_arcade[1], radius=radius, color=blue)
            for i in range(len(self.path)-1):
                p1 = self.path[i] + self._half_size_array
                p2 = self.path[i+1] + self._half_size_array
                arcade.draw_line(p1[0], p1[1], p2[0], p2[1], color=green, line_width=3)

    
        

        my_screen_pos = self.current_pose[:2] + self._half_size_array
        arcade.draw_circle_filled(my_screen_pos[0], my_screen_pos[1], 
                                radius=18, color=detection_color)
        
    

        try:
            current_pose_screen = self.current_pose[:2] + self._half_size_array
            state_name = self.state.name if hasattr(self.state, 'name') else str(self.state)
            arcade.draw_text(state_name, 
                           current_pose_screen[0] - 30, 
                           current_pose_screen[1] + 25, 
                           (255, 255, 255), 
                           12, 
                           bold=True)
        except Exception:
            pass
            
        try:
            if hasattr(self, 'return_area_points') and self.return_area_points:
                for (rx, ry) in self.return_area_points:
                    pt = np.array([rx, ry]) + self._half_size_array
                    arcade.draw_rectangle_outline(pt[0], pt[1], width=30, height=30, color=(0, 160, 255), border_width=2)
                    arcade.draw_text("RA", pt[0] + 12, pt[1] + 12, (0, 120, 200), 10)
        except Exception:
            pass

 
    def follow_path(self, lidar_data) -> CommandsDict:
        """
        Follow a computed path using Pure Pursuit with PID control.
        """
        if not self.path:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        
        min_lidar_dist = min(lidar_data) if lidar_data is not None and len(lidar_data) > 0 else 999
        
        if min_lidar_dist < 40.0:
            lookahead_dist = 15.0
        else:
            lookahead_dist = getattr(self, 'path_lookahead_distance', 40.0)

        lookahead_idx = 0
        for i, wp in enumerate(self.path):
            if np.linalg.norm(wp - self.current_pose[:2]) > lookahead_dist:
                lookahead_idx = i
                break
        target_pos = self.path[min(lookahead_idx, len(self.path)-1)]

        delta_pos = target_pos - self.current_pose[:2]
        heading = self.current_pose[2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        x_err = cos_h * delta_pos[0] + sin_h * delta_pos[1]
        y_err = -sin_h * delta_pos[0] + cos_h * delta_pos[1]

        angle_error = normalize_angle(target_angle - heading)
        deriv_angle = angle_error - self.prev_angle_error
        
        Kp_rot = self.Kp
        Kd_rot = self.Kd

        if abs(angle_error) < math.radians(10):
            Kp_rot *= 0.6
            Kd_rot *= 0.8

        rotation_speed = Kp_rot * angle_error + Kd_rot * deriv_angle
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error

        Kp_lat = 0.08
        Kd_lat = 0.03
        
        if not hasattr(self, 'prev_lat_error'): self.prev_lat_error = 0.0
        lat_deriv = y_err - self.prev_lat_error
        lateral_cmd = Kp_lat * y_err + Kd_lat * lat_deriv
        
        if abs(angle_error) < 0.1:
            lateral_cmd *= 0.7
            
        lateral_cmd = float(np.clip(lateral_cmd, -1.0, 1.0))
        self.prev_lat_error = y_err
        
        max_speed = 16.0
        target_speed = max(0.0, min(max_speed, x_err * 0.25 + 0.5))

        measured_vel = self.measured_velocity()
        if measured_vel is None:
            measured_speed = 0.0
        else:
            measured_speed = math.sqrt(measured_vel[0] ** 2 + measured_vel[1] ** 2)
        
        speed_error = target_speed - measured_speed
        deriv_speed = speed_error - self.prev_speed_error
        
        Kp_f = self.Kp_pos
        Kd_f = self.Kd_pos 
        forward_cmd = Kp_f * speed_error + Kd_f * deriv_speed

        abs_angle_error = abs(angle_error)
        if abs_angle_error > 1.0:
            forward_cmd *= 0.1
        elif abs_angle_error > 0.6:
            forward_cmd *= 0.3
        elif abs_angle_error > 0.3:
            forward_cmd *= 0.6
        elif abs_angle_error > 0.15:
            forward_cmd *= 0.85

        forward_cmd = float(np.clip(forward_cmd, -1.0, 1.0))
        self.prev_speed_error = speed_error

        close_thresh = 50.0 
        drop_until = -1
        max_check = min(5, len(self.path))
        for i in range(max_check):
            d = np.linalg.norm(self.path[i] - self.current_pose[:2])
            if d < close_thresh:
                drop_until = i
            else:
                break
                
        if drop_until >= 0:
            self.path = self.path[drop_until+1:]

        return {"forward": forward_cmd, "lateral": lateral_cmd, "rotation": rotation_speed}


    def update_pose(self):
        """
        Update drone pose using Kalman filtering for GPS and odometry for no-GPS zones.
        """
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        measured_vel = self.measured_velocity()
        
        current_time = self.iteration * 0.1
        if self.kf_last_time > 0:
            self.kf_dt = current_time - self.kf_last_time
        else:
            self.kf_dt = 0.1
        self.kf_last_time = current_time

        if compass_angle is not None:
            self.current_pose[2] = compass_angle
        
        if gps_pos is not None and not np.isnan(gps_pos[0]):
            if not self.kf_initialized:
                self.kf_state[0] = gps_pos[0]
                self.kf_state[1] = gps_pos[1]
                self.kf_state[2] = 0.0
                self.kf_state[3] = 0.0
                self.kf_initialized = True
            
            F = np.array([
                [1, 0, self.kf_dt, 0],
                [0, 1, 0, self.kf_dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            self.kf_state = F @ self.kf_state
            self.kf_P = F @ self.kf_P @ F.T + self.kf_Q

            if measured_vel is not None:
                self.kf_state[2] = 0.7 * measured_vel[0] + 0.3 * self.kf_state[2]
                self.kf_state[3] = 0.7 * measured_vel[1] + 0.3 * self.kf_state[3]

            
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            
            z = np.array([gps_pos[0], gps_pos[1]])
            y = z - H @ self.kf_state
            
            S = H @ self.kf_P @ H.T + self.kf_R
            
            K = self.kf_P @ H.T @ np.linalg.inv(S)
            
            self.kf_state = self.kf_state + K @ y
            
            I = np.eye(4)
            self.kf_P = (I - K @ H) @ self.kf_P
            
            self.current_pose[0] = self.kf_state[0]
            self.current_pose[1] = self.kf_state[1]

        else:
            odom_data = self.odometer_values()
            if odom_data is None:
                return
            
            dist_travel = odom_data[0]
            alpha = odom_data[1]
            theta = odom_data[2]
            
            if compass_angle is None:
                self.current_pose[2] += theta
                self.current_pose[2] = normalize_angle(self.current_pose[2])
            
            heading = self.current_pose[2]
            
            travel_direction = heading + alpha
            
            dx_world = dist_travel * math.cos(travel_direction)
            dy_world = dist_travel * math.sin(travel_direction)
            
            if self.kf_initialized:
                if self.kf_dt > 0:
                    vx_odom = dx_world / self.kf_dt
                    vy_odom = dy_world / self.kf_dt
                    
                    self.kf_state[2] = 0.8 * vx_odom + 0.2 * self.kf_state[2]
                    self.kf_state[3] = 0.8 * vy_odom + 0.2 * self.kf_state[3]
                
                F = np.array([
                    [1, 0, self.kf_dt, 0],
                    [0, 1, 0, self.kf_dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                self.kf_state = F @ self.kf_state
                
                self.kf_P = F @ self.kf_P @ F.T + self.kf_Q * 1.2
                
                self.current_pose[0] = self.kf_state[0]
                self.current_pose[1] = self.kf_state[1]
            else:
                self.current_pose[0] += dx_world
                self.current_pose[1] += dy_world
                


    def process_communication_sensor(self):
        """
        Process messages from other drones with reduced overhead.
        """
        if not self.communicator:
            return

        dedup_radius = 50.0
        received_messages = self.communicator.received_messages
        current_iteration = self.iteration

        all_wounded = []
        all_assignments = {}
        all_grasped = set()
        all_rescue_zones = []
        all_frontier_clusters = []
        other_drones_positions = []

        for msg in received_messages:

            other_message = msg[1]
            other_id = other_message.get("drone_id")
            
            if other_id == self.identifier:
                continue     

            pos = other_message.get("drone_pose")
            if pos is not None:
                other_drones_positions.append((np.array(pos), other_id))

            
            if "wounded_list" in other_message:
                all_wounded.extend(other_message["wounded_list"])
        
            all_assignments.update(other_message.get("wounded_assignments", {}))
        
            all_grasped.update(
                tuple(w) for w in other_message.get("grasped_wounded", []) if w is not None
            )
            
            if "rescue_list" in other_message:
                for r in all_rescue_zones:
                    self._add_or_merge_rescue_point(r)
        

            if "removed_wounded" in other_message:
                for rw_new in other_message["removed_wounded"]:
                    key = (round(rw_new[0] / dedup_radius) * dedup_radius, 
                        round(rw_new[1] / dedup_radius) * dedup_radius)
                    
                    if key not in self.removed_wounded_set:
                        self.removed_wounded.append(rw_new)
                        self.removed_wounded_set.add(key)
                           
        
            if "frontier_clusters" in other_message:
                all_frontier_clusters.extend(other_message["frontier_clusters"])
        
            if "grid_data" in other_message:
                other_grid = np.array(other_message["grid_data"])
                
                other_found_wall = (other_grid >= 4.0)
                
                other_found_free = (other_grid <= -5.0) & (self.grid.grid < 4.0)
                
                self.grid.grid[other_found_wall] = other_grid[other_found_wall]
                self.grid.grid[other_found_free] = other_grid[other_found_free]

        self.other_drones_positions = other_drones_positions
    
        all_frontier_clusters.extend([
            {"barycenter": cluster["barycenter"].tolist()}
            for cluster in getattr(self, "frontier_clusters", [])
        ])
    
        merged_wounded = list(self.wounded_to_rescue)
        for w in all_wounded:
            if all(math.hypot(w[0] - wx, w[1] - wy) > dedup_radius for (wx, wy) in merged_wounded):
                merged_wounded.append(tuple(w))
    
        for rw in self.removed_wounded:
            merged_wounded = [
                (wx, wy) for (wx, wy) in merged_wounded
                if math.hypot(rw[0] - wx, rw[1] - wy) > dedup_radius
                ]
    
        deduped_wounded = []
        for w in merged_wounded:
            if all(math.hypot(w[0] - wx, w[1] - wy) > dedup_radius for (wx, wy) in deduped_wounded):
                deduped_wounded.append(w)
    
        self.wounded_to_rescue = deduped_wounded
    
        for w, drone_id in all_assignments.items():
            if w not in self.wounded_assignments:
                self.wounded_assignments[w] = drone_id
    
        if not hasattr(self, "other_grasped_wounded"):
            self.other_grasped_wounded = set()
        self.other_grasped_wounded = all_grasped
    
        for r in all_rescue_zones:
            if r not in self.rescue_zone_points:
                self.rescue_zone_points.append(r)
    
        if all_frontier_clusters:
            deduped_barycenters = []
            dedup_radius_frontier = 60.0
            for cl in all_frontier_clusters:
                bc = np.array(cl["barycenter"])
                if all(np.linalg.norm(bc - np.array(b)) > dedup_radius_frontier for b in deduped_barycenters):
                    deduped_barycenters.append(bc.tolist())
            self.shared_frontier_barycenters = deduped_barycenters

        self.wounded_to_rescue = merged_wounded

            
    def find_free_position_for_unstuck(self):
        """
        Find the first safe free position at a medium distance to escape when stuck.
        Returns a position (x, y) in world coordinates, or None if not found.
        """
        grid_map = self.grid.grid
        SEUIL_FREE = -10.0
        SEUIL_MUR = 5.0

        is_free = (grid_map < SEUIL_FREE)
        is_wall = (grid_map >= SEUIL_MUR)
        struct = np.ones((6, 6), dtype=bool)
        danger_zone = binary_dilation(is_wall, structure=struct, iterations=1)
        safe_free = is_free & (~danger_zone)

        if not np.any(safe_free):
            return None

        current_grid_pos = self.grid._conv_world_to_grid(self.current_pose[0], self.current_pose[1])
        current_y, current_x = int(current_grid_pos[0]), int(current_grid_pos[1])

        safe_positions = np.argwhere(safe_free)
        if len(safe_positions) == 0:
            return None

        distances = np.sqrt(
            (safe_positions[:, 0] - current_y) ** 2 +
            (safe_positions[:, 1] - current_x) ** 2
        )

        min_dist = 3
        max_dist = 6
        for idx, d in enumerate(distances):
            if min_dist <= d <= max_dist:
                chosen_pos = safe_positions[idx]
                x_free, y_free = self.grid._conv_grid_to_world(chosen_pos[0], chosen_pos[1])
                return (x_free, y_free)

        return None

    def check_and_handle_general_stuck(self):
        """
        Check if the drone is stuck and handle it by finding a new target position.
        Returns True if currently unstucking, False otherwise.
        """
        if self.iteration % 10 != 0:
            if self.is_unstucking and self.path:
                return True
            return False
        
        if self.last_unstuck_check_pos is None:
            self.last_unstuck_check_pos = self.current_pose[:2].copy()
            return False
        
        movement = np.linalg.norm(self.current_pose[:2] - self.last_unstuck_check_pos)
        self.last_unstuck_check_pos = self.current_pose[:2].copy()
        
        if movement > 10.0:
            self.general_stuck_counter = 0
            self.is_unstucking = False
            return False
        
        self.general_stuck_counter += 1
        
        if self.general_stuck_counter > 5:
            print(f"[{self.identifier}] General stuck detected! Counter: {self.general_stuck_counter}")
            
            if self.unstuck_target is None or self.general_stuck_counter % 10 == 0:
                self.unstuck_target = self.find_free_position_for_unstuck()
                
                if self.unstuck_target:
                    print(f"[{self.identifier}] Found unstuck target: {self.unstuck_target}")
                    self.path = self.create_path(self.current_pose[:2], self.unstuck_target)
                    self.is_unstucking = True
                else:
                    print(f"[{self.identifier}] No unstuck target found")
                    self.is_unstucking = False
            
            return True
        
        return False


    def _add_or_merge_rescue_point(self, new_point):
            """
            Add a rescue point only if it is far from all existing points.
            This maintains stability by not updating points unnecessarily.
            """
            nx, ny = new_point
            MAX_RESCUE_POINTS = 5
            DEDUP_RADIUS = 80.0 

            for (rx, ry) in self.rescue_zone_points:
                dist = math.hypot(rx - nx, ry - ny)
                if dist < DEDUP_RADIUS:
                    return

            if len(self.rescue_zone_points) < MAX_RESCUE_POINTS:
                self.rescue_zone_points.append((nx, ny))



    def drone_repulsion(self, command):
            """
            Apply potential field repulsion to push the drone away from others if too close.
            """
            if not hasattr(self, 'other_drones_positions') or not self.other_drones_positions:
                return command

            SAFE_DIST = 120.0
            GAIN = 3.5

            repulsion_forward = 0.0
            repulsion_lateral = 0.0
            
            my_pos = self.current_pose[:2]
            my_theta = self.current_pose[2]

            for other_info in self.other_drones_positions:
                other_pos = other_info[0]
                
                dx = my_pos[0] - other_pos[0]
                dy = my_pos[1] - other_pos[1]
                dist = math.hypot(dx, dy)

                if 0 < dist < SAFE_DIST:
                    force = (SAFE_DIST - dist) / SAFE_DIST
                    angle_global = math.atan2(dy, dx)
                    angle_local = normalize_angle(angle_global - my_theta)
                    
                    repulsion_forward += force * math.cos(angle_local)
                    repulsion_lateral += force * math.sin(angle_local)

            if abs(repulsion_forward) > 0.01 or abs(repulsion_lateral) > 0.01:
                command["forward"] += repulsion_forward * GAIN
                command["lateral"] += repulsion_lateral * GAIN
                
                command["forward"] = float(np.clip(command["forward"], -1.0, 1.0))
                command["lateral"] = float(np.clip(command["lateral"], -1.0, 1.0))
                
            return command
    

    def update_breadcrumbs(self):
        """
        Update breadcrumbs for dead reckoning in case path finding fails.
        """
        if self.state == self.Activity.GOING_TO_RESCUE_CENTER:
            return

        current_pos_tuple = (self.current_pose[0], self.current_pose[1])

        if self.last_breadcrumb_pos is None:
            self.breadcrumbs.append(current_pos_tuple)
            self.last_breadcrumb_pos = current_pos_tuple
            return

        dist = math.hypot(current_pos_tuple[0] - self.last_breadcrumb_pos[0], 
                        current_pos_tuple[1] - self.last_breadcrumb_pos[1])

        if dist >= self.breadcrumb_spacing:
            self.breadcrumbs.append(current_pos_tuple)
            self.last_breadcrumb_pos = current_pos_tuple
    

    def go_to_wounded(self, lidar_data) -> CommandsDict:
        """
        Navigate to the wounded person.
        Uses path following for distant targets and direct approach when close.
        """
        if self.current_target_wounded is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        dist_to_wounded = np.linalg.norm(np.array(self.current_target_wounded) - self.current_pose[:2])

        if dist_to_wounded > 80.0:
            return self.follow_path(lidar_data)
        
        else:
            self.grasped_wounded_angle = self.get_wounded_orientation()

        delta_pos = np.array(self.current_target_wounded) - self.current_pose[:2]
        target_angle = math.atan2(delta_pos[1], delta_pos[0])
        heading = self.current_pose[2]
        angle_error = normalize_angle(target_angle - heading)

        deriv_error = angle_error - getattr(self, "prev_angle_error", 0.0)
        rotation_speed = self.Kp * angle_error + self.Kd * deriv_error
        rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
        self.prev_angle_error = angle_error

        ALIGNMENT_THRESHOLD = math.radians(5.0)

        if abs(angle_error) > ALIGNMENT_THRESHOLD:
            return {"forward": 0.0, "lateral": 0.0, "rotation": rotation_speed}
        else:
            return {"forward": 1.0, "lateral": 0.0, "rotation": rotation_speed}


    def get_wounded_orientation(self):
        """
        Get the orientation angle of the currently detected wounded person.
        """
        try:
            detections = self.semantic_values()
            if not detections:
                return None

            for data in detections:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    angle = float(getattr(data, 'angle', 0.0))
                    return angle
                
        except Exception as e:
            print(f"[{self.identifier}] Error processing semantic data: {e}")
        return None
        



    def go_to_rescue_center_oriented(self, lidar_data) -> CommandsDict:
        """
        Navigate to the rescue center while maintaining orientation for pushing the grasped wounded.
        """
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        if not self.rescue_zone_points:
            return command

        target_point = np.array(self.rescue_zone_points[0])
        target_vector = target_point - self.current_pose[:2]
        dist_to_center = np.linalg.norm(target_vector)
        
        angle_to_center = math.atan2(target_vector[1], target_vector[0])
        
        grasp_angle = getattr(self, "grasped_wounded_angle", None)
        if grasp_angle is None:
            grasp_angle = math.pi
            
        target_orientation = normalize_angle(angle_to_center - grasp_angle)
        
        angle_error = normalize_angle(target_orientation - self.current_pose[2])

        
        if abs(angle_error) > 0.03:
            command["rotation"] = np.clip(self.Kp * angle_error, -0.6, 0.6)
            command["forward"] = 1.0
        else:
            command["rotation"] = 0.0
            
            if dist_to_center > 12.0:
                direction = 1.0 if abs(grasp_angle) < math.pi/2 else -0.3
                command["forward"] = direction
            else:
                command["forward"] = 1.0

        return command
    

    def add_return_area_point(self, new_point):
        """
        Add a return area point if it is far from existing points and rescue centers.
        """
        nx, ny = new_point
        DEDUP_RADIUS = 50.0 
        MIN_DIST_FROM_RESCUE = 170.0

        if hasattr(self, 'rescue_zone_points') and self.rescue_zone_points:
                    for (xr, yr) in self.rescue_zone_points:
                        dist_to_rescue = math.hypot(nx - xr, ny - yr)
                        if dist_to_rescue < MIN_DIST_FROM_RESCUE:
                            return
                        
        for (rx, ry) in self.return_area_points:
            dist = math.hypot(rx - nx, ry - ny)
            if dist < DEDUP_RADIUS:
                return

        self.return_area_points.append((nx, ny))


    def local_exploration_fallback(self):
        """
        Fallback mechanism: move towards the nearest unexplored cell or random walk if none found.
        Returns True if a path was found, False otherwise.
        """
        grid_map = self.grid.grid
        SEUIL_UNEXPLORED_MIN = -4.99
        SEUIL_UNEXPLORED_MAX = 2.99
        is_unexplored = (grid_map >= SEUIL_UNEXPLORED_MIN) & (grid_map <= SEUIL_UNEXPLORED_MAX)
        is_wall = (grid_map >= 3.0)
        unexplored_mask = is_unexplored & (~is_wall)
        
        if np.any(unexplored_mask):
            try:
                current_grid = self.grid._conv_world_to_grid(self.current_pose[0], self.current_pose[1])
                current_grid = np.array([int(current_grid[0]), int(current_grid[1])])
                unexplored_indices = np.argwhere(unexplored_mask)
                distances = np.linalg.norm(unexplored_indices - current_grid, axis=1)
                nearest_idx = int(np.argmin(distances))
                nearest_unexplored = unexplored_indices[nearest_idx]
                
                target_world = self.grid._conv_grid_to_world(int(nearest_unexplored[0]), int(nearest_unexplored[1]))
                target_world = (float(target_world[0]), float(target_world[1]))
                
                path = self.create_path(self.current_pose[:2], target_world)
                if path and len(path) > 0:
                    self.path = path
                    self.target_point = np.array(target_world)
                    print(f"[{self.identifier}] [FALLBACK] SUCCESS: Moving to nearest unexplored cell at {target_world}")
                    return True
                else:
                    print(f"[{self.identifier}] [FALLBACK] Could not find path to unexplored cell at {target_world}")
                    return False
            except Exception as e:
                print(f"[{self.identifier}] [FALLBACK] Exception: {e}")
                return False
        else:
            print(f"[{self.identifier}] [FALLBACK] No unexplored areas found")
            return False
            

    def go_to_return_area(self, lidar_data) -> CommandsDict:
        """
        Navigate to the return area.
        """
        if self.return_area_points:
            target_index = int(self.identifier) % len(self.return_area_points)
            target_zone = self.return_area_points[target_index]
            self.target_point = target_zone
            self.path = self.create_path(self.current_pose[:2], target_zone, explored_only=True)
            self.state = self.Activity.GOING_TO_RETURN_AREA

            return self.follow_path(lidar_data) if lidar_data is not None else {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        else:
            print(f"[{self.identifier}] No return area points available!")
