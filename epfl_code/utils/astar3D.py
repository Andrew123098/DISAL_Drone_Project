import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional


class AStar3D:
    def __init__(self, grid: np.ndarray, move_type: str = '6d'):
        """
        Initialize the 3D A* algorithm with the grid and movement type.

        Args:
            grid: numpy array where:
                 0 = free space
                -1 = gates
                -2 = control points
                -4 = flight area
                 1 = obstacles (beams/takeoff pads)
            move_type: '6d' for cardinal directions or '26d' for all possible 3D movements
        """
        self.grid = self.convert_to_binary(grid)
        self.move_type = move_type

        # Define movement patterns
        if move_type == '6d':
            # 6-directional movement (cardinal directions)
            self.moves = [(1, 0, 0), (-1, 0, 0),  # x
                          (0, 1, 0), (0, -1, 0),  # y
                          (0, 0, 1), (0, 0, -1)]  # z
        elif move_type == '26d':
            # 26-directional movement (all possible 3D combinations)
            self.moves = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue  # skip no movement
                        self.moves.append((dx, dy, dz))
        else:
            raise ValueError("move_type must be either '6d' or '26d'")

    def heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """Calculate the 3D Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def is_valid(self, pos: Tuple[int, int, int]) -> bool:
        """Check if a position is valid (within grid and not an obstacle)."""
        x, y, z = pos
        if (x < 0 or y < 0 or z < 0 or
                x >= self.grid.shape[0] or
                y >= self.grid.shape[1] or
                z >= self.grid.shape[2]):
            return False
        return self.grid[x, y, z] != 1  # 1 is obstacle

    def reconstruct_path(self, came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]],
                         current: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Reconstruct the path from start to goal."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start to goal

    def search(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """
        Perform A* search between two points and return the path if found.

        Args:
            start: Tuple of (x, y, z) grid coordinates
            goal: Tuple of (x, y, z) grid coordinates

        Returns:
            List of grid coordinates representing the path, or None if no path found
        """
        # Priority queue: (f_score, g_score, position)
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))

        came_from = {}
        g_scores = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for move in self.moves:
                neighbor = (current[0] + move[0], current[1] + move[1], current[2] + move[2])

                if not self.is_valid(neighbor):
                    continue

                # Cost calculation
                move_cost = np.sqrt(move[0] ** 2 + move[1] ** 2 + move[2] ** 2)
                tentative_g = current_g + move_cost

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None  # No path found

    def navigate_control_points(self, start: Tuple[int, int, int],
                                control_points: List[Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
        """
        Navigate through a sequence of control points.

        Args:
            start: Initial starting position (grid coordinates)
            control_points: Ordered list of control points to visit

        Returns:
            List of paths between consecutive points
        """
        full_path = []
        current_pos = start

        for point in control_points:
            path_segment = self.search(current_pos, point)
            if not path_segment:
                print(f"Warning: No path found from {current_pos} to {point}")
                break

            full_path.append(path_segment)
            current_pos = point  # Update position to end of this segment

        return full_path

    def convert_to_binary(self, occupancy_map):
        """
        Convert the labeled occupancy map to a binary version where:
            - 0 = free space (includes original free space, gates, control points)
            - 1 = obstacles (includes original obstacles and flight area)

        Parameters:
        - occupancy_map: The original 3D numpy array with multiple labels

        Returns:
        - binary_map: A 3D numpy array with only 0s (free) and 1s (obstacles)
        """
        # Create a copy to avoid modifying the original
        binary_map = np.zeros_like(occupancy_map)

        # Set obstacles (original obstacles (1) and flight area (-4))
        binary_map[(occupancy_map == 1) | (occupancy_map == -4)] = 1

        # Everything else (0, -1, -2) becomes free space (0)

        return binary_map

    def order_control_points(self, control_points, num_loops):
        if len(control_points) != 6:
            print(f"Warning: Expected 6 control points, but got {len(control_points)}. Returning original order.")
            return control_points

        order = [0, 1, 3, 5, 4, 2]
        ordered_control_points = []
        for i in order:
            ordered_control_points.append(tuple(control_points[i]))

        one_loop = ordered_control_points
        first_loop = ordered_control_points[1:]

        if num_loops == 1:
            ordered_control_points = first_loop
        elif num_loops == 2:
            ordered_control_points = first_loop + one_loop
        elif num_loops > 2:
            ordered_control_points = first_loop + one_loop
            for j in range(num_loops-2):
                ordered_control_points = ordered_control_points + one_loop

        return ordered_control_points
























# import numpy as np
# import heapq
# from typing import List, Tuple, Optional, Dict
#
# class AStar3D:
#     def __init__(self, grid: np.ndarray, move_type: str = '6d'):
#         """
#         Initialize the 3D A* algorithm with the grid and movement type.
#
#         Args:
#             grid: numpy array where:
#                  0 = free space
#                  1 = obstacle
#                 -1 = start position
#                 -2 = goal position
#             move_type: '6d' for cardinal directions or '26d' for all possible 3D movements
#         """
#         self.grid = grid
#         self.move_type = move_type
#         self.start = self._find_position(-1)
#         self.goal = self._find_position(-2)
#
#         # Validate grid
#         if self.start is None or self.goal is None:
#             raise ValueError("Start or goal position not found in grid")
#
#         # Define movement patterns
#         if move_type == '6d':
#             # 6-directional movement (cardinal directions)
#             self.moves = [(1, 0, 0), (-1, 0, 0),  # x
#                           (0, 1, 0), (0, -1, 0),  # y
#                           (0, 0, 1), (0, 0, -1)]  # z
#         elif move_type == '26d':
#             # 26-directional movement (all possible 3D combinations)
#             self.moves = []
#             for dx in [-1, 0, 1]:
#                 for dy in [-1, 0, 1]:
#                     for dz in [-1, 0, 1]:
#                         if dx == 0 and dy == 0 and dz == 0:
#                             continue  # skip no movement
#                         self.moves.append((dx, dy, dz))
#         else:
#             raise ValueError("move_type must be either '6d' or '26d'")
#
#     def _find_position(self, value: int) -> Optional[Tuple[int, int, int]]:
#         """Find the (x, y, z) position of a specific value in the grid."""
#         positions = np.argwhere(self.grid == value)
#         return tuple(positions[0]) if len(positions) > 0 else None
#
#     def heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
#         """Calculate the 3D Manhattan distance between two points."""
#         return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
#
#     def is_valid(self, pos: Tuple[int, int, int]) -> bool:
#         """Check if a position is valid (within grid and not an obstacle)."""
#         x, y, z = pos
#         if (x < 0 or y < 0 or z < 0 or
#             x >= self.grid.shape[0] or
#             y >= self.grid.shape[1] or
#             z >= self.grid.shape[2]):
#             return False
#         return self.grid[x, y, z] != 1  # 1 is obstacle
#
#     def reconstruct_path(self, came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]],
#                         current: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
#         """Reconstruct the path from start to goal using the came_from dictionary."""
#         path = [current]
#         while current in came_from:
#             current = came_from[current]
#             path.append(current)
#         return path[::-1]  # Reverse to get start to goal
#
#     def search(self) -> Optional[List[Tuple[int, int, int]]]:
#         """Perform A* search and return the path if found."""
#         # Priority queue: (f_score, g_score, position)
#         open_set = []
#         heapq.heappush(open_set, (0 + self.heuristic(self.start, self.goal), 0, self.start))
#
#         came_from = {}  # For path reconstruction
#         g_scores = {self.start: 0}  # Cost from start to current position
#
#         while open_set:
#             _, current_g, current = heapq.heappop(open_set)
#
#             if current == self.goal:
#                 return self.reconstruct_path(came_from, current)
#
#             for move in self.moves:
#                 neighbor = (current[0] + move[0], current[1] + move[1], current[2] + move[2])
#
#                 if not self.is_valid(neighbor):
#                     continue
#
#                 # Cost calculation:
#                 # For 6-directional: all moves cost 1
#                 # For 26-directional:
#                 #   - Cardinal moves (1 direction change) cost 1
#                 #   - Face diagonal (2 direction changes) cost sqrt(2)
#                 #   - Space diagonal (3 direction changes) cost sqrt(3)
#                 move_cost = np.sqrt(move[0]**2 + move[1]**2 + move[2]**2)
#                 tentative_g = current_g + move_cost
#
#                 if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
#                     came_from[neighbor] = current
#                     g_scores[neighbor] = tentative_g
#                     f_score = tentative_g + self.heuristic(neighbor, self.goal)
#                     heapq.heappush(open_set, (f_score, tentative_g, neighbor))
#
#         return None  # No path found