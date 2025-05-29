import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import binary_dilation


class AStar3D:
    def __init__(self, grid: np.ndarray, move_type: str = '6d', safety_margin: int = 0):
        """
        Initialize the 3D A* algorithm with grid and movement type.

        Args:
            grid: 3D numpy array with labeled occupancy values.
            move_type: '6d' or '26d' movement types.
            safety_margin: Number of voxels to keep as buffer from obstacles.
        """
        self.original_grid = grid  # Save original
        self.grid = self.convert_to_binary(grid)
        self.move_type = move_type

        # Inflate obstacles for margin safety
        if safety_margin > 0:
            self.inflate_obstacles(safety_margin)

        # Movement directions
        if move_type == '6d':
            self.moves = [(1, 0, 0), (-1, 0, 0),
                          (0, 1, 0), (0, -1, 0),
                          (0, 0, 1), (0, 0, -1)]
        elif move_type == '26d':
            self.moves = [(dx, dy, dz)
                          for dx in [-1, 0, 1]
                          for dy in [-1, 0, 1]
                          for dz in [-1, 0, 1]
                          if not (dx == dy == dz == 0)]
        else:
            raise ValueError("move_type must be '6d' or '26d'")

    def heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """Calculate the 3D Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def inflate_obstacles(self, n: int):
        """
        Expand obstacle areas by n voxels in all directions to enforce a buffer zone.

        Args:
            n: Number of voxels to inflate around each obstacle.
        """
        structure = np.ones((2 * n + 1, 2 * n + 1, 2 * n + 1))  # cube structuring element
        inflated = binary_dilation(self.grid == 1, structure=structure).astype(np.uint8)
        self.grid = inflated  # Replace the current grid with the inflated version

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



