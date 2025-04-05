import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict

class AStar3D:
    def __init__(self, grid: np.ndarray, move_type: str = '6d'):
        """
        Initialize the 3D A* algorithm with the grid and movement type.
        
        Args:
            grid: numpy array where:
                 0 = free space
                 1 = obstacle
                -1 = start position
                -2 = goal position
            move_type: '6d' for cardinal directions or '26d' for all possible 3D movements
        """
        self.grid = grid
        self.move_type = move_type
        self.start = self._find_position(-1)
        self.goal = self._find_position(-2)
        
        # Validate grid
        if self.start is None or self.goal is None:
            raise ValueError("Start or goal position not found in grid")
        
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
    
    def _find_position(self, value: int) -> Optional[Tuple[int, int, int]]:
        """Find the (x, y, z) position of a specific value in the grid."""
        positions = np.argwhere(self.grid == value)
        return tuple(positions[0]) if len(positions) > 0 else None
    
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
        """Reconstruct the path from start to goal using the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start to goal
    
    def search(self) -> Optional[List[Tuple[int, int, int]]]:
        """Perform A* search and return the path if found."""
        # Priority queue: (f_score, g_score, position)
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(self.start, self.goal), 0, self.start))
        
        came_from = {}  # For path reconstruction
        g_scores = {self.start: 0}  # Cost from start to current position
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current == self.goal:
                return self.reconstruct_path(came_from, current)
            
            for move in self.moves:
                neighbor = (current[0] + move[0], current[1] + move[1], current[2] + move[2])
                
                if not self.is_valid(neighbor):
                    continue
                
                # Cost calculation:
                # For 6-directional: all moves cost 1
                # For 26-directional: 
                #   - Cardinal moves (1 direction change) cost 1
                #   - Face diagonal (2 direction changes) cost sqrt(2)
                #   - Space diagonal (3 direction changes) cost sqrt(3)
                move_cost = np.sqrt(move[0]**2 + move[1]**2 + move[2]**2)
                tentative_g = current_g + move_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        return None  # No path found