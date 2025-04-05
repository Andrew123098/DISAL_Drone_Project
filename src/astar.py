import src.visualize as viz
import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict

class AStar:
    def __init__(self, grid: np.ndarray, move_type: str = '4d'):
        """
        Initialize the A* algorithm with the grid and movement type.
        
        Args:
            grid: numpy array where:
                 0 = free space
                 1 = obstacle
                -1 = start position
                -2 = goal position
            move_type: '4d' for cardinal directions or '8d' for diagonal movement
        """
        self.grid = grid
        self.move_type = move_type
        self.start = self._find_position(-1)
        self.goal = self._find_position(-2)
        
        # Validate grid
        if self.start is None or self.goal is None:
            raise ValueError("Start or goal position not found in grid")
        
        # Define movement patterns
        if move_type == '4d':
            self.moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        elif move_type == '8d':
            self.moves = [(0, 1), (1, 1), (1, 0), (1, -1), 
                          (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        else:
            raise ValueError("move_type must be either '4d' or '8d'")
    
    def _find_position(self, value: int) -> Optional[Tuple[int, int]]:
        """Find the (row, col) position of a specific value in the grid."""
        positions = np.argwhere(self.grid == value)
        return tuple(positions[0]) if len(positions) > 0 else None
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within grid and not an obstacle)."""
        row, col = pos
        if row < 0 or col < 0 or row >= self.grid.shape[0] or col >= self.grid.shape[1]:
            return False
        return self.grid[row, col] != 1  # 1 is obstacle
    
    def reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                        current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to goal using the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start to goal
    
    def search(self) -> Optional[List[Tuple[int, int]]]:
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
                neighbor = (current[0] + move[0], current[1] + move[1])
                
                if not self.is_valid(neighbor):
                    continue
                
                # Cost for 4-directional moves is 1, diagonal moves cost sqrt(2)
                move_cost = 1 if abs(move[0]) + abs(move[1]) == 1 else np.sqrt(2)
                tentative_g = current_g + move_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        return None  # No path found


def main():
    # Compute the occupancy map
    visualizer = viz.Visualize()
    map = visualizer.occupancy_map()

    # 4-directional movement
    astar_4d = AStar(map, '4d')
    path_4d = astar_4d.search()
    
    # 8-directional movement
    astar_8d = AStar(map, '8d')
    path_8d = astar_8d.search()

    # Colorize astar path on the map
    for pos in path_4d[1:-1]:
        map[pos] = 2

    for pos in path_8d[1:-1]:
        map[pos] = 3

    # Plot the occupancy map
    visualizer.plot_map(map)

if __name__ == "__main__":
    main()
