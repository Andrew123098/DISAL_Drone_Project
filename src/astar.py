# import drone_pkg.astar.visualize as viz
import visualize as viz
import numpy as np
import matplotlib.pyplot as plt
import heapq
from typing import List, Tuple, Optional, Dict
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize

class AStar:
    def __init__(self, grid: np.ndarray, grid_res: float, start_pos: Tuple[float, float], goal_pos: Tuple[float, float], move_type: str = '8d'):
        """
        Initialize with explicit start and goal positions in world coordinates
        """
        self.grid_resolution = grid_res
        self.grid = grid  # Use the provided grid instead of creating a new one
        
        # Convert and mark positions
        self.start = self._world_to_grid(start_pos)
        self.goal = self._world_to_grid(goal_pos)
        
        # Validate positions are within grid bounds
        if not self.is_valid(self.start):
            raise ValueError(f"Start position {start_pos} (grid: {self.start}) is out of bounds or in obstacle")
        if not self.is_valid(self.goal):
            raise ValueError(f"Goal position {goal_pos} (grid: {self.goal}) is out of bounds or in obstacle")
        
        # Mark positions in grid (optional)
        self.grid[self.start] = -1
        self.grid[self.goal] = -2
        
        self.move_type = move_type
        
        # Define movement patterns
        if move_type == '4d':
            self.moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        elif move_type == '8d':
            self.moves = [(0, 1), (1, 1), (1, 0), (1, -1), 
                         (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        else:
            raise ValueError("move_type must be either '4d' or '8d'")
    
    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid coordinates with (0,0) at center"""
        center_x = self.grid.shape[1] // 2
        center_y = self.grid.shape[0] // 2
        grid_x = center_x + int(round(world_pos[0] / self.grid_resolution))
        grid_y = center_y - int(round(world_pos[1] / self.grid_resolution))  # Note: y axis is inverted
        
        # Ensure coordinates are within bounds
        grid_x = max(0, min(grid_x, self.grid.shape[1] - 1))
        grid_y = max(0, min(grid_y, self.grid.shape[0] - 1))
        
        return (grid_y, grid_x)  # Return as (row, col)

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates back to world coordinates"""
        center_x = self.grid.shape[1] // 2
        center_y = self.grid.shape[0] // 2
        row, col = grid_pos
        world_x = (col - center_x) * self.grid_resolution
        world_y = (center_y - row) * self.grid_resolution  # Note: y axis is inverted
        return (world_x, world_y)
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within grid and not an obstacle)."""
        row, col = pos
        if row < 0 or col < 0 or row >= self.grid.shape[0] or col >= self.grid.shape[1]:
            return False
        return self.grid[row, col] <= 0  # Treat values <= 0 as traversable
    
    def reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                        current: Tuple[int, int], smooth_type: Optional[str] = None, **smooth_kwargs) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to goal using the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)

        astar_path = path[::-1]  # Reverse to get start to goal

        if smooth_type is None:
            return None, astar_path
        else:
            return self.smooth_path(astar_path, smooth_type, **smooth_kwargs), astar_path
    
    def search(self, smooth_type: Optional[str] = None, **smooth_kwargs) -> Optional[List[Tuple[int, int]]]:
        """Perform A* search and return the path if found."""
        # Priority queue: (f_score, g_score, position)
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(self.start, self.goal), 0, self.start))
        
        came_from = {}  # For path reconstruction
        g_scores = {self.start: 0}  # Cost from start to current position
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current == self.goal:
                return self.reconstruct_path(came_from, current, smooth_type, **smooth_kwargs)
            
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
    
    def smooth_path(self, path: List[Tuple[int, int]], smooth_type: str = 'spline', 
                   smoothing_factor: int = 3, alpha: float = 0.1, beta: float = 0.3) -> List[Tuple[float, float]]:
        """
        Smooth path using specified method
        Args:
            path: Raw A* path in grid coordinates
            smooth_type: 'spline' or 'gradient'
            smoothing_factor: For spline (number of interpolated points per segment)
            alpha: Smoothness weight for gradient method
            beta: Obstacle avoidance weight for gradient method
        Returns:
            Smoothed path in world coordinates
        """
        if len(path) < 4:
            return [self._grid_to_world(p) for p in path]
            
        world_path = np.array([self._grid_to_world(p) for p in path])
        
        if smooth_type == 'spline':
            return self._spline_smoothing(world_path, smoothing_factor)
        elif smooth_type == 'gradient':
            return self._gradient_smoothing(world_path, alpha, beta)
        else:
            raise ValueError(f"Unknown smooth_type: {smooth_type}")
        
    def _spline_smoothing(self, path: np.ndarray, smoothing_factor: int) -> List[Tuple[float, float]]:
        """Spline smoothing with reduced point density."""
        from scipy.interpolate import splev

        # Step 1: Reduce control points more aggressively
        skip = max(1, len(path) // 20)  # Use ~5 control points max (was 10)
        control_points = path[::skip]

        # Step 2: Cumulative distance parameterization
        dist = np.cumsum(np.sqrt(np.sum(np.diff(control_points, axis=0)**2, axis=1)))
        dist = np.insert(dist, 0, 0) / dist[-1]

        # Step 3: Fit splines
        spl_x = make_interp_spline(dist, control_points[:,0], k=3, bc_type='natural')
        spl_y = make_interp_spline(dist, control_points[:,1], k=3, bc_type='natural')

        # Step 4: Start with fewer initial points (reduced by 1 magnitude)
        initial_points = max(10, len(path) // 10)  # Was 20, now scaled by path length
        u = np.linspace(0, 1, initial_points)

        # Step 5: Reduced curvature-based refinement (2 iterations instead of 3)
        for _ in range(4):  # Was 3 iterations
            # Calculate curvature
            dx = splev(u, spl_x, der=1)
            ddx = splev(u, spl_x, der=2)
            dy = splev(u, spl_y, der=1)
            ddy = splev(u, spl_y, der=2)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

            # Normalize curvature
            curvature = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature) + 1e-6)
            curvature = curvature + 0.05  # Reduced from 0.1 to add fewer points

            # Add new points more conservatively
            new_u = []
            for i in range(len(u)-1):
                segment_length = u[i+1] - u[i]
                num_new_points = int(np.ceil(curvature[i] * 2))  # Reduced from 3
                if num_new_points > 0:
                    new_u.extend(np.linspace(u[i], u[i+1], num_new_points + 2)[:-1])
            new_u.append(u[-1])
            u = np.sort(np.unique(new_u))

        # Step 6: Generate final points
        smooth_x = splev(u, spl_x)
        smooth_y = splev(u, spl_y)

        # Downsample final path by 2 so we keep every other point
        ds = 10
        smooth_x = smooth_x[::ds]
        smooth_y = smooth_y[::ds]
        u = u[::ds]

        # Debug output
        print(f"Generated {len(u)} waypoints (reduced density)")
        print(f"Point density variation: {np.std(np.diff(u)):.4f}")
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        print(f"Curvature range: {np.min(curvature):.4f} to {np.max(curvature):.4f}")
        
        return list(zip(smooth_x, smooth_y))
        
    
    def _gradient_smoothing(self, path: np.ndarray, alpha: float, beta: float) -> List[Tuple[float, float]]:
        """Gradient-based path optimization"""
        def cost(optim_path):
            optim_path = optim_path.reshape((-1, 2))
            
            # Smoothness term (minimize curvature)
            smooth_cost = alpha * np.sum(np.diff(optim_path, n=2, axis=0)**2)
            
            # Obstacle term (penalize entering occupied cells)
            grid_coords = np.round(
                (optim_path - np.array([self._grid_to_world((0,0))])) / self.grid_resolution
            ).astype(int)
            
            # Clip coordinates to grid bounds
            grid_coords[:,0] = np.clip(grid_coords[:,0], 0, self.grid.shape[0]-1)
            grid_coords[:,1] = np.clip(grid_coords[:,1], 0, self.grid.shape[1]-1)
            
            obs_cost = beta * np.sum(self.grid[grid_coords[:,0], grid_coords[:,1]] > 0)
            
            # Path length term (optional)
            length_cost = 0.01 * np.sum(np.sqrt(np.sum(np.diff(optim_path, axis=0)**2, axis=1)))
            
            return smooth_cost + obs_cost + length_cost

        # Optimize using L-BFGS-B with bounds checking
        bounds = []
        for _ in path:
            bounds.extend([(None, None), (None, None)])  # No bounds on x,y
            
        result = minimize(
            cost,
            path.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        return list(map(tuple, result.x.reshape((-1, 2))))

def main():
    START_POS = (2,2)
    GOAL_POS = (-2,-2)
    RESOLUTION = 0.1
    GRID_SIZE = 6.0
    
    visualizer = viz.Visualize(size=(GRID_SIZE, GRID_SIZE), resolution=RESOLUTION, 
                             start_pos=START_POS, goal_pos=GOAL_POS)
    map = visualizer.occupancy_map()

    # Generate paths
    astar = AStar(map, grid_res=RESOLUTION, start_pos=START_POS, goal_pos=GOAL_POS, move_type='8d') 
    
    # Get all path versions
    # gradient_path, astar_path = astar.search(smooth_type='gradient', alpha=0.2, beta=0.4)
    spline_path, astar_path = astar.search(smooth_type='spline', smoothing_factor=5)
    
    # Mark original A* path on grid
    for pos in astar_path[1:-1]:
        map[pos] = 3  # Using value 3 for original path

    # Visualize all paths
    visualizer.plot_map_with_paths(
        map,
        astar_path=astar_path,
        spline_path=spline_path,
        # gradient_path=gradient_path
    )

if __name__ == "__main__":
    main()
