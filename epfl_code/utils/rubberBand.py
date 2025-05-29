import numpy as np
from typing import List, Tuple


class RubberBand:
    def __init__(self,
                 occupancy_map: np.ndarray,
                 path_indices: List[Tuple[int, int, int]],
                 tension: float = 0.5,
                 repulsion_scale: float = 1.0,
                 obstacle_scale: float = 2.0,
                 neighbor_radius: int = 2):
        """
        Numerically stable rubber band smoother

        Args:
            occupancy_map: 3D grid where 0=free, 1=obstacle
            path_indices: Initial path as grid indices
            tension: Spring stiffness (0-1)
            repulsion_scale: Point repulsion strength
            obstacle_scale: Obstacle repulsion multiplier
            neighbor_radius: Grid cells to check for obstacles
        """
        # Convert grid to float32 to prevent overflow
        self.grid = occupancy_map.astype(np.float32)
        self.grid_shape = np.array(occupancy_map.shape, dtype=np.float32)

        # Convert path to float64 for precise calculations
        self.points = [np.array(p, dtype=np.float64) for p in path_indices]

        # Physics parameters
        self.tension = np.float64(tension)
        self.point_repulsion = np.float64(repulsion_scale)
        self.obstacle_repulsion = np.float64(obstacle_scale)
        self.radius = int(neighbor_radius)

        # Validate all points are within grid bounds
        for pt in self.points:
            assert self._is_in_bounds(pt), f"Point {pt} is out of grid bounds"

    def _is_in_bounds(self, point: np.ndarray) -> bool:
        """Check if point is within grid dimensions"""
        return all(0 <= coord < dim for coord, dim in zip(point, self.grid_shape))

    def _get_obstacle_force(self, point: np.ndarray) -> np.ndarray:
        """Safe obstacle force calculation with floating-point math"""
        force = np.zeros(3, dtype=np.float64)
        center = np.clip(np.round(point), 0, self.grid_shape - 1).astype(int)

        # Convert bounds to integers explicitly
        x_min = int(max(0, center[0] - self.radius))
        x_max = int(min(self.grid_shape[0] - 1, center[0] + self.radius))
        y_min = int(max(0, center[1] - self.radius))
        y_max = int(min(self.grid_shape[1] - 1, center[1] + self.radius))
        z_min = int(max(0, center[2] - self.radius))
        z_max = int(min(self.grid_shape[2] - 1, center[2] + self.radius))

        # Check each cell in neighborhood
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for z in range(z_min, z_max + 1):
                    if self.grid[x, y, z] == 1:
                        obstacle_pos = np.array([x, y, z], dtype=np.float64)
                        direction = point - obstacle_pos
                        distance = np.linalg.norm(direction)
                        if distance > 1e-6:
                            force += direction / (distance ** 2 + 1e-6)

        return force * self.obstacle_repulsion

    def smooth(self, iterations: int = 50, damping: float = 0.1) -> List[Tuple[int, int, int]]:
        """Numerically stable smoothing"""
        fixed_indices = {0, len(self.points) - 1}

        for _ in range(iterations):
            new_points = []
            for i in range(len(self.points)):
                if i in fixed_indices:
                    new_points.append(self.points[i].copy())
                    continue

                # Get neighbors with bounds checking
                prev = self.points[i - 1] if i > 0 else self.points[i].copy()
                next = self.points[i + 1] if i < len(self.points) - 1 else self.points[i].copy()

                # Calculate all forces in float64
                tension_force = self.tension * ((prev - self.points[i]) + (next - self.points[i]))
                obstacle_force = self._get_obstacle_force(self.points[i])

                # Combine forces with damping
                displacement = (tension_force + obstacle_force) * np.float64(damping)

                # Update position with bounds checking
                new_point = np.clip(
                    self.points[i] + displacement,
                    0,
                    self.grid_shape - 1
                )
                new_points.append(new_point)

            self.points = new_points

        # Convert back to integer indices with validation
        final_path = []
        for pt in self.points:
            rounded = np.round(pt).astype(int)
            if self._is_in_bounds(rounded):
                final_path.append(tuple(rounded))
            else:
                # Fallback to nearest valid point
                clamped = np.clip(rounded, 0, np.array(self.grid_shape) - 1)
                final_path.append(tuple(clamped))

        return final_path