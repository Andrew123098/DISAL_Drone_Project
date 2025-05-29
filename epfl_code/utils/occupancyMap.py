import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go


class OccupancyMap:
    def __init__(self, world_size: Tuple[float, float, float] = (10, 10, 5), resolution: float = 0.05):
        """
        Initialize the occupancy map generator.

        Parameters:
        - world_size: The physical dimensions of the space in meters (X, Y, Z).
        - resolution: Size of each voxel cell in meters.
        """
        self.world_size = world_size
        self.resolution = resolution
        self.grid_dims = (np.array(world_size) / resolution).astype(int)
        self.occupancy_grid = np.zeros(self.grid_dims, dtype=np.int8)
        self.control_points = []

    def create(self, objects: List[Dict], control_points: List[tuple]) -> np.ndarray:
        """
        Populate the occupancy grid with labeled voxels based on objects and control points.

        Parameters:
        - objects: List of object dictionaries containing translation, rotation, scale, and type.
        - control_points: List of (x,y,z) tuples representing path control points.

        Returns:
        - occupancy_grid: A 3D numpy array representing the labeled space with:
            - 1 = obstacles (beams/takeoff pads)
            - 0 = free space
            - -1 = gates
            - -2 = control points
            - -3 = path waypoints
            - -4 = flight area
        """
        print(f"Creating {self.world_size}m world grid: {self.grid_dims} cells")

        # Mark central flight area (-4)
        flight_cells = int(8 / self.resolution)
        start = int(1 / self.resolution)  # 2m offset (1m in from each wall)
        self.occupancy_grid[start:start + flight_cells,
        start:start + flight_cells,
        0] = -4

        # Process all objects
        for obj in objects:
            pos = np.array(obj['translation'])
            size = np.array(obj['scale'])
            rot = Rotation.from_rotvec(obj['rotation'][3] * np.array(obj['rotation'][:3]))

            half_size = size / 2

            # Create voxel ranges
            x_range = np.arange(
                max(0, pos[0] - half_size[0]),
                min(self.world_size[0], pos[0] + half_size[0]),
                self.resolution
            )
            y_range = np.arange(
                max(0, pos[1] - half_size[1]),
                min(self.world_size[1], pos[1] + half_size[1]),
                self.resolution
            )
            z_range = np.arange(
                max(0, pos[2] - half_size[2]),
                min(self.world_size[2], pos[2] + half_size[2]),
                self.resolution
            )

            # Grid and transform points
            xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
            rotated_points = rot.apply(points - pos) + pos
            grid_coords = (rotated_points / self.resolution).astype(int)

            valid = np.all((grid_coords >= 0) & (grid_coords < self.grid_dims), axis=1)
            unique_coords = np.unique(grid_coords[valid], axis=0)

            # Assign voxel values based on object type
            if obj['type'] == 'gate':
                value = -1
            elif obj['type'] in ('takeoff_pad', 'beam', 'obstacle'):
                value = 1
            else:
                value = 0

            if unique_coords.size > 0:
                self.occupancy_grid[
                    unique_coords[:, 0],
                    unique_coords[:, 1],
                    unique_coords[:, 2]
                ] = value

        # Mark control points (-2)
        self.control_points = control_points
        for point in control_points:
            x, y, z = point
            # Convert world coordinates to grid indices
            xi = int(x / self.resolution)
            yi = int(y / self.resolution)
            zi = int(z / self.resolution)

            # Only mark if within bounds
            if (0 <= xi < self.grid_dims[0] and
                    0 <= yi < self.grid_dims[1] and
                    0 <= zi < self.grid_dims[2]):
                self.occupancy_grid[xi, yi, zi] = -2
            else:
                print(f"Warning: Control point {point} outside grid bounds")

        return self.occupancy_grid

    def plot(self, grid = None):
        """
        Visualize the occupancy grid using Plotly in 3D.
        """
        fig = go.Figure()
        resolution = self.resolution

        if grid is None:
            grid = self.occupancy_grid


        def add_trace(condition_value, color, name, size=2, opacity=0.6, stride=1):
            coords = np.where(grid == condition_value)

            if len(coords[0]) > 0:
                fig.add_trace(go.Scatter3d(
                    x=coords[0][::stride] * resolution,
                    y=coords[1][::stride] * resolution,
                    z=coords[2][::stride] * resolution,
                    mode='markers',
                    marker=dict(size=size, color=color, opacity=opacity),
                    name=name
                ))

        # Add object types
        add_trace(-4, 'gray', 'Flight Area (-4)', size=1, opacity=0.2)
        add_trace(-1, 'red', 'Gates (-1)', size=3, opacity=0.02, stride=max(1, int(0.05 / resolution)))
        add_trace(1, 'blue', 'Takeoff Pad / Beams (1)', size=1, opacity=0.7)
        add_trace(-3, 'pink', 'A* Path (-3)', size=2, opacity=0.7)
        add_trace(-5, 'orange', 'Smoothed Path (-5)', size=3, opacity=0.9)
        add_trace(-2, 'green', 'Control Points', size=4, opacity=1)


        # Add boundary outline
        boundary_x = [1, 9, 9, 1, 1]
        boundary_y = [1, 1, 9, 9, 1]
        boundary_z = [0] * 5
        fig.add_trace(go.Scatter3d(
            x=boundary_x,
            y=boundary_y,
            z=boundary_z,
            mode='lines',
            line=dict(color='black', width=2),
            name='Flight Boundary'
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, 10], title='X (m)'),
                yaxis=dict(range=[0, 10], title='Y (m)'),
                zaxis=dict(range=[0, 5], title='Z (m)'),
                aspectmode='manual',
                aspectratio=dict(x=2, y=2, z=1)
            ),
            width=900,
            height=700,
            title="3D Occupancy Grid Map",
            showlegend=True,
        )

        fig.show()
        return fig

    def add_new_object(self, type: str, object: List[tuple], grid: np.ndarray = None) -> np.ndarray:
        """
        Update grid cells at specified coordinates with a given value.

        Parameters:
        - type: String ("Free", "Obstacle", "Gate", "Path", or "Control Point")
        - object: List of (i,j,k) grid coordinate tuples
        - grid: Optional grid to modify (defaults to self.occupancy_grid)

        Returns:
        - Modified grid

        Raises:
        - ValueError: If invalid type or out-of-bounds coordinates
        """
        # Type to value mapping
        type_values = {
            "Free": 0,
            "Obstacle": 1,
            "Gate": -1,
            "Control Point": -2,
            "Path": -3,
            "Smoothed Path": -5,
            "Flight Area": -4,
        }

        # Validate type
        if type not in type_values:
            raise ValueError(f"Invalid type '{type}'. Must be one of: {list(type_values.keys())}")

        value = type_values[type]

        # Use provided grid or default
        target_grid = self.occupancy_grid if grid is None else grid

        # Update each coordinate
        for coord in object:
            if len(coord) != 3:
                raise ValueError(f"Coordinate {coord} must be a 3-tuple (i,j,k)")

            i, j, k = coord

            # Bounds checking
            if not (0 <= i < self.grid_dims[0] and
                    0 <= j < self.grid_dims[1] and
                    0 <= k < self.grid_dims[2]):
                raise ValueError(f"Coordinate {coord} out of grid bounds {self.grid_dims}")

            target_grid[i, j, k] = value

        # Update self reference if using default grid
        if grid is None:
            self.occupancy_grid = target_grid

        return target_grid

    def clear_objects(self, object_type: str, grid: np.ndarray = None) -> np.ndarray:
        """
        Clear all cells of a specific object type by setting them to free space (0).

        Parameters:
        - object_type: String type to clear ("Obstacle", "Gate", "Path", etc.)
        - grid: Optional grid to modify (defaults to self.occupancy_grid)

        Returns:
        - Modified grid

        Raises:
        - ValueError: If invalid object type
        """
        # Type to value mapping (must match add_new_object)
        type_values = {
            "Free": 0,
            "Obstacle": 1,
            "Gate": -1,
            "Control Point": -2,
            "Path": -3,
            "Smoothed Path": -5,
            "Flight Area": -4,
        }

        # Validate type
        if object_type not in type_values:
            raise ValueError(f"Invalid type '{object_type}'. Must be one of: {list(type_values.keys())}")

        # Get the value we're looking to clear
        target_value = type_values[object_type]

        # Use provided grid or default
        target_grid = self.occupancy_grid if grid is None else grid

        # Find and clear all matching cells
        target_grid[target_grid == target_value] = 0

        # Update self reference if using default grid
        if grid is None:
            self.occupancy_grid = target_grid

        return target_grid


    def world_to_grid(self, position: List[float] | List[List[float]]) -> tuple[int, ...] | List[tuple[int, ...]]:
        """Convert continuous world coordinates to discrete grid indices.

        Parameters:
            position: Either a single point [x, y, z] or a list of points [[x, y, z], ...]
                 in world coordinates (meters)

        Returns:
        - For single point: tuple of grid indices (i, j, k)
        - For multiple points: list of tuples of grid indices [(i, j, k), ...]
        """
        # Handle single point
        if not isinstance(position[0], (list, tuple)):
            return tuple(int(round(p / self.resolution)) for p in position)
        
        # Handle list of points
        return [tuple(int(round(p / self.resolution)) for p in point) for point in position]

    def grid_to_world(self, grid_coords):
        """Convert discrete grid indices back to continuous world coordinates.

        Parameters:
            grid_coords: Grid coordinates to convert. Can be a single point or sequence of points.

        Returns:
            - For single point: List of world coordinates [x, y, z] in meters
            - For multiple points: List of world coordinate points
        """
        # Calculate number of decimal places based on resolution
        decimal_places = abs(int(np.floor(np.log10(self.resolution))))

        # Handle single point
        if len(grid_coords) > 0 and not isinstance(grid_coords[0], (tuple, list)):
            return [round(coord * self.resolution, decimal_places) for coord in grid_coords]

        # Handle sequence of points
        return [[round(coord * self.resolution, decimal_places) for coord in point] for point in grid_coords]

    def world_to_sim(self, path: list[tuple]) -> list[tuple]:
        """Convert plotly world coordinates to simulation coordinates.

        Args:
            path: List of tuples containing (x, y, z) world coordinates

        Returns:
            List of tuples containing (x, y, z) simulation coordinates
            with x and y offset by -1, rounded to the same precision as grid_to_world
        """
        decimal_places = abs(int(np.floor(np.log10(self.resolution))))
        return [(round(x - 1, decimal_places), round(y - 1, decimal_places), z) for x, y, z in path]

    def get_triplets(self, control_points, gate_angles, distance, flip_triplets=None):
        """Create and optionally flip triplets for each control point."""
        if flip_triplets is None:
            flip_triplets = [1, 1, 1, 1, 1, 1]  # Default flip pattern

        triplets = []
        for i, (center, theta) in enumerate(zip(control_points, gate_angles)):
            x, y, z = center
            if theta != 0:  # Only create full triplets for gates with angles
                dx = distance * np.cos(theta)
                dy = distance * np.sin(theta)

                left = (x + dx, y + dy, z)
                right = (x - dx, y - dy, z)

                if flip_triplets[i]:
                    left, right = right, left

                triplets.append([left, center, right])
            else:
                triplets.append([center])  # Single point for non-gates
        return triplets

    def order_control_points(self, control_points, num_loops, gate_angles=None, distance=0.5):
        """Order control points while maintaining triplet grouping."""
        if len(control_points) != 6:
            print(f"Warning: Expected 6 control points, got {len(control_points)}")
            return control_points

        gate_order = [5, 0, 1, 2, 3, 4]  # Special gate ordering

        # Create triplets with flipping
        triplets = self.get_triplets(
            control_points,
            gate_angles or [0] * 6,
            distance
        )

        # Reorder and print verification
        # print("\nGate Processing Order:")
        ordered_triplets = []

        for new_pos, orig_idx in enumerate(gate_order):
            triplet = triplets[orig_idx]
            ordered_triplets.append(triplet)

            # # Print verification
            # print(f"\nGate {orig_idx} -> Position {new_pos}:")
            # for i, point in enumerate(triplet):
            #     label = ["Left", "Center", "Right"][i] if len(triplet) == 3 else "Single"
            #     print(f"{label}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")

        # Add to occupancy grid
        for triplet in ordered_triplets:
            for point in triplet:
                try:
                    self.add_new_object("Control Point", [self.world_to_grid(point)])
                except ValueError as e:
                    print(f"Warning: Couldn't add point {point}: {e}")

        # Flatten and handle looping
        ordered_points = [p for triplet in ordered_triplets for p in triplet]

        if num_loops == 1:
            print("Length of Control Points", len(ordered_points))
            return ordered_points[1:] + [ordered_points[0]]
        elif num_loops == 2:
            return ordered_points[1:] + [ordered_points[0]] + ordered_points[1:] + [ordered_points[0]]
        else:
            result = ordered_points[1:] + ordered_points
            for _ in range(num_loops - 2):
                result += ordered_points
            return result


