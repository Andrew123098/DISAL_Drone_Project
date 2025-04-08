import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional, Dict


class Visualize:
    def __init__(self, size=(12, 12), resolution=0.1, start_pos = (0,0), goal_pos= (5,5)):
        # Window Size
        self.window_size = 500

        # Random occupancy map parameters
        self.min_x, self.max_x = 0, size[0]  # meter
        self.min_y, self.max_y = 0, size[1]  # meter
        self.res_pos = resolution  # meter
        self.t = 0  # only for plotting
        self.map = None  # occupancy map

        # Random Start and goal positions with np.random
        self.start = start_pos
        self.goal = goal_pos

    def occupancy_map(self):
        """
        :return: occupancy map
        """
        # Create a grid of positions
        x = np.arange(self.min_x, self.max_x, self.res_pos)
        y = np.arange(self.min_y, self.max_y, self.res_pos)
        X, Y = np.meshgrid(x, y)

        # Create an occupancy map
        self.map = np.zeros(X.shape)

        # Add obstacles to the map
        self.map = self.add_obstacles(self.map, X, Y, self.t)

        # Add a perimeter of obstacles around the map
        self.map = self.add_perimeter_obstacles(self.map)

        # Plot the start and goal positions
        self.map = self.plot_start_goal(self.map, self.start, self.goal)

        map = self.map.copy()

        return map
    
    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid coordinates"""
        grid_center = (self.map.shape[0] / 2, self.map.shape[1] / 2)
        grid_x = int(round(grid_center[1] + world_pos[0] / self.res_pos))
        grid_y = int(round(grid_center[0] - world_pos[1] / self.res_pos))  # Invert Y axis
        
        # Ensure coordinates are within bounds
        grid_x = max(0, min(grid_x, self.map.shape[1] - 1))
        grid_y = max(0, min(grid_y, self.map.shape[0] - 1))
        
        return (grid_y, grid_x)  # Return as (row, col)

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates back to world coordinates"""
        grid_center = (self.map.shape[0] / 2, self.map.shape[1] / 2)
        row, col = grid_pos
        world_x = (col - grid_center[1]) * self.res_pos
        world_y = (grid_center[0] - row) * self.res_pos  # Invert Y axis
        return (world_x, world_y)

    def colorize_map(self, map):
        """
        :param map: occupancy map
        :return: colorized map
        """
        map3d = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)

        # Give each pixel an RGB value based on the occupancy value
        map3d[map == 1] = [255, 255, 255]  # white
        map3d[map == 0] = [0, 0, 0]  # black
        map3d[map == -1] = [0, 0, 255]  # red
        map3d[map == -2] = [0, 255, 0]  # green
        map3d[map == 2] = [255, 255, 0]  # blue
        map3d[map == 3] = [255, 0, 0] # yellow
        map3d[map == 4] = [100, 100, 100] # grey

        return map3d

    def add_obstacles(self, map, X, Y, t):
        """
        :param map: occupancy map
        :param X: x positions
        :param Y: y positions
        :param t: time step
        :return: occupancy map with obstacles (1 for core, 2 for boundary)
        """
        # First create 3x3 obstacles (core)
        obstacle_chance = 1/200
        # Create initial random obstacles (single pixels)
        initial_obstacles = np.random.choice([0, 1], size=map.shape, p=[1-obstacle_chance, obstacle_chance])
        
        # Create 3x3 core obstacles (marked as 1)
        core_map = np.zeros_like(map)
        core_offsets = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]
        for dx, dy in core_offsets:
            rolled = np.roll(np.roll(initial_obstacles, dx, axis=0), dy, axis=1)
            core_map[rolled == 1] = 1
        
        # Now create 5x5 boundary (marked as 2) around the core
        boundary_map = np.zeros_like(map)
        # Generate all offsets for 5x5 square
        boundary_offsets = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]
        
        # First mark all 5x5 areas (both core and boundary)
        for dx, dy in boundary_offsets:
            rolled = np.roll(np.roll(core_map, dx, axis=0), dy, axis=1)
            boundary_map[rolled == 1] = 1
        
        # Now subtract the core to get just the boundary
        boundary_only = np.where((boundary_map == 1) & (core_map != 1), 4, 0)
        
        # Combine the maps (core stays 1, boundary becomes 2)
        final_map = np.where(core_map == 1, 1, 
                            np.where(boundary_only == 4, 4, map))
        
        return final_map

    def add_perimeter_obstacles(self, map):
        """
        :param map: occupancy map
        :return: occupancy map with perimeter obstacles
        """
        # Add a perimeter of obstacles around the map
        map[0, :] = 1
        map[-1, :] = 1
        map[:, 0] = 1
        map[:, -1] = 1

        return map

    def plot_start_goal(self, map, start, goal):
        """
        :param map: occupancy map
        :param start: start position
        :param goal: goal position
        :return: None
        """
        
        # Convert and mark positions
        start = self._world_to_grid(start)
        goal = self._world_to_grid(goal)


        # Find start and goal positions and define them as -1 and -2 respectively
        map[start[0], start[1]] = -1
        map[goal[0], goal[1]] = -2

        return map
    
    def upscale_map(self, map):
        """
        Upscale the map to a fixed display size regardless of resolution.
        :param map: The occupancy map to upscale
        :return: Upscaled map
        """
        # Define fixed display size (e.g., 800x800 pixels)
        display_size = (self.window_size, self.window_size)

        # Resize the map to the fixed display size using interpolation
        upscaled_map = cv2.resize(map, display_size, interpolation=cv2.INTER_NEAREST)

        return upscaled_map
    

    def plot_map(self, map):
        """
        :param map: occupancy map
        :return: None
        """
        # Upscale the map for better visualization
        img = self.upscale_map(map)
        # img = np.kron(map, np.ones((8, 8)))

        # Colorize the map
        img = self.colorize_map(img)
        
        cv2.imshow("Occupancy Map", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def draw_path(self, map_img: np.ndarray, path: List[Tuple[float, float]], 
                color: Tuple[int, int, int], thickness: int = 2, 
                as_dots: bool = True, dot_size: int = 3) -> np.ndarray:
        """
        Draw a path on the map image with proper coordinate transformation
        Args:
            map_img: Colorized map image (3-channel)
            path: List of (x,y) world coordinates
            color: BGR tuple
            thickness: Line thickness (for continuous lines)
            as_dots: If True, draw as individual dots instead of lines
            dot_size: Radius of dots when as_dots=True
        Returns:
            Map image with path drawn
        """
        h, w = map_img.shape[:2]
        
        # Calculate scaling factors
        x_scale = w / (self.max_x - self.min_x)
        y_scale = h / (self.max_y - self.min_y)
        
        points = []
        for x, y in path:
            # Convert world coordinates to image coordinates:
            # 1. Shift to make (0,0) at map center
            centered_x = x + (self.max_x - self.min_x)/2
            centered_y = y + (self.max_y - self.min_y)/2
            
            # 2. Scale to image dimensions
            img_x = int(centered_x * x_scale)
            
            # 3. Flip y-axis (world y increases upwards, image y increases downwards)
            img_y = h - int(centered_y * y_scale)
            
            # 4. Clamp to image bounds
            img_x = max(0, min(img_x, w-1))
            img_y = max(0, min(img_y, h-1))
            
            points.append((img_x, img_y))
        
        if as_dots:
            # Draw as individual dots
            for pt in points:
                cv2.circle(map_img, pt, dot_size, color, -1)  # Filled circle
        else:
            # Draw lines between points
            for i in range(len(points)-1):
                cv2.line(map_img, points[i], points[i+1], color, thickness)
        
        return map_img

    def plot_map_with_paths(self, map, astar_path=None, spline_path=None, gradient_path=None):
        """
        Plot map with multiple path options
        Args:
            map: Occupancy grid
            astar_path: Raw A* path (grid coordinates)
            spline_path: Smoothed spline path (world coordinates)
            gradient_path: Smoothed gradient path (world coordinates)
        """
        # Create colorized map
        img = self.colorize_map(self.upscale_map(map))
        
        # Draw paths if provided
        if astar_path:
            world_path = [self._grid_to_world(p) for p in astar_path]
            img = self.draw_path(img, world_path, (255, 255, 0))  # Yellow for raw path
        
        if spline_path:
            # Convert to numpy array for analysis
            spline_pts = np.array(spline_path)
            
            # Calculate point density
            distances = np.sqrt(np.sum(np.diff(spline_pts, axis=0)**2, axis=1))
            avg_segment_length = np.mean(distances)
            
            # Color dots by local density (red = dense, blue = sparse)
            colors = plt.cm.viridis(distances / np.max(distances))
            
            # Plot with color-coded dots and a black background
            plt.figure(figsize=(10, 10))

            # Set the entire figure background to black
            plt.gcf().patch.set_facecolor('black')

            # Scatter plot with color-coded dots
            plt.scatter(spline_pts[1:, 0], spline_pts[1:, 1], 
                        c=colors, s=10, cmap='viridis_r')

            # Set the axis background to black
            plt.gca().set_facecolor('black')

            # Customize axis lines and text to be white
            plt.gca().spines['top'].set_color('white')
            plt.gca().spines['bottom'].set_color('white')
            plt.gca().spines['left'].set_color('white')
            plt.gca().spines['right'].set_color('white')
            plt.gca().tick_params(axis='x', colors='white')
            plt.gca().tick_params(axis='y', colors='white')
            plt.gca().xaxis.label.set_color('white')
            plt.gca().yaxis.label.set_color('white')

            # Add colorbar with white text
            cbar = plt.colorbar(label='Segment Length (m)')
            cbar.set_label('Segment Length (m)', fontsize=22)  # Set the font size of the label
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
            cbar.ax.yaxis.label.set_color('white')

            # Set the title with white text
            plt.title("Spline Waypoint With Density", color='white', fontsize=26)

            plt.show()
            
            # Also show in OpenCV for consistency
            img = self.colorize_map(self.upscale_map(map))
            img = self.draw_path(img, spline_path, (0,255,255), 
                                as_dots=True, dot_size=2)
            cv2.imshow("Spline Waypoints (Raw)", img)
        
        if gradient_path:
            img = self.draw_path(img, gradient_path, (255, 0, 255))  # Purple for gradient
        
        # Add legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  # Increase this value to make the text larger
        thickness = 2  # Increase this value to make the text bolder

        if astar_path:
            cv2.putText(img, "Raw Path", (10, 40), font, font_scale, (255, 255, 0), thickness)
        if spline_path:
            cv2.putText(img, "Spline (dots)", (10, 80), font, font_scale, (0, 255, 255), thickness)
        if gradient_path:
            cv2.putText(img, "Gradient", (10, 120), font, font_scale, (255, 0, 255), thickness)
        
        cv2.imshow("Path Planning Comparison", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    visualizer = Visualize()
    occupancy_map = visualizer.occupancy_map()
    visualizer.plot_map(occupancy_map)