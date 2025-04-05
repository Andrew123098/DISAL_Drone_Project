import numpy as np
import matplotlib.pyplot as plt
import cv2

class Visualize:
    def __init__(self):
        # Window Size
        self.window_size = 500

        # Random occupancy map parameters
        self.min_x, self.max_x = 0, 8  # meter
        self.min_y, self.max_y = 0, 8  # meter
        self.res_pos = 0.1  # meter
        self.t = 0  # only for plotting
        
        # Random Start and goal positions with np.random
        self.start = (np.random.randint(0, self.max_x/self.res_pos), np.random.randint(0, self.max_x/self.res_pos))
        self.goal = (np.random.randint(0, self.max_x/self.res_pos), np.random.randint(0, self.max_x/self.res_pos))

    def occupancy_map(self):
        """
        :return: occupancy map
        """
        # Create a grid of positions
        x = np.arange(self.min_x, self.max_x, self.res_pos)
        y = np.arange(self.min_y, self.max_y, self.res_pos)
        X, Y = np.meshgrid(x, y)

        # Create an occupancy map
        map = np.zeros(X.shape)

        # Add obstacles to the map
        map = self.add_obstacles(map, X, Y, self.t)

        # Add a perimeter of obstacles around the map
        map = self.add_perimeter_obstacles(map)

        # Plot the start and goal positions
        map = self.plot_start_goal(map, self.start, self.goal)

        return map

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
        map3d[map == 3] = [0, 255, 255]  # yellow

        return map3d

    def add_obstacles(self, map, X, Y, t):
        """
        :param map: occupancy map
        :param X: x positions
        :param Y: y positions
        :param t: time step
        :return: occupancy map with obstacles
        """
        # Add random obstacles to the map with 1 pixel by giving every pixel a 1/3 chance of being an obstacle
        map += np.random.choice([0, 1], size=map.shape, p=[9/10, 1/10])

        return map

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
        # Find random start and goal positions and define them as -1 and -2 respectively
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


if __name__ == "__main__":
    visualizer = Visualize()
    occupancy_map = visualizer.occupancy_map()
    visualizer.plot_map(occupancy_map)