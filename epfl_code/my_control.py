# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import cv2

# Global variables
on_ground = True
height_desired = 1.0
timer = None
startpos = None
start_time = None
timer_done = None
max_yaw = 0.5 # rad/s
max_vel = 0.1 # m/s
max_val = 25
visitedNodes = []
ax_astar = None
mode = "Takeoff" # Tell the drone if it is in take off modce, A* path following mode, or radar scanning mode.
path = None

def list_of_coords(max_val):
    x,y = np.mgrid[0:max_val:1, 0:max_val:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])
    return pos, coords

pos, coords = list_of_coords(max_val)

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py lines 296-323. 
# The "item" values that you can later use in the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# "range_down": Downward range finder distance (Used instead of Global Z distance)
# "range_front": Front range finder distance
# "range_left": Leftward range finder distance 
# "range_right": Rightward range finder distance
# "range_back": Backward range finder distance
# "yaw": Yaw angle (rad)

# This is the main function where you will implement your control algorithm
def get_command(sensor_data, camera_data, dt):
    global on_ground, startpos, start_time, max_val, pos, coords, visitedNodes, t, mode, path

    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)
    
    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down']]    
        start_time = time.time()
        print("Start Time: ", start_time)
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    # ---- YOUR CODE HERE ----
    control_command = [0.0, 0.0, height_desired, 0.0]
    on_ground = False
    map = occupancy_map(sensor_data)
    max_val = map.shape[0] 
    
    # If the occupancy map has some free space, run the A* algorithm
    if time.time() - start_time > 5:
        # Occupancy grid is +1 for free, -1 for occupied and 0 for unknown
        astar_map = convert_to_astar_grid(map)
        # print(f'Map: {astar_map}')

        # Define the heuristic, here = distance to goal ignoring obstacles
        start = [int(np.round(sensor_data['x_global'] * 5.0)), int(np.round(sensor_data['y_global'] * 5.0))]
        astar_map[start[0], start[1]] = 0 # We know the start is free
        goal = find_goal_location(astar_map)
        h = np.linalg.norm(pos - goal, axis=-1)
        h = dict(zip(coords, h))

        # Run the A* algorithm
        path, visitedNodes = A_Star(tuple(start), tuple(goal), h, coords, astar_map, movement_type="4N")
        path = np.array(path).reshape(-1, 2).transpose()
        visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()

        # Save the plot
        save_astar_grid_plot(astar_map, visitedNodes, path, start, goal, t, max_val)
        mode = "AStar"

    # If the path is not empty, move the drone to the next point in the path
    if path is not None and mode == "AStar":
        # Check if arrived at each point in the path
        if has_reached_goal([sensor_data['x_global'], sensor_data['y_global']], [path[0][0]/5.0, path[0][1]/5.0]):
            if len(path) > 1:
                path = path[1:] # Get rid of first point in path
            else: # If the path is empty, stop the drone
                path = []
                mode = "Radar Scan"
                print("Path completed...Scanning for obstacles")
                control_command = [0.0, 0.0, height_desired, 0.0]
                return control_command
        
        # Calculate distance and angle to next position
        next_point = path[0]
        dx = next_point[0] - sensor_data['x_global']
        dy = next_point[1] - sensor_data['y_global']
        yaw_desired = np.arctan2(dy, dx)
        distance = np.linalg.norm([dx, dy])

        # Calculate velocity forward and left commands (assuming the drone always faces forward)
        vel_fwd_cmd = min(distance, max_vel)

        control_command = [vel_fwd_cmd, 0.0, height_desired, yaw_desired]
        return control_command

    # print(map)

    #print(sensor_data) 
    # Sensor Data Dictionary: {'t': 5.0, 'x_global': 0.8, 'y_global': 2.1, 'z_global': 1.2, 'roll': 0.1, 'pitch': -3.7e-05, 'yaw': 2.3,
    #                         'q_x': 3.7e-05, 'q_y': 4.0e-05, 'q_z': 0.9, 'q_w': 0.4, 'v_x': 0.0, 'v_y': 0.0, 'v_z': -0.0, 
    #                         'v_forward': -0.0, 'v_left': -0.0, 'v_down': -0.04, 'ax_global': 0.7, 'ay_global': 0.03, 'az_global': 0.1, 
    #                         'range_front': 2.0, 'range_left': 2.0, 'range_back': 0.2, 'range_right': 2.0, 'range_down': 1.2, 
    #                         'rate_roll': -0.0, 'rate_pitch': 0.0, 'rate_yaw': 0.9}
    # Camera Data: 3D numpy array with shape (height, width, 3) representing the camera image

    # Wrap code in a switch statement to select the desired control algorithm based on X position
    # if sensor_data['x_global'] < 3.5:
    #     # Goal: Move in a positive X direction while avoiding obstacles.
        
    #     move_towards_landing_region(sensor_data, map)





    # elif sensor_data['x_global'] >= 3.5:
    #     # Goal: Search for landing pad while avoiding obstacles.
    #     control_command = [0.0, 0.0, height_desired, 0.0]



    # ------------------------


    
    return control_command # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]

def move_towards_landing_region(sensor_data, map):
    # Move towards the landing region while avoiding obstacles.

    """
    Idea: We want to move in the direction of positive X and centered Y, we also need to prevent the drone from hitting obstables 
    its sensors do not see because we are using laser sensors with no FoV. Therefore, to prevent collision, the drone must first 
    scan a 90 degree cone in front of it to detect obstacles and add them to the occupancy map. The drone will then move in the
    positive x direction while maintaining a redius of 0.5m from obstacles and uncertain areas in the occupancy map. When it 
    reaches an uncertain zone, the drone should then scan a 90 degree cone again and repeat this cycle until we reach the landing region.
    

    # TODO: 
    # Do all these:
    # - orient_in_pos_X() # 45 degrees off center
    # - scan_90_degrees() # scan a 90 degree cone in front of the drone
    # - stop_rotation()   # stop the drone from rotating
    # - check_occupancy_map() # returns a boolean of whether the drone is near an uncertain zone it needs to scan.
   
    # Then Choose one of these: (wrapped in choose_action() function)
    # - move_in_pos_x()
    # - move_around_obstacle()

    # """

    # if check_occupancy_map(sensor_data, map):
    #     scan_90_degrees(sensor_data)
    #     stop_rotation(sensor_data)
    # else:
    #     choose_action(sensor_data, map)


    # control_command = [0.0, 0.0, height_desired, 0.0]
    # return control_command
    return

def orient_towards_goal(robot_pos, goal_pos):
    # Orient the drone towards the goal position
    dx = goal_pos[0] - robot_pos[0]
    dy = goal_pos[1] - robot_pos[1]
    yaw_desired = np.arctan2(dy, dx)
    control_command = [0.0, 0.0, height_desired, yaw_desired]
    return control_command  # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]

def orient_in_pos_X(sensor_data):
    # Orient the drone in the positive X direction. Yaw is the angle of the drone in the global frame, therefore we can use it for orientation.
    control_command = [0.0, 0.0, height_desired, max_yaw]
    return control_command  # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]


# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 5.0 # meter
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.2 # meter
conf = 0.2 # certainty given by each measurement
t = 0 # only for plotting

map = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied

def occupancy_map(sensor_data):
    global map, t
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    yaw = sensor_data['yaw']
    
    for j in range(4): # 4 sensors
        yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']
        
        for i in range(int(range_max/res_pos)): # range is 2 meters
            dist = i*res_pos
            idx_x = int(np.round((pos_x - min_x + dist*np.cos(yaw_sensor))/res_pos,0))
            idx_y = int(np.round((pos_y - min_y + dist*np.sin(yaw_sensor))/res_pos,0))

            # make sure the current_setpoint is within the map
            if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > range_max:
                break

            # update the map
            if dist < measurement:
                map[idx_x, idx_y] += conf
            else:
                map[idx_x, idx_y] -= conf
                break
    
    map = np.clip(map, -1, 1) # certainty can never be more than 100%

    # only plot every Nth time step (comment out if not needed)
    if t % 50 == 0:
        plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        plt.savefig("map.png")
        plt.close()
    t +=1

    return map


# Control from the exercises
index_current_setpoint = 0
def path_to_setpoint(path,sensor_data,dt):
    global on_ground, height_desired, index_current_setpoint, timer, timer_done, startpos

    # Take off
    if startpos is None:
        startpos = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down']]    
    if on_ground and sensor_data['range_down'] < 0.49:
        current_setpoint = [startpos[0], startpos[1], height_desired, 0.0]
        return current_setpoint
    else:
        on_ground = False

    # Start timer
    if (index_current_setpoint == 1) & (timer is None):
        timer = 0
        print("Time recording started")
    if timer is not None:
        timer += dt
    # Hover at the final setpoint
    if index_current_setpoint == len(path):
        # Uncomment for KF
        control_command = [startpos[0], startpos[1], startpos[2]-0.05, 0.0]

        if timer_done is None:
            timer_done = True
            print("Path planing took " + str(np.round(timer,1)) + " [s]")
        return control_command

    # Get the goal position and drone position
    current_setpoint = path[index_current_setpoint]
    x_drone, y_drone, z_drone, yaw_drone = sensor_data['x_global'], sensor_data['y_global'], sensor_data['range_down'], sensor_data['yaw']
    distance_drone_to_goal = np.linalg.norm([current_setpoint[0] - x_drone, current_setpoint[1] - y_drone, current_setpoint[2] - z_drone, clip_angle(current_setpoint[3]) - clip_angle(yaw_drone)])

    # When the drone reaches the goal setpoint, e.g., distance < 0.1m
    if distance_drone_to_goal < 0.1:
        # Select the next setpoint as the goal position
        index_current_setpoint += 1
        # Hover at the final setpoint
        if index_current_setpoint == len(path):
            current_setpoint = [0.0, 0.0, height_desired, 0.0]
            return current_setpoint

    return current_setpoint

def clip_angle(angle):
    angle = angle%(2*np.pi)
    if angle > np.pi:
        angle -= 2*np.pi
    if angle < -np.pi:
        angle += 2*np.pi
    return angle

def _get_movements_4n():
    """
    Get all possible 4-connectivity movements (up, down, left right).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]

def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1)
    (up, down, left, right and the 4 diagonals).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = np.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]

def reconstruct_path(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately 
                     preceding it on the cheapest path from start to n 
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        total_path.insert(0, cameFrom[current]) 
        current=cameFrom[current]
    return total_path

def A_Star(start, goal, h, coords, occupancy_grid, movement_type="4N", max_val=map.shape[0]):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal_m: goal node (x, y)
    :param occupancy_grid: the grid map
    :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """
    
    # -----------------------------------------
    # DO NOT EDIT THIS PORTION OF CODE
    # -----------------------------------------
    
    # Check if the start and goal are within the boundaries of the map
    for point in [start, goal]:
        for coord in point:
            assert coord>=0 and coord<max_val, "start or end goal not contained in the map"
    
    # check if start and goal nodes correspond to free spaces
    if occupancy_grid[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if occupancy_grid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')
    
    # get the possible movements corresponding to the selected connectivity
    if movement_type == '4N':
        movements = _get_movements_4n()
    elif movement_type == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')
    
    # --------------------------------------------------------------------------------------------
    # A* Algorithm implementation - feel free to change the structure / use another pseudo-code
    # --------------------------------------------------------------------------------------------
    
    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [start]
    
    # The set of visited nodes that no longer need to be expanded.
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    # while there are still elements to investigate
    while openSet != []:
        
        #the node in openSet having the lowest fScore[] value
        fScore_openSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet
        
        #If the goal is reached, reconstruct and return the obtained path
        if current == goal:
            return reconstruct_path(cameFrom, current), closedSet

        openSet.remove(current)
        closedSet.append(current)
        
        #for each neighbor of current:
        for dx, dy, deltacost in movements:
            
            neighbor = (current[0]+dx, current[1]+dy)
            
            # if the node is not in the map, skip
            if (neighbor[0] >= occupancy_grid.shape[0]) or (neighbor[1] >= occupancy_grid.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                continue
            
            # if the node is occupied or has already been visited, skip
            if (occupancy_grid[neighbor[0], neighbor[1]]) or (neighbor in closedSet): 
                continue
                
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[current] + deltacost
            
            if neighbor not in openSet:
                openSet.append(neighbor)
                
            if tentative_gScore < gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return [], closedSet

# Define the start and end goal
# start = (0,0)
# goal = (43,33)

def convert_to_astar_grid(map):
    """
    Occupancy grid is +1 for free, -1 for occupied and 0 for unknown
    This function convers this occupancy grid to one that can be used by the A* algorithm
    where all occupied and unknonw cells are set to 1 and all free cells are set to 0.
    """
    astar_grid = np.zeros(map.shape)
    astar_grid[map != 1] = 0
    return astar_grid

def find_goal_location(astar_map):
   """
   This function finds the goal location in the A* grid by finding the cell with the maximum X value, 
   and then, if there is more than 1 space with the same X value, find the cell with the most average Y value of the found cells.
   """
   # Find cells with value 0
   zero_indices = np.where(astar_map == 0)
   #print(zero_indices)
   
   # Find cell indices with maximum X value as array of tuples where he first row is the X value and the second row is the Y value
   max_x = np.where(zero_indices[0] == np.max(zero_indices[0]))
   #print(max_x)

   # If there is only one cell with maximum X value, return it
   if len(max_x[0]) == 1:
       goal = np.array([zero_indices[0][max_x][0],zero_indices[1][max_x][0]])
       return goal
   else:
       # Of the cells with maximum X value and find the one with the most average Y index
       middle_y_val = astar_map.shape[1] / 2
       max_x_indices = zero_indices[1][max_x]
       distance_from_middle = np.abs(max_x_indices - middle_y_val)
       most_avg_y_indices = np.argmin(distance_from_middle)
       goal = np.array([zero_indices[0][max_x][most_avg_y_indices], zero_indices[1][max_x][most_avg_y_indices]])
       return goal

def create_empty_plot(max_val):
    """
    Helper function to create a figure of the desired dimensions & grid
    
    :param max_val: dimension of the map along the x and y dimensions
    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=(7,7))
    
    major_ticks = np.arange(0, max_val+1, 5)
    minor_ticks = np.arange(0, max_val+1, 1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([-1,max_val])
    ax.set_xlim([-1,max_val])
    ax.grid(True)
    
    return fig, ax

def save_astar_grid_plot(astar_map, visitedNodes, path, start, goal, t, max_val):
    global ax_astar
    
    if not ax_astar:
        # Create the empty plot to fill in
        fig_astar, ax_astar = create_empty_plot(max_val)
    
    if t % 50 == 0:
        # Save the plot as a PNG file every 50 iterations
        plt.savefig(f"astar.png")
        plt.close()  # Close the plot after saving

    # Clear the previous path data
    ax_astar.clear()

    # Displaying the map
    cmap = colors.ListedColormap(['white', 'red'])  # Select the colors for obstacles and free cells
    ax_astar.imshow(astar_map.transpose(), cmap=cmap)

    # Plot the best path found and the list of visited nodes
    ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color='orange')
    ax_astar.plot(path[0], path[1], marker="o", color='blue')
    ax_astar.scatter(start[0], start[1], marker="o", color='green', s=200)
    ax_astar.scatter(goal[0], goal[1], marker="o", color='purple', s=200)

    #plt.show()  # Display the plot

def has_reached_goal(current_position, goal_position, threshold_distance=0.1):
    """
    Check if the drone has reached close enough to the goal position.

    :param current_position: Current position of the drone (x, y, z).
    :param goal_position: Goal position to reach (x, y, z).
    :param threshold_distance: Threshold distance to consider the goal reached.
    :return: True if the drone has reached close enough to the goal, False otherwise.
    """
    distance_to_goal = np.linalg.norm(np.array(goal_position[:2]) - np.array(current_position[:2]))
    return distance_to_goal <= threshold_distance


if __name__ == '__main__':
    astar_map = np.array([[1,1,1,1,1,1,1,1,1,0],
                          [1,1,1,1,1,1,1,0,0,1],
                          [1,1,1,1,1,0,1,0,1,1],
                          [1,1,0,1,1,0,1,1,1,1],
                          [1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,1,1,1,1,1,1,1],
                          [0,1,1,0,1,1,0,0,1,1],
                          [0,0,1,0,1,1,1,0,0,1]])
    goal_location = find_goal_location(astar_map)
    #print(f'goal_location: {goal_location}')
    #print(goal_location)

                         
        