# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

# Global variables
STARTING_POSE = [0.5,1.5]
UPDATE_FREQUENCY = 2
on_ground = True
height_desired = 0.5
timer = None
ctrl_timer = None
startpos = None
timer_done = None
mode = 'takeoff' # 'takeoff', 'find goal', 'land'
firstpass_goal = np.array([4.5, 1]) # Location of the first goal
goal = firstpass_goal
canvas = None
fwd_vel_prev = 0
left_vel_prev = 0
yaw_desired = 0.3*180/np.pi
prev_pos = []
num_loops_stuck = 0
k_a = 1.0
k_r = 1.0
permanant_obstacles = 0
first_landpad_location = None
second_landpad_location = None
middle_landpad_location = None
landpad_timer = 0
is_landed = False
is_landed_finale = False
prev_height = 0
num_possible_pads_locations = None
possible_pad_locations = None
list_of_visited_locations = np.empty((0,2), dtype=object)
grid_switcher = 0
grade = 0 # Change Grade to change type of control

path_index = 0
"""
Grade 4.0: Take off, avoid obstacles and reach the landing region whilst being airborne
Grade 4.5: Land on the landing pad
Grade 5.0: Take off from the landing pad and leave the landing region whilst being airborne
Grade 5.25: Avoid obstacles and reach the starting region whilst being airborne
Grade 5.5: Land on the take-off pad
Grade + 0.25: Detect and pass through the pink square during flight from the starting region towards the landing region
Grade + 0.25: Pass through the location of the pink square during flight from the landing region towards the starting region
"""


# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py lines 296-323. 
# The "item" values that you can later use in the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "range_down": Downward range finder distance (Used instead of Global Z distance)
# "range_front": Front range finder distance
# "range_left": Leftward range finder distance 
# "range_right": Rightward range finder distance
# "range_back": Backward range finder distance
# "yaw": Yaw angle (rad)

# This is the main function where you will implement your control algorithm
def get_command(sensor_data):
    global on_ground, startpos, mode, ctrl_timer, t, fwd_vel_prev, left_vel_prev, yaw_desired
    global prev_pos, num_loops_stuck, firstpass_goal, k_a, k_r, possible_pad_locations, num_possible_pads_locations
    global list_of_visited_locations, grade, goal, is_landed, landpad_timer, first_landpad_location, second_landpad_location, middle_landpad_location
    # Open a window to display the camera image
    # NOTE: Displaying the camera image will slow down the simulation, this is just for testing
    # cv2.imshow('Camera Feed', camera_data)
    # cv2.waitKey(1)
    
    # Take off
    if startpos is None:
        startpos = [STARTING_POSE[0], STARTING_POSE[1], sensor_data['range_down']]    
    if on_ground and sensor_data['range_down'] < 0.3:
        control_command = [0.0, 0.0, height_desired, 0.0]
        return control_command
    else:
        on_ground = False

    # adjust the relative position
    sensor_data['x_global'] += startpos[0]
    sensor_data['y_global'] += startpos[1]

    # ---- YOUR CODE HERE ----
    # Set Control Command
    control_command = [0.0, 0.0, height_desired, 0.0]
    
    # Get the occupancy map data
    map = occupancy_map(sensor_data)

    # Decide Where the Goal is Based on the Grade
    assign_goal(sensor_data, map)

    match mode:
        case 'takeoff':
            # Set a timer and rise to desired height for 3 seconds
            if ctrl_timer is None:
                ctrl_timer = time.time()

            if time.time() - ctrl_timer < 3:
                control_command = [0.0, 0.0, height_desired, 0.0]
                return control_command
            else:
                mode = 'find goal'
        case 'find goal':
            
            if t % UPDATE_FREQUENCY == 0:
                # Get Drone location
                drone_location = np.array([sensor_data['x_global'], sensor_data['y_global']])

                # Get the vector from the drone to the goal
                attractive_force, attractive_force_wf, attractive_magnitude = calc_attractive_force(sensor_data) 

                # Get the repulsive force from nearby obstacles
                repulsive_force, repulsive_force_wf, repulsive_magnitude = calc_repulsive_force(sensor_data, map)

                # Adjust attractive and repulsive gains based on if the drone is stuck
                adjust_gains(drone_location, prev_pos)

                # Calculate Resultant Force in Body Frame
                resultant_force = (k_a*attractive_force) + (k_r*repulsive_force)

                update_visualization(sensor_data, map, attractive_force_wf, attractive_magnitude, repulsive_force_wf, repulsive_magnitude)

                # Set the forward and left velocities
                fwd_vel = resultant_force[0] / attractive_magnitude / 5
                left_vel = resultant_force[1] / attractive_magnitude / 5

                # Set control command to move towards the goal while avoiding obstacles
                control_command = [fwd_vel, left_vel, height_desired, yaw_desired]
                
                # Set previous velocities
                fwd_vel_prev = fwd_vel
                left_vel_prev = left_vel

                # Update the previous position
                prev_pos.append(drone_location)
                if len(prev_pos) > 50:
                    prev_pos.pop(0)
            else:
                control_command = [fwd_vel_prev, left_vel_prev, height_desired, yaw_desired]

        case 'land':
            drone_location = np.array([sensor_data['x_global'], sensor_data['y_global']])
            

            if grade == 5.5:
                control_command = [0.0, 0.0, 0.01, 0.0]
            else:
                # If we only have one landing pad location, continue moving towards goal until we have the second one
                if second_landpad_location is None and landpad_timer < 4 * 100:
                    # Ignore repulsive forces and move towards goal
                    # print('First Landing Pad Location Found! Continue Moving Towards Goal')
                    attractive_force, attractive_force_wf, attractive_magnitude = calc_attractive_force(sensor_data)
                    control_command = [attractive_force[0] / 50, attractive_force[1] / 50, height_desired, 0.0]
                    landpad_timer += 1
                    # print('Landpad Timer: ', landpad_timer)
                elif landpad_timer >= 4 * 200:
                    # print('No Second Landing Pad Found. Just Fail me already!')
                    mode = 'find goal'
                    first_landpad_location = None
                    second_landpad_location = None 
                    middle_landpad_location = None
                    landpad_timer = 0
                    control_command = [0.0, 0.0, height_desired, 0.0]

                elif second_landpad_location is not None and np.linalg.norm(drone_location - goal) > 0.1: # If we have two landing pad locations, move towards the midpoint between the two
                    # print('Second Landing Pad Location Found! Move Towards the Midpoint Between the Two Landing Pads')
                    attractive_force, attractive_force_wf, attractive_magnitude = calc_attractive_force(sensor_data)
                    control_command = [attractive_force[0] / 50, attractive_force[1] / 50, height_desired, 0.0]
                else:
                    if not is_landed:
                        # print('Landing on the Landing Pad')
                        control_command = [0.0, 0.0, 0.01, 0.0]
                        if sensor_data['range_down'] < 0.1:
                            control_command = [0.0, 0.0, height_desired, 0.0]
                            is_landed = True
                            mode = 'find goal'
                            goal = firstpass_goal
                            grade = 5.0
                            # print('Landed on the Landing Pad. \n Grade Increased to 5.0 \n')
                    else:
                        mode = 'find goal'
                        # landpad_timer = 0
                        control_command = [0.0, 0.0, height_desired, 0.0]

    map = occupancy_map(sensor_data)
    t += 1
    return control_command # Ordered as array with: [v_forward_cmd, v_left_cmd, alt_cmd, yaw_rate_cmd]

def calc_attractive_force(sensor_data):
    '''
    Middle Layer Method: Attractive Force Calculation
    '''
    attractive_force_wf, attractive_magnitude = get_vector_wf_to_goal(sensor_data) # Get world frame Vector from drone location to Goal
    attractive_force = convert_to_body_frame(attractive_force_wf, sensor_data['yaw']) # Convert from world frame to body frame
    attractive_force, attractive_force_wf, attractive_magnitude = ensure_strength(attractive_force, attractive_force_wf, attractive_magnitude) # Ensure the strength of the attractive force is not too high
    return attractive_force, attractive_force_wf, attractive_magnitude

def calc_repulsive_force(sensor_data, map):
    '''
    Middle Layer Method: Repulsive Force Calculation
    '''
    drone_location = np.array([sensor_data['x_global'], sensor_data['y_global']])
    repulsive_force_wf, repulsive_magnitude = compute_repulsive_force(map, drone_location)
    repulsive_force = convert_to_body_frame(repulsive_force_wf, sensor_data['yaw'])
    return repulsive_force, repulsive_force_wf, repulsive_magnitude

def adjust_gains(drone_location, prev_pos):
    '''
    Middle Layer Method: Adjust Gains based on if the drone is stuck.
    '''
    global k_a, k_r
    if len(prev_pos) > 0:
        if is_stuck(drone_location, prev_pos):
            # If the drone is stuck, increase attractive forcea and decrease repulsive force
            k_a *= 1.05
            k_r *= 0.95
        else:
            # Reset the gains
            k_a = 1.0
            k_r = 1.8
    return

def assign_goal(sensor_data, map):
    '''
    Assigns the goal location based on the current goal. Eg. Cross the map, find landing pad, find pink box, etc.
    '''
    global mode, firstpass_goal, grade, list_of_visited_locations, goal, first_landpad_location, second_landpad_location, prev_height, middle_landpad_location
    drone_location = np.array([sensor_data['x_global'], sensor_data['y_global']])
    match mode:
        case 'takeoff':
            return firstpass_goal # Take off to the first goal
        case 'find goal':
            
            # First Goal: Get to the Other Side
            if grade == 0.0: # Change later when adding visualization
                if drone_location[0] > 3.5:
                    # print('Increase Grade to 4.0')
                    grade = 4.0
                return firstpass_goal
            
            # Second Goal: Find and Land on the Landing Pad
            elif grade == 4.0 or grade == 4.5:
                # Do grid search for the landing pad by assigning next goal in grid
                # Update Visited Locations making sure not to add the same location twice
                # print('Range Down: ', sensor_data['range_down'])
                landing = is_above_landpad(sensor_data)
                if landing:
                    # print('First Landing Pad Location Found!')
                    prev_height = sensor_data['range_down']
                    first_landpad_location = drone_location
                    # print('First Landing Pad Location: ', first_landpad_location)
                    mode = 'land'
                    return goal

                drone_location = np.round(drone_location * 10)
                
                if drone_location.tolist() not in list_of_visited_locations.tolist():
                    list_of_visited_locations = np.append(list_of_visited_locations, [drone_location], axis=0)
                   
                # Build a grid of possible landing pad locations
                # possible_locations = get_possible_pad_locations(drone_location, map)        # fuck this, returns a list of tuples (he thinks so)

                x_list = [3.7, 3.9, 4.1, 4.3, 4.5, 4.7]
                y_low = 0.2
                y_high = 2.8
                p1 = [x_list[0] , y_low, height_desired]
                p2 = [x_list[0], y_high, height_desired]
                p3 = [x_list[1], y_high, height_desired]
                p4 = [x_list[1], y_low, height_desired]
                p5 = [x_list[2], y_low, height_desired]
                p6 = [x_list[2], y_high, height_desired]
                p7 = [x_list[3], y_high, height_desired]
                p8 = [x_list[3], y_low, height_desired]
                p9 = [x_list[4], y_low, height_desired]
                p10 = [x_list[4], y_high, height_desired]
                p10_1 = [4.85, 1.5, height_desired]
                p11 = [x_list[5], y_high, height_desired]
                p12 = [x_list[5], y_low, height_desired]

                # path = [p1,p3, p5, p7, p9, p11, p12]
                # path = [p12,p11, p9, p7, p5, p3, p1, p6, p2, p8, p10, p12] #after p1 it's random
                possible_locations = [p12, p12, p10, p10_1, p10, p8, p6, p4, p2, p11, p9, p7, p5, p3, p1]
                
                if True:
                    
                    # Make sure drone has reached goal location
                    if np.array_equal(goal, firstpass_goal):
                        goal = set_next_goal(possible_locations, drone_location) / 10
                        return

                    if np.linalg.norm(drone_location*0.1 - goal) < 0.075:
                        # Assign the next goal location
                        # print('Made it to the Goal Location!')
                        goal = set_next_goal(possible_locations, drone_location) / 10
                        # print('Next Goal: ', goal)
                    return 
                
                else:
                    # print('No Pad Found. Just Fail me already!')
                    goal = firstpass_goal

        
                
            elif grade == 5.0 or grade == 5.25:
                
                # If drone is not at starting position, return to starting position
                if np.linalg.norm(drone_location - startpos[:2]) > 0.1:
                    # print('Return to the Starting Location: ', startpos[:2])
                    goal = startpos[:2]
                    mode = 'find goal'
                    return goal
                else:
                    # print('Increase Grade to 5.5')
                    grade = 5.5
                    mode = 'land'
                    goal = startpos[:2]
                    return goal
        
        case 'land':
            # Find when the down sensor jumps to a higher number (When it leaves the landing pad)
            if grade == 5.5:
                # Land on starting pad
                goal = startpos[:2]
                return goal
            
            
            if sensor_data['range_down'] - prev_height > 0.1 and second_landpad_location is None:
                # print('Second Landing Pad Location Found!')
                second_landpad_location = drone_location
                # print('Second Landing Pad Location: ', second_landpad_location)
                
                # Change the goal location to the midpoint between the two landing pad locations
                goal = (first_landpad_location + second_landpad_location) / 2
                middle_landpad_location = goal
                return goal
            elif second_landpad_location is not None:
                # Change the goal location to the middle of the landing pad and the starting location
                goal = middle_landpad_location
                return goal
            else:
                return goal
            
            
def is_above_landpad(sensor_data):
    '''
    This function checks if the drone is above the landing pad.
    '''
    if sensor_data['range_down'] < 0.45:
        return True
    else:
        return False

def get_possible_pad_locations(drone_location, map):
    '''
    This function finds possible landing pad locations based on the current and 
     previous drone locations and the obstacle locations. It returns a list of
     the possible landing pad locations for the drone to loop through.
    '''
    global num_possible_pads_locations, possible_pad_locations, t, list_of_visited_locations
    if num_possible_pads_locations is None or t % 50 == 0:
        # Get all the free or unknown locations in landing zone.
        possible_pad_indices = np.column_stack(np.where(map > 0))
        # print('#########################################################')
        # print('Number of Possible Pad Locations Before: ', possible_pad_indices.shape[0])
        
        # Filter out locations outside the landing zone
        possible_pad_indices = possible_pad_indices[possible_pad_indices[:, 0] > 35]
        # print('Number of Possible Pad Locations After Y Filter: ', possible_pad_indices.shape[0])

        # Filter out all locations with even numbered x and y coordinates
        possible_pad_indices = even_number_filter(possible_pad_indices)
        # print('Number of Possible Pad Locations After Even Number Filter: ', possible_pad_indices.shape[0])

        # Filter out locations that are next to an obstacle
        possible_pad_indices = adjacent_obstacle_filter(possible_pad_indices, map)
        # print('Number of Possible Pad Locations After Adjacent Filter: ', possible_pad_indices.shape[0])
        

        # Filter out locations that have already been visited
        possible_pad_indices = visited_location_filter(possible_pad_indices, list_of_visited_locations)
        # print('Number of Possible Pad Locations After Visited Filter: ', possible_pad_indices.shape[0])
        
        # Update Global Variables
        num_possible_pads_locations = possible_pad_indices.shape[0]
        possible_pad_locations = possible_pad_indices

        return possible_pad_locations
    
    elif num_possible_pads_locations == 0:
        # print('No Pad Found. Just Fail me already!')
        return None
    
    else:
        return possible_pad_locations

def even_number_filter(possible_pad_indices):
    '''
    This function filters out possible landing pad locations that have even numbered x and y coordinates.
    '''
    possible_pad_indices = possible_pad_indices[(possible_pad_indices[:, 0] % 2 != 0) & (possible_pad_indices[:, 1] % 2 != 0)]

    return possible_pad_indices

def adjacent_obstacle_filter(possible_pad_indices, occupancy_map):
    '''
    This function filters out possible landing pad locations that are adjacent to obstacles.
    '''
    num_adjacent_neighbors = 1

    # Find the indices of all the obstacles on the map
    obstacle_indices = np.column_stack(np.where(occupancy_map < 0))

    # Filter out obstacles with an x coordinate less than 35
    obstacle_indices = obstacle_indices[obstacle_indices[:, 0] > 35]

    for N in range(num_adjacent_neighbors):

        # Create an array of the offsets
        offsets = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])

        # Add the offsets to the obstacle indices
        adjacent_obstacle_indices = obstacle_indices[:, None] + offsets

        # Reshape and convert to a list of indices
        adjacent_obstacle_indices = adjacent_obstacle_indices.reshape(-1, 2).tolist()

        # Append the adjacent obstacle indices to the list of obstacle indices
        obstacle_indices = np.append(obstacle_indices, adjacent_obstacle_indices, axis=0)

        # Remove adjacent obstacle indices from the list of possible pad indices
        # Convert 2D arrays to structured arrays
        structured_obstacle_indices = np.core.records.fromarrays(obstacle_indices.transpose(), 
                                                                names='f0, f1', 
                                                                formats = 'i8, i8')

        structured_possible_pad_indices = np.core.records.fromarrays(possible_pad_indices.transpose(), 
                                                                    names='f0, f1', 
                                                                    formats = 'i8, i8')

        # Use numpy setdiff1d to find rows in possible_pad_indices that are not in obstacle_indices
        possible_pad_indices = np.setdiff1d(structured_possible_pad_indices, structured_obstacle_indices).view(np.int64).reshape(-1, 2)
        #possible_pad_indices = possible_pad_indices.view(np.int64).reshape(-1, 2)

    return possible_pad_indices

def visited_location_filter(possible_pad_indices, list_of_visited_locations):
    '''
    This function filters out possible landing pad locations that have already been visited.
    '''
    # If the list of visited locations is empty, return the possible pad indices
    if list_of_visited_locations.size == 0:
        return possible_pad_indices

    # Convert Visited Locations Array from Tuples to Numpy Array
    # list_of_visited_locations = np.array(list(list_of_visited_locations.tolist()))
    # Convert array of tuples to numpy array
    visited_locations = np.vstack(list_of_visited_locations).astype(int)
    possible_pad_indices = np.column_stack(possible_pad_indices.T)
   
    structured_visited_indices = np.core.records.fromarrays(visited_locations.transpose(), 
                                                                names='f0, f1', 
                                                                formats = 'i8, i8')

    structured_possible_pad_indices = np.core.records.fromarrays(possible_pad_indices.transpose(), 
                                                                names='f0, f1', 
                                                                formats = 'i8, i8')
    
    # Use numpy setdiff1d to find rows in possible_pad_indices that are not in the visited list
    possible_pad_indices = np.setdiff1d(structured_possible_pad_indices, structured_visited_indices).view(np.int64).reshape(-1, 2)

    return possible_pad_indices

def set_next_goal(possible_locations, drone_location):
    global path_index
    '''
    This function sets the next goal location for the drone to move to.
    '''
    current_setpoint = possible_locations[path_index]
    if np.linalg.norm([drone_location[0] - current_setpoint[0], drone_location[1] - current_setpoint[1]]) < 0.1 :
        print("next setpoint in landing region")
        path_index += 1

    goal = possible_locations[path_index]
    return goal

def is_stuck(current_pos, prev_pos, threshold=0.2, N=50):
    """
    Check if the drone is stuck in one position. If more than N loops, then the drone is stuck.
    params:
    current_pos: Current position of the drone in the world frame (x, y)
    prev_pos: Previous 5 positions of the drone in the world frame [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
    num_loops_stuck: Number of loops the drone has been stuck in one position
    threshold: Threshold for determining if the drone is stuck in meters
    """
    global num_loops_stuck, firstpass_goal, goal, mode, first_landpad_location, second_landpad_location, middle_landpad_location
    # Get the Distances between the current position and the previous N positions
    prev_pos_np = np.array(prev_pos[-N:])  # Convert the last N positions to a NumPy array
    distances = np.linalg.norm(current_pos - prev_pos_np, axis=1)

    # Check if the drone is stuck
    if np.all(distances < threshold):
        num_loops_stuck += 1
        # print('Drone is stuck in one position for {} loops'.format(num_loops_stuck))
        mode = 'find goal'
        first_landpad_location = None
        second_landpad_location = None 
        middle_landpad_location = None
        if num_loops_stuck > 150 and num_loops_stuck < 200:
            # print('Drone is stuck in one position for {} loops. Drone is Stuck! Change Goal Location.'.format(num_loops_stuck))
            goal = np.array([4.0, 0.3 ]) # Change the goal location
        elif num_loops_stuck >= 150:
            # print('Drone is stuck in one position for {} loops. Drone is Stuck! Change Goal Location.'.format(num_loops_stuck))
            goal = np.array([4.0, 2.7]) # Change the goal location back to the original
        return True
    else:
        if num_loops_stuck > 0:
            if num_loops_stuck > 200:
                goal = firstpass_goal # Change the goal location back to the original
            # print('Drone is not stuck anymore!')
        num_loops_stuck = 0
        return False

def compute_repulsive_force(occupancy_map, drone_location):
    repulsive_force = np.zeros(2)  # Initialize repulsive force vector
    drone_location = np.flip(drone_location)  # Swap X and Y axes because input is by default (y,x)
    # Find indices of obstacles in the occupancy map
    obstacle_indices = np.flip(np.where(occupancy_map < 0))
    # print('\n\n#########################################################\n\n')
    # print('Number of Obstacles Before', obstacle_indices[0].shape[0])
    if len(obstacle_indices[0]) > 0:  # Check if obstacles are present
        # Compute repulsive forces from each obstacle
        obstacle_locations = np.column_stack((obstacle_indices[0], obstacle_indices[1]))
        
        # Convert Obstacle Locations to from Grid to Meters
        obstacle_locations = obstacle_locations * 0.1
        # print('Obstacle Locations Before Limit: ', obstacle_locations)
        # print('Drone Location: ', drone_location)
        distances = np.linalg.norm(obstacle_locations - drone_location, axis=1)

        
        directions = (drone_location - obstacle_locations) / distances[:, np.newaxis]
        # print('Distances: ', distances)

        # Delete Obstacles more than N meters away from the drone location
        N = 1.0
        
        obstacle_locations = obstacle_locations[distances < N]
        directions = directions[distances < N]
        distances = distances[distances < N]
        # Get indices of obstacles within N meters
        close_obstacle_indices = np.where(distances < N)
        num_close_obstacles = close_obstacle_indices[0].shape[0]
        # print('Number of Obstacles After', num_close_obstacles)
        # print('Obstacle Locations: ', obstacle_locations)
        # print('Directions: ', directions)

        magnitudes = 1 / distances**1.8 / num_close_obstacles # Example: inverse-distance function
        # Sum up repulsive forces from all obstacles
        repulsive_force = np.flip(np.sum(magnitudes[:, np.newaxis] * directions, axis=0))
    
    # Calculate the magnitude of the repulsive force
    magnitude = np.linalg.norm(repulsive_force)
    # print('Magnitude: ', magnitude)
    
    # Avoid division by zero
    if magnitude == 0: 
        magnitude = 1e-6
    #print('Magnitude: ', magnitude)
    return repulsive_force, magnitude

def get_vector_wf_to_goal(sensor_data):
    """
    Get the  normalized vector from the drone to the goal in the world frame
    """
    global goal
    vector = goal - np.array([sensor_data['x_global'], sensor_data['y_global']])
    magnitude = np.linalg.norm(vector)
    # magnitude = 5 # Set equal to 3 because we always want to move forward at a reasonable speed
    return vector, magnitude

def ensure_strength(vector, vector_wf, magnitude):
    """
    Ensure the strength of the vector is at least N:
    Good values for N range from 3 to 10
    """
    N = 5
    if magnitude < N:
        vector = (vector / magnitude) * N
        vector_wf = (vector_wf / magnitude) * N
        magnitude = N
    return vector, vector_wf, magnitude

def convert_to_body_frame(vector, yaw):
    """
    Convert the vector from world frame to body frame
    """
    rotation_matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    vector = rotation_matrix@vector
    return vector

def update_visualization(sensor_data, map, attractive_force, attractive_magnitude, repulsive_force, repulsive_magnitude):
    global canvas, t, k_a, k_r, goal, possible_pad_locations, num_possible_pads_locations
    arrow_size = 10
    map_size_x = 300
    map_size_y = 500
    
    # Calculate Resultant Force in World Frame for Visualization
    resultant_force = (k_a*attractive_force) + (k_r*repulsive_force)
    
    if t % 50 == 0:
        #print(f'xglobal: {sensor_data["x_global"]}, yglobal: {sensor_data["y_global"]}')
        xdrone = int(sensor_data['y_global'] * 100)  # Swap X and Y axes
        ydrone = int(sensor_data['x_global'] * 100)  # Swap X and Y axes
        xgoal = int(goal[1] * 100)  # Swap X and Y axes
        ygoal = int(goal[0] * 100)  # Swap X and Y axes
        # print(f'xdrone: {xdrone}, ydrone: {ydrone}, xgoal: {xgoal}, ygoal: {ygoal}')
        if canvas is None:
            # Create an empty canvas
            canvas = np.zeros((map_size_y, map_size_x, 3), dtype=np.uint8) * 255  # Swap canvas dimensions

        # Clear canvas
        canvas.fill(255)
        
        # Plot the map with upscaling (Comment out if maps are the same size)
        map = np.kron(map, np.ones((10, 10)))
        idx_obstacles = np.where(map < 0)

        canvas[map_size_y-idx_obstacles[0]-1, map_size_x-idx_obstacles[1]-1] = (0, 0, 255)  # Red
        # Plot Sensor Data
        text_position = (60, 20)  # Adjust as needed
        # text_position = (text_position[0], text_position[1] + 20)  # Move text position down for next item

        # Plot Sensor Data
        text_position = (10, 350)  # Adjust as needed
        cv2.putText(canvas, f'Range Down: {round(sensor_data["range_down"], 3)}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f'Range Front: {round(sensor_data["range_front"], 3)}', (text_position[0], text_position[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f'Range Left: {round(sensor_data["range_left"], 3)}', (text_position[0], text_position[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f'Range Right: {round(sensor_data["range_right"], 3)}', (text_position[0], text_position[1] + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f'Range Back: {round(sensor_data["range_back"], 3)}', (text_position[0], text_position[1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Plot drone and goal positions
        cv2.circle(canvas, (map_size_x - xdrone, map_size_y - ydrone), 5, (0, 0, 255), -1)  # Red for drone, mirror X coordinate
        cv2.circle(canvas, (map_size_x - xgoal, map_size_y - ygoal), 5, (255, 0, 0), -1)  # Blue for goal, mirror X coordinate

        # Plot the possible landing pad locations
        if num_possible_pads_locations is not None and num_possible_pads_locations > 0:
            for location in possible_pad_locations:
                cv2.circle(canvas, (map_size_x - int(location[1] * 10), map_size_y - int(location[0] * 10)), 5, (0, 255, 0), -1)

        # Plot the attractive force vector
        if attractive_magnitude != 0:
            arrow_end_point = (map_size_x - (xdrone + int(attractive_force[1] * arrow_size)), map_size_y - (ydrone + int(attractive_force[0] * arrow_size)))
            cv2.arrowedLine(canvas, (map_size_x - xdrone, map_size_y - ydrone), arrow_end_point, (0, 255, 0), thickness=1, tipLength=0.3)

        # Plot the repulsive force vector
        if repulsive_magnitude != 0:
            arrow_end_point = (map_size_x - (xdrone + int(repulsive_force[1] * arrow_size)), map_size_y - (ydrone + int(repulsive_force[0] * arrow_size)))
            cv2.arrowedLine(canvas, (map_size_x - xdrone, map_size_y - ydrone), arrow_end_point, (255, 0, 0), thickness=1, tipLength=0.3)

        # Plot the resultant force vector
        resultant_magnitude = np.linalg.norm(resultant_force)
        if resultant_magnitude != 0:
            arrow_end_point = (map_size_x - (xdrone + int(resultant_force[1] * arrow_size)), map_size_y - (ydrone + int(resultant_force[0] * arrow_size)))
            cv2.arrowedLine(canvas, (map_size_x - xdrone, map_size_y - ydrone), arrow_end_point, (0, 0, 0), thickness=1, tipLength=0.3)
        
        # Show the updated canvas
        cv2.imshow("Map", canvas)
        cv2.waitKey(1)  # Wait for a short time to update the display


# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 3.0 # meter
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.1 # meter
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

    # add a perimeter of obstacles around the occupancy map
    map = add_perimeter_obstacles(map)

    # only plot every Nth time step (comment out if not needed)
    # if t % 50 == 0:
    #     plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
    #     plt.savefig("map.png")
    #     plt.close()
    return map

# Add a boarder of obstacles around the perimeter of the occupancy map
def add_perimeter_obstacles(map):
    '''
    This function adds a perimeter of obstacles around the occupancy map
    '''
    map[0, :] = -1
    map[-1, :] = -1
    map[:, 0] = -1
    map[:, -1] = -1
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