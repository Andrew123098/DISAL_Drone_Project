import numpy as np
import time
import cv2

# from epfl_code.Crazyflie_AAA.Code.my_control import control_command


# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

def get_command(sensor_data, camera_data, dt):
    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    # Get or create the singleton instance
    controller = Command(sensor_data, camera_data, dt)

    # Update with current data
    controller.update_sensor_data(sensor_data, camera_data, dt)

    # Run the state machine
    controller.state_machine()

    print("Current Command: ", controller.control_command)
    return controller.control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


class Command:
    # Singleton instance holder
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Controls instance creation (singleton pattern)"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, sensor_data=None, camera_data=None, dt=None):
        """Initialization that only happens once"""
        if not self._initialized:
            self.path = self.load_path()
            print("Path Loaded: ", self.path)
            self.path_index = 0
            self.path_length = len(self.path)
            self.time_start = time.time()
            self.time_elapsed = 0.0
            self.tolerance = 0.1
            self.state = "Takeoff"
            self.previous_state = "Idle"
            self.sensor_data = sensor_data
            self.current_XYZ = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]
            self.camera_data = camera_data
            self.dt = dt
            self.control_command = [0.0, 0.0, 0.0, 0.0]
            self._initialized = True

    def update_sensor_data(self, sensor_data, camera_data, dt):
        """Update current sensor readings"""
        self.sensor_data = sensor_data
        self.camera_data = camera_data
        self.dt = dt
        self.current_XYZ = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]

    def load_path(self):
        """
        Loads the precomputed path from a .npy file and returns it as a numpy array.
        """
        print("Loading Path...")
        path_file = "full_path"
        fid = "../../utils/" + path_file + ".npy"
        path = np.load(fid)
        print("Loaded Path" + path_file + ".npy" + " with shape: " + str(path.shape) + "\n")
        return path

    def state_machine(self):
        """
        Handler for drone state transitions. States include the following:
        - "Idle"
        - "Takeoff"
        - "Follow Path"
        - "Land"
        - "Done"
        - "Error"
        """
        # Check and print if we are making a state transition.
        state = self.state
        previous_state = self.previous_state
        if previous_state != state:
            self.previous_state = state
            print(f"State transition: {previous_state} -> {state}")

        match state:
            case "Idle":
                self.state = "Takeoff"
                return

            case "Takeoff":
                self._takeoff()
                return

            case "Follow Path":
                self._follow_path()
                return

            case "Land":
                self._land()
                return

            case "Done":
                self._done()
                return

            case "Error":
                self._error()
                return

            case _:
                self.state = "Error"
                print("Error: Invalid state.")
                return

    def _takeoff(self):
        """
        Starting state for the drone. Takeoff to a height of 1.0 meters.
        """
        # Rise at least 0.5 meters above the ground before starting path.
        if self.sensor_data['z_global'] < 0.84:
            self.control_command = [self.sensor_data['x_global'], self.sensor_data['y_global'], 1.0, self.sensor_data['yaw']]
            return self.control_command

        # Once over X meters, move towards first waypoint in path.
        else:
            first_waypoint = self.path[1]
            self.control_command = [first_waypoint[0], first_waypoint[1], first_waypoint[2], self.sensor_data['yaw']]
            self.state = "Follow Path"
            return self.control_command

    def _follow_path(self):
        """
        Follows the precomputed path.
        """
        print("Current Path Index: " + str(self.path_index))
        index = self.path_index
        current_setpoint = self.path[index]

        # Case: End of Path
        if index == self.path_length - 1:
            print("End of Path")
            self.state = "Land"
            self.control_command = [self.sensor_data['x_global'], self.sensor_data['y_global'], 0.0, self.sensor_data['yaw']]
            return self.control_command

        # Case: Following Path
        else:

            # Case: Not at Current Setpoint (Do not change Command)
            if not self.is_at_current_setpoint(current_setpoint):
                print("Not at Current Setpoint")
                # print("Current Command: ", self.control_command)
                return self.control_command

            # Case: At Current Setpoint, Increment Along Path
            else:
                print("At Current Setpoint")
                self.path_index += 1
                self.control_command = [self.path[self.path_index][0], self.path[self.path_index][1], self.path[self.path_index][2], self.sensor_data['yaw']]
                return self.control_command

    def _land(self):
        """
        Land the Drone
        """
        self.control_command = [self.path[0][0], self.path[0][1], 0.1, self.sensor_data['yaw']]
        return self.control_command

    def _done(self):
        """
        TODO: Save Logged Data to File
        """
        return None

    def _error(self):
        """
        Quit the program if the drone crashed.
        """
        quit()

    def is_at_current_setpoint(self, current_setpoint, tolerance=0.4):
        """
        Returns True if the drone is at the current setpoint within the specified tolerance.
        """
        if self.path_index == self.path_length - 2:
            print("TOLERANCE ADJUSTED")
            tolerance = 0.02

        distance = np.linalg.norm([current_setpoint[0] - self.current_XYZ[0], current_setpoint[1] - self.current_XYZ[1], current_setpoint[2] - self.current_XYZ[2]])
        if distance < tolerance:
            return True
        else:
            return False





