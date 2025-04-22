import sys
from pathlib import Path

from pyparsing import Word, nums, Literal, Group, Suppress, OneOrMore


def parse_with_pyparsing(world_file_path):
    """More robust parsing using pyparsing"""
    real = Word(nums + '.-')
    translation = Suppress('translation') + Group(real + real + real)
    gate_def = Suppress('DEF') + Word('GATE' + nums) + Suppress('RacingGate {') + translation

    with open(world_file_path, 'r') as f:
        content = f.read()

    return [{'name': t[0], 'position': list(map(float, t[1]))}
            for t in gate_def.searchString(content)]


world_file = "C:/Users/andre/OneDrive/Documents/EPFL-DESKTOP-0FFTIDB/Xplore/DISAL_Drone_Project/epfl_code/worlds/crazyflie_world_assignment.wbt"
gates = parse_with_pyparsing(world_file)

# # Add Webots Python API to path (absolute path)
# webots_python_path = Path(r"C:\Program Files\Webots\lib\controller\python")
# if str(webots_python_path) not in sys.path:
#     sys.path.insert(0, str(webots_python_path))
#
# # Now use the standard Webots import
# from controller import Supervisor
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Initialize supervisor
# supervisor = Supervisor()
# timestep = int(supervisor.getBasicTimeStep())
#
# # Define 3D grid parameters
# world_size = (10, 10, 6)  # x, y, z dimensions in meters
# resolution = 0.2          # meters per voxel
# grid_shape = tuple(int(s/resolution) for s in world_size)
# occupancy_grid = np.zeros(grid_shape, dtype=np.int8)
#
# def world_to_grid(pos):
#     """Convert world coordinates to grid indices"""
#     return tuple(int((p + s/2)/resolution) for p, s in zip(pos, world_size))
#
# def mark_obstacle(node):
#     """Mark occupied voxels for a given Solid node"""
#     # Get node's absolute position and orientation
#     translation = np.array(node.getPosition())
#     rotation = node.getOrientation()
#
#     # Get bounding object
#     bounding_object = node.getField('boundingObject').getSFNode()
#     if not bounding_object:
#         return
#
#     # Handle different bounding object types
#     if bounding_object.getTypeName() == 'Box':
#         size = np.array(bounding_object.getField('size').getSFVec3f())
#         min_corner = translation - size/2
#         max_corner = translation + size/2
#
#         # Convert to grid coordinates
#         min_idx = world_to_grid(min_corner)
#         max_idx = world_to_grid(max_corner)
#
#         # Mark occupied voxels (with bounds checking)
#         for x in range(max(0, min_idx[0]), min(grid_shape[0], max_idx[0])):
#             for y in range(max(0, min_idx[1]), min(grid_shape[1], max_idx[1])):
#                 for z in range(max(0, min_idx[2]), min(grid_shape[2], max_idx[2])):
#                     occupancy_grid[x, y, z] = 1
#
#     elif bounding_object.getTypeName() == 'Plane':
#         size = bounding_object.getField('size').getSFVec2f()
#         # Planes are infinite in one dimension - we'll treat them as thin boxes
#         if abs(rotation[0]) > 0.9:  # YZ plane (facing x)
#             thickness = 0.1
#             size_3d = [thickness, size[0], size[1]]
#         elif abs(rotation[1]) > 0.9:  # XZ plane (facing y)
#             size_3d = [size[0], thickness, size[1]]
#         else:  # XY plane (facing z)
#             size_3d = [size[0], size[1], thickness]
#
#         min_corner = translation - np.array(size_3d)/2
#         max_corner = translation + np.array(size_3d)/2
#
#         min_idx = world_to_grid(min_corner)
#         max_idx = world_to_grid(max_corner)
#
#         for x in range(max(0, min_idx[0]), min(grid_shape[0], max_idx[0])):
#             for y in range(max(0, min_idx[1]), min(grid_shape[1], max_idx[1])):
#                 for z in range(max(0, min_idx[2]), min(grid_shape[2], max_idx[2])):
#                     occupancy_grid[x, y, z] = 1
#
# # Process all Solid nodes in the world
# root = supervisor.getRoot()
# children = root.getField('children')
#
# for i in range(children.getCount()):
#     node = children.getMFNode(i)
#     if node.getTypeName() == 'Solid' and node.getField('name').getSFString() not in ['crazyflie']:
#         mark_obstacle(node)
#
# # Visualize the 3D occupancy grid
# def plot_3d_occupancy(grid):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Get coordinates of occupied voxels
#     occupied = np.where(grid == 1)
#     ax.scatter(occupied[0], occupied[1], occupied[2], c='red', marker='s', alpha=0.3)
#
#     # Set labels and title
#     ax.set_xlabel('X (grid cells)')
#     ax.set_ylabel('Y (grid cells)')
#     ax.set_zlabel('Z (grid cells)')
#     ax.set_title('3D Occupancy Map')
#
#     # Set equal aspect ratio
#     ax.set_box_aspect([1, 1, 1])
#     plt.show()
#
# plot_3d_occupancy(occupancy_grid)
#
# # Save the occupancy grid for path planning
# np.save('3d_occupancy_grid.npy', occupancy_grid)
