from src.visualize3D import Visualize3D
from src.astar3D import AStar3D


def main():
     # Create visualizer and map
    visualizer = Visualize3D()
    map_3d = visualizer.occupancy_map()
    
    # Run A* (using your existing AStar3D class)
    astar = AStar3D(map_3d, '26d')  # or '26d' for full 3D movement
    path = astar.search()
    
    # Visualize
    if path:
        print(f"Path found with {len(path)} steps")
        visualizer.plot_path(map_3d, path)
    else:
        print("No path found")
        visualizer.plot_path(map_3d, None, '3d_map_no_path.png')

# Example usage with the A* algorithm
if __name__ == "__main__":
   main()