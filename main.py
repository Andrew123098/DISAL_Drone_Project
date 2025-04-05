from src.visualize import Visualize
from src.astar import AStar

def main():
    # Compute the occupancy map
    visualizer = Visualize()
    map = visualizer.occupancy_map()

    # 4-directional movement
    astar_4d = AStar(map, '4d')
    path_4d = astar_4d.search()
    
    # 8-directional movement
    astar_8d = AStar(map, '8d')
    path_8d = astar_8d.search()

    # Colorize astar path on the map
    for pos in path_4d[1:-1]:
        map[pos] = 2

    for pos in path_8d[1:-1]:
        map[pos] = 3

    # Plot the occupancy map
    visualizer.plot_map(map)

if __name__ == "__main__":
    main()