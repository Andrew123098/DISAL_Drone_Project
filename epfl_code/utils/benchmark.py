import numpy as np
from scipy import interpolate, signal
from typing import List, Tuple
import matplotlib.pyplot as plt


class Benchmarker:
    def __init__(self, trajectory: List[Tuple[float, float, float]],
                 occupancy_grid: np.ndarray,
                 grid_resolution: float,
                 drone_physical_limits: dict = None):
        """
        Initialize the trajectory benchmarker.

        Args:
            trajectory: List of (x,y,z) tuples representing the path
            occupancy_grid: 3D numpy array representing the environment (1=occupied, 0=free)
            grid_resolution: Size of each grid cell in meters
            drone_physical_limits: Dictionary containing drone's physical limits
        """
        self.trajectory = np.array(trajectory)
        self.occupancy_grid = occupancy_grid
        self.grid_resolution = grid_resolution
        self.drone_limits = drone_physical_limits or {
            'max_velocity': 10.0,
            'max_acceleration': 20.0,
            'max_jerk': 100.0,
            'max_snap': 500.0,

            'paper_mass': 32,                    # [g]
            'paper_max_thrust': 0.41202,         # [N]
            'paper_max_fwd_velocity': 7.0,      # [m/s]
            'paper_max_rise_velocity': 2.96,     # [m/s]
            'paper_max_fwd_acceleration': 6.2,  # [m/s^2]
        }

        """
        Maximum Drone Limits for Crazyflie 2.0 were benchmarked in this 2021 paper:
        https://www.researchgate.net/publication/348240012_Deep_Reinforcement_Learning_based_Aggressive_Flight_Trajectory_Tracker
        """

        # Pre-compute derivatives for efficiency
        self._compute_derivatives()

    def _compute_derivatives(self):
        """Pre-compute time derivatives of the trajectory."""
        # Assuming uniform time sampling for simplicity
        # In practice, you might have actual timestamps
        self.velocity = np.gradient(self.trajectory, axis=0)
        self.acceleration = np.gradient(self.velocity, axis=0)
        self.jerk = np.gradient(self.acceleration, axis=0)
        self.snap = np.gradient(self.jerk, axis=0)

    def benchmark(self):
        """Run all benchmarks and return comprehensive results."""
        results = {
            'path_length': self.calculate_path_length(),
            'clearance': self.calculate_clearance_metrics(),
            'curvature': self.calculate_curvature_metrics(),
            'dynamics': self.calculate_dynamic_metrics(),
            'smoothness': self.calculate_smoothness_metrics(),
            'feasibility': self.check_feasibility(),
            'time_estimate': self.estimate_time_to_target()
        }
        return results

    def calculate_path_length(self):
        """Calculate total length of the trajectory."""
        diffs = np.diff(self.trajectory, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))

    def calculate_clearance_metrics(self):
        """Calculate minimum and average clearance from obstacles."""
        # Convert trajectory to grid coordinates
        grid_coords = self.real_to_grid(self.trajectory)

        min_clearance = float('inf')
        clearance_history = []

        for point in grid_coords:
            clearance = self._distance_to_nearest_obstacle(point)
            clearance_history.append(clearance)
            if clearance < min_clearance:
                min_clearance = clearance

        return {
            'min_clearance': min_clearance * self.grid_resolution,
            'avg_clearance': np.mean(clearance_history) * self.grid_resolution,
            'clearance_history': np.array(clearance_history) * self.grid_resolution
        }

    def _distance_to_nearest_obstacle(self, grid_point):
        """Helper to find distance to nearest obstacle for a grid point."""
        occupied_voxels = np.argwhere(self.occupancy_grid == 1)
        if len(occupied_voxels) == 0:
            return float('inf')
        distances = np.linalg.norm(occupied_voxels - grid_point, axis=1)
        return np.min(distances)

    def calculate_curvature_metrics(self):
        """Calculate curvature-related metrics."""
        # Compute curvature at each point
        velocity = np.gradient(self.trajectory, axis=0)
        acceleration = np.gradient(velocity, axis=0)

        cross_product = np.cross(velocity, acceleration)
        curvature = np.linalg.norm(cross_product, axis=1) / (np.linalg.norm(velocity, axis=1) ** 3 + 1e-6)

        # Compute curvature derivative for continuity analysis
        curvature_derivative = np.gradient(curvature)

        return {
            'max_curvature': np.max(curvature),
            'avg_curvature': np.mean(curvature),
            'curvature_continuity': np.max(np.abs(curvature_derivative)),
            'curvature_history': curvature
        }

    def calculate_dynamic_metrics(self):
        """Calculate velocity, acceleration, jerk, snap metrics."""
        velocity_norms = np.linalg.norm(self.velocity, axis=1)
        acceleration_norms = np.linalg.norm(self.acceleration, axis=1)
        jerk_norms = np.linalg.norm(self.jerk, axis=1)
        snap_norms = np.linalg.norm(self.snap, axis=1)

        return {
            'max_velocity': np.max(velocity_norms),
            'avg_velocity': np.mean(velocity_norms),
            'max_acceleration': np.max(acceleration_norms),
            'avg_acceleration': np.mean(acceleration_norms),
            'max_jerk': np.max(jerk_norms),
            'avg_jerk': np.mean(jerk_norms),
            'max_snap': np.max(snap_norms),
            'avg_snap': np.mean(snap_norms)
        }

    def calculate_smoothness_metrics(self):
        """Calculate combined smoothness metrics."""
        jerk_norms = np.linalg.norm(self.jerk, axis=1)
        snap_norms = np.linalg.norm(self.snap, axis=1)

        # Integrated squared jerk (common smoothness metric)
        integrated_jerk_squared = np.trapz(jerk_norms ** 2)

        return {
            'integrated_jerk_squared': integrated_jerk_squared,
            'smoothness_index': integrated_jerk_squared / len(self.trajectory)
        }

    def check_feasibility(self):
        """Check if trajectory stays within drone's physical limits."""
        metrics = self.calculate_dynamic_metrics()

        violations = {
            'velocity': metrics['max_velocity'] > self.drone_limits['max_velocity'],
            'acceleration': metrics['max_acceleration'] > self.drone_limits['max_acceleration'],
            'jerk': metrics['max_jerk'] > self.drone_limits['max_jerk'],
            'snap': metrics['max_snap'] > self.drone_limits['max_snap']
        }

        return {
            'within_limits': not any(violations.values()),
            'violations': violations
        }

    def estimate_time_to_target(self):
        """Estimate time to complete trajectory based on dynamics."""
        # Simple estimation assuming constant acceleration between points
        # More sophisticated methods could use the actual dynamics model
        path_length = self.calculate_path_length()
        avg_speed = np.mean(np.linalg.norm(self.velocity, axis=1))

        if avg_speed > 0:
            return path_length / avg_speed
        return float('inf')

    def real_to_grid(self, real_coords):
        """Convert real-world coordinates to grid coordinates."""
        return (real_coords / self.grid_resolution).astype(int)

    def grid_to_real(self, grid_coords):
        """Convert grid coordinates to real-world coordinates."""
        return grid_coords * self.grid_resolution

    def visualize_benchmarks(self, results=None):
        """Create visualization plots of key metrics."""
        if results is None:
            results = self.benchmark()

        plt.figure(figsize=(15, 10))

        # Velocity profile
        plt.subplot(3, 2, 1)
        velocity = np.linalg.norm(self.velocity, axis=1)
        plt.plot(velocity)
        plt.title('Velocity Profile')
        plt.xlabel('Path index')
        plt.ylabel('Velocity (m/s)')

        # Clearance
        plt.subplot(3, 2, 2)
        plt.plot(results['clearance']['clearance_history'])
        plt.title('Clearance from Obstacles')
        plt.xlabel('Path index')
        plt.ylabel('Clearance (m)')

        # Curvature
        plt.subplot(3, 2, 3)
        plt.plot(results['curvature']['curvature_history'])
        plt.title('Path Curvature')
        plt.xlabel('Path index')
        plt.ylabel('Curvature (1/m)')

        # Acceleration
        plt.subplot(3, 2, 4)
        acceleration = np.linalg.norm(self.acceleration, axis=1)
        plt.plot(acceleration)
        plt.title('Acceleration Profile')
        plt.xlabel('Path index')
        plt.ylabel('Acceleration (m/s²)')

        # Jerk
        plt.subplot(3, 2, 5)
        jerk = np.linalg.norm(self.jerk, axis=1)
        plt.plot(jerk)
        plt.title('Jerk Profile')
        plt.xlabel('Path index')
        plt.ylabel('Jerk (m/s³)')

        plt.tight_layout()
        plt.show()

    def mass_visualize(self, results, spline_degrees=None):
        """
        Visualize comparative benchmarks with separate plots for each metric,
        except for max/avg pairs which are shown together.

        Args:
            results: List of benchmark dictionaries
            spline_degrees: List of spline degrees used
        """
        if spline_degrees is None:
            spline_degrees = list(range(len(results)))

        # Set up color palette
        colors = plt.cm.viridis(np.linspace(0, 1, len(spline_degrees)))

        # Plot individual metrics
        self._plot_single_metric(results, spline_degrees, 'path_length', 'Path Length (m)')

        # Plot clearance metrics together
        self._plot_paired_metrics(
            results, spline_degrees,
            'clearance.min_clearance', 'clearance.avg_clearance',
            'Clearance (m)', ['Min Clearance', 'Avg Clearance']
        )

        # Plot curvature metrics together
        self._plot_paired_metrics(
            results, spline_degrees,
            'curvature.max_curvature', 'curvature.avg_curvature',
            'Curvature (1/m)', ['Max Curvature', 'Avg Curvature']
        )

        # Plot individual dynamics metrics
        self._plot_single_metric(results, spline_degrees, 'dynamics.max_velocity', 'Max Velocity (m/s)')
        self._plot_single_metric(results, spline_degrees, 'dynamics.max_acceleration', 'Max Acceleration (m/s²)')
        self._plot_single_metric(results, spline_degrees, 'dynamics.max_jerk', 'Max Jerk (m/s³)')
        self._plot_single_metric(results, spline_degrees, 'dynamics.max_snap', 'Max Snap (m/s⁴)')

        # Plot individual smoothness metrics
        self._plot_single_metric(results, spline_degrees, 'smoothness.integrated_jerk_squared',
                                 'Integrated Jerk Squared')
        self._plot_single_metric(results, spline_degrees, 'smoothness.smoothness_index', 'Smoothness Index')

        # Plot feasibility violations
        self._plot_violations(results, spline_degrees)

    def _plot_single_metric(self, results, spline_degrees, metric_path, title):
        """Plot a single metric across spline degrees."""
        values = []
        for res in results:
            val = res
            for part in metric_path.split('.'):
                val = val.get(part, None)
            values.append(val)

        plt.figure(figsize=(10, 5))
        bars = plt.bar(spline_degrees, values, color=plt.cm.viridis(np.linspace(0, 1, len(spline_degrees))))

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     f'{height:.4f}',
                     ha='center', va='bottom')

        plt.title(title)
        plt.xlabel('Spline Degree')
        plt.ylabel(title.split('(')[0].strip() if '(' in title else title)
        plt.grid(True, axis='y', alpha=0.5)
        plt.xticks(spline_degrees)
        plt.show()

    def _plot_paired_metrics(self, results, spline_degrees, metric1_path, metric2_path, ylabel, legend_labels):
        """Plot two related metrics together (e.g., max and avg)."""
        values1 = []
        values2 = []
        for res in results:
            # Get first metric
            val = res
            for part in metric1_path.split('.'):
                val = val.get(part, None)
            values1.append(val)

            # Get second metric
            val = res
            for part in metric2_path.split('.'):
                val = val.get(part, None)
            values2.append(val)

        plt.figure(figsize=(10, 5))
        x = np.arange(len(spline_degrees))
        width = 0.35

        bars1 = plt.bar(x - width / 2, values1, width, label=legend_labels[0])
        bars2 = plt.bar(x + width / 2, values2, width, label=legend_labels[1])

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height,
                         f'{height:.2f}',
                         ha='center', va='bottom')

        plt.title(f'{legend_labels[0].split()[0]} {ylabel}')
        plt.xlabel('Spline Degree')
        plt.ylabel(ylabel)
        plt.xticks(x, spline_degrees)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.5)
        plt.show()

    def _plot_violations(self, results, spline_degrees):
        """Plot feasibility violations."""
        metrics = [
            'feasibility.violations.velocity',
            'feasibility.violations.acceleration',
            'feasibility.violations.jerk',
            'feasibility.violations.snap'
        ]
        labels = ['Velocity', 'Acceleration', 'Jerk', 'Snap']

        values = {label: [] for label in labels}
        for res in results:
            for metric, label in zip(metrics, labels):
                val = res
                for part in metric.split('.'):
                    val = val.get(part, False)
                values[label].append(int(val))

        plt.figure(figsize=(10, 5))
        x = np.arange(len(spline_degrees))
        width = 0.2

        for i, (label, vals) in enumerate(values.items()):
            offset = width * (i - len(labels) / 2 + 0.5)
            plt.bar(x + offset, vals, width, label=label)

        plt.title('Feasibility Violations')
        plt.xlabel('Spline Degree')
        plt.ylabel('Violation (1=yes)')
        plt.xticks(x, spline_degrees)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.5)
        plt.show()

