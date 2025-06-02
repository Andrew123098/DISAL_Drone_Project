import numpy as np
from scipy.interpolate import make_interp_spline, BSpline, CubicSpline
from scipy.spatial import KDTree
from typing import List, Tuple


class BSplineSmoother:
    def __init__(self, degree: int = 3, downsample_factor: int = 20,
                 min_point_distance: float = 0.3, spline_type: str = 'natural'):
        self.degree = degree
        self.downsample_factor = max(downsample_factor, 1)
        self.min_dist = min_point_distance
        self.spline_type = spline_type

    def _prepare_points(self, astar_path: List[Tuple[float, float, float]],
                        control_points: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Downsample the A* path by keeping every N-th point.
        Always keeps the first and last points.
        """
        # Convert to numpy array for easier manipulation
        path = np.array(astar_path)

        # If path is too short, return as is
        if len(path) <= self.degree + 1:
            return path

        # Create mask for points to keep
        num_points = len(path)
        keep_mask = np.zeros(num_points, dtype=bool)

        # Always keep first and last points
        keep_mask[0] = True
        keep_mask[-1] = True

        # Keep every N-th point based on downsample_factor
        step = max(1, int(self.downsample_factor))
        keep_mask[1:-1:step] = True

        # Get the downsampled path
        downsampled_path = path[keep_mask]

        # Ensure we have enough points for the spline degree
        while len(downsampled_path) < self.degree + 1:
            # Find longest segment and add a point in the middle
            segments = np.diff(downsampled_path, axis=0)
            segment_lengths = np.linalg.norm(segments, axis=1)
            longest_segment_idx = np.argmax(segment_lengths)

            # Insert middle point of longest segment
            mid_point = (downsampled_path[longest_segment_idx] +
                         downsampled_path[longest_segment_idx + 1]) / 2
            downsampled_path = np.insert(downsampled_path,
                                         longest_segment_idx + 1,
                                         mid_point,
                                         axis=0)

        return downsampled_path

    def smooth(self, astar_path: List[Tuple[float, float, float]],
               control_points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        points = self._prepare_points(astar_path, control_points)

        # Length-based parameterization
        diffs = np.diff(points, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        cum_length = np.insert(np.cumsum(seg_lengths), 0, 0)
        t = cum_length / cum_length[-1]

        # Create and evaluate spline
        spline = make_interp_spline(t, points, k=self.degree, bc_type=self.spline_type)

        n_samples = max(10, int(cum_length[-1] * 2))
        t_eval = np.linspace(0, 1, n_samples)

        return [tuple(pt) for pt in spline(t_eval)]

