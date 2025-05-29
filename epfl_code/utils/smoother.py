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
        """Combine points with proximity filtering while preserving loop structure"""
        astar = np.array(astar_path)
        controls = np.array(control_points)

        # Combine all points while maintaining original order
        all_points = np.vstack([astar, controls])
        original_indices = np.arange(len(all_points))

        # Create list of (point, original_index) tuples
        indexed_points = [(pt, idx) for idx, pt in enumerate(all_points)]

        # Always keep first and last points
        keep_mask = np.zeros(len(all_points), dtype=bool)
        keep_mask[0] = True
        keep_mask[-1] = True

        # Build KDTree for spatial queries
        if len(all_points) > 1:
            points_array = np.array([pt for pt, idx in indexed_points])
            tree = KDTree(points_array)

            for i, (pt, current_idx) in enumerate(indexed_points):
                if not keep_mask[i]:  # Only process points not already marked to keep
                    neighbors = tree.query_ball_point(pt, r=self.min_dist)

                    # Filter neighbors based on index proximity
                    close_neighbors = [
                        n for n in neighbors
                        if abs(original_indices[n] - current_idx) <= 4
                    ]

                    # If no close neighbors (within 4 indices), keep this point
                    if not close_neighbors:
                        keep_mask[i] = True
                    else:
                        # Only keep this point if it's the first in the close group
                        if i == min(close_neighbors):
                            keep_mask[i] = True

        filtered_points = all_points[keep_mask]

        # Final check to ensure we have enough points for the spline degree
        if len(filtered_points) < self.degree + 1:
            # Fallback: keep more points to satisfy spline requirements
            additional_needed = (self.degree + 1) - len(filtered_points)
            unfiltered_indices = np.where(~keep_mask)[0]
            keep_mask[unfiltered_indices[:additional_needed]] = True
            filtered_points = all_points[keep_mask]

        return filtered_points

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

