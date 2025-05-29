import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Tuple


class AdaptivePathPostprocessor:
    def __init__(self,
                 min_knot_spacing: float = 0.1,
                 max_knot_spacing: float = 1.0,
                 curvature_threshold: float = 0.5,
                 density_factor: float = 3.0):
        """
        Fixed version with robust array handling

        Args:
            min_knot_spacing: Minimum distance between knots (high curvature)
            max_knot_spacing: Maximum distance between knots (low curvature)
            curvature_threshold: Curvature level triggering high density
            density_factor: How much denser knots should be in high-curvature zones
        """
        self.min_spacing = min_knot_spacing
        self.max_spacing = max_knot_spacing
        self.curv_thresh = curvature_threshold
        self.density_factor = density_factor

    def _calculate_curvature(self, path: np.ndarray) -> np.ndarray:
        """Compute normalized curvature for 3D path with shape checks"""
        # Ensure proper 2D array shape (N,3)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("Path must be an Nx3 array")

        # Calculate derivatives
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        dz = np.gradient(path[:, 2])

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        ddz = np.gradient(dz)

        # Compute curvature
        numerator = np.sqrt(ddx ** 2 + ddy ** 2 + ddz ** 2)
        denominator = dx ** 2 + dy ** 2 + dz ** 2 + 1e-6
        curvature = numerator / denominator

        # Normalize to [0,1]
        return (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-6)

    def _generate_adaptive_knots(self, path: np.ndarray, curvature: np.ndarray) -> np.ndarray:
        """Create knot positions based on curvature"""
        # Calculate cumulative path length
        diffs = np.diff(path, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        cum_length = np.insert(np.cumsum(segment_lengths), 0, 0)
        total_length = cum_length[-1]

        knots = [0.0]  # Start at beginning
        current_pos = 0.0

        while current_pos < total_length:
            # Find nearest path point
            idx = np.argmin(np.abs(cum_length - current_pos))
            local_curvature = curvature[idx]

            # Adaptive spacing
            if local_curvature > self.curv_thresh:
                spacing = self.min_spacing
            else:
                spacing = np.interp(local_curvature,
                                    [0, self.curv_thresh],
                                    [self.max_spacing, self.min_spacing / self.density_factor])

            current_pos += spacing
            if current_pos < total_length:
                knots.append(current_pos)

        # Ensure endpoint is included
        if not np.isclose(knots[-1], total_length):
            knots.append(total_length)

        return np.array(knots) / total_length  # Normalize to [0,1]

    def resample_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Robust version with proper array handling

        Args:
            path: Input path as list of (x,y,z) tuples

        Returns:
            Resampled path with adaptive knot density
        """
        # Convert to properly shaped numpy array
        path_array = np.array(path)
        if path_array.ndim == 1:
            path_array = path_array.reshape(-1, 3)  # Ensure Nx3 shape

        # Calculate curvature
        curvature = self._calculate_curvature(path_array)

        # Generate adaptive knots
        knots = self._generate_adaptive_knots(path_array, curvature)

        # Create natural cubic spline
        t_original = np.linspace(0, 1, len(path_array))
        spline = CubicSpline(t_original, path_array, bc_type='natural')

        # Resample and convert back to tuples
        return [tuple(pt) for pt in spline(knots)]