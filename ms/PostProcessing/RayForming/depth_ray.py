"""
RayForming/depth_ray.py — Depth-aware ray length estimation.

Uses monocular depth maps to scale ray length based on the estimated
depth at the gaze target region.  Near-camera targets produce shorter
rays; far targets produce longer rays.
"""
from __future__ import annotations

import numpy as np

from ms.utils.geometry import sample_depth_patch


def depth_adjusted_length(depth_map: np.ndarray,
                          target_xy: np.ndarray,
                          base_length: float,
                          length_min: float = 0.5,
                          length_max: float = 3.0,
                          sample_radius: int = 2) -> float:
    """Scale ray length by depth at the target region.

    Parameters
    ----------
    depth_map    : HxW normalized depth map (0 = near, 1 = far).
    target_xy    : (x, y) pixel coordinates of the gaze target.
    base_length  : unscaled ray length (face_width * ray_length_multiplier).
    length_min   : multiplier at depth=0 (nearest).
    length_max   : multiplier at depth=1 (farthest).
    sample_radius : patch half-size for robust depth sampling.

    Returns
    -------
    Scaled ray length.
    """
    d = sample_depth_patch(depth_map, float(target_xy[0]), float(target_xy[1]),
                           radius=sample_radius)
    # Clamp depth to [0, 1] for safety
    d = max(0.0, min(1.0, d))
    return base_length * (length_min + d * (length_max - length_min))
