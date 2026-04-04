"""
utils — Cross-cutting utility functions shared across pipeline stages.

Currently provides geometry helpers (ray-AABB intersection, pitch/yaw
conversions, bounding-box operations) used by gaze tracking and
phenomena analysis.
"""

from .geometry import (
    bbox_center,
    extend_ray,
    pitch_yaw_to_2d,
    ray_hits_box,
    ray_hits_cone,
)

__all__ = [
    "pitch_yaw_to_2d",
    "ray_hits_box",
    "ray_hits_cone",
    "extend_ray",
    "bbox_center",
]
