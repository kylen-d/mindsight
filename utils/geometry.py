"""
utils/geometry.py — Shared ray-geometry and spatial utilities.

Functions here are used by both the gaze pipeline (gaze_processing.py,
gaze_pipeline.py) and the phenomena pipeline (phenomena_tracking.py).
Centralising them avoids cross-module coupling and exposes previously
private helpers through a neutral import path.
"""

import numpy as np

from constants import RAY_EXT_LENGTH as _RAY_EXT_LENGTH


def pitch_yaw_to_2d(pitch, yaw):
    """Convert pitch/yaw angles (radians) to a normalized 2-D direction vector."""
    d = np.array([-np.sin(pitch) * np.cos(yaw), -np.sin(yaw)])
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else d


def ray_hits_box(start, end, x1, y1, x2, y2):
    """Liang-Barsky segment-AABB intersection."""
    dx, dy = end[0]-start[0], end[1]-start[1]
    p = [-dx, dx, -dy, dy]
    q = [start[0]-x1, x2-start[0], start[1]-y1, y2-start[1]]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0: return False
        elif pi < 0: t0 = max(t0, qi / pi)
        else:        t1 = min(t1, qi / pi)
    return t0 <= t1


def ray_hits_cone(origin, direction, x1, y1, x2, y2, cone_half_angle_deg: float) -> bool:
    """
    Return True if any part of the bounding box intersects the gaze cone
    defined by `origin`, unit `direction`, and the given half-angle.

    Three exhaustive cases cover all bbox-cone intersections:
      1. The gaze origin is inside the bbox.
      2. Any corner of the bbox lies inside the cone.
      3. Either boundary ray of the cone intersects the bbox
         (catches the case where the cone straddles the bbox without any
         corner falling inside the cone).
    """
    half_rad   = np.radians(cone_half_angle_deg)
    cos_thresh = np.cos(half_rad)

    def _in_cone(px, py):
        tx, ty = px - origin[0], py - origin[1]
        dist   = np.sqrt(tx * tx + ty * ty)
        if dist < 1e-6:
            return True
        dot = tx * direction[0] + ty * direction[1]
        if dot <= 0:
            return False
        return (dot / dist) >= cos_thresh

    if x1 <= origin[0] <= x2 and y1 <= origin[1] <= y2:
        return True
    if any(_in_cone(px, py) for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))):
        return True
    c, s   = np.cos(half_rad), np.sin(half_rad)
    dx, dy = direction[0], direction[1]
    left_dir  = np.array([ c * dx - s * dy,  s * dx + c * dy])
    right_dir = np.array([ c * dx + s * dy, -s * dx + c * dy])
    far = (x2 - x1 + y2 - y1) * 100.0
    return (ray_hits_box(origin, origin + left_dir  * far, x1, y1, x2, y2) or
            ray_hits_box(origin, origin + right_dir * far, x1, y1, x2, y2))


def extend_ray(origin, end, length: float = _RAY_EXT_LENGTH):
    """Extend the (origin->end) direction to `length` pixels for face-to-face hit tests."""
    dv = np.asarray(end, float) - np.asarray(origin, float)
    dl = np.linalg.norm(dv)
    if dl < 1e-6:
        return np.asarray(end, float)
    return np.asarray(origin, float) + dv / dl * length


def bbox_center(obj) -> np.ndarray:
    """Return the centre of a detection bounding-box as a float numpy array."""
    return np.array([(obj['x1'] + obj['x2']) / 2,
                     (obj['y1'] + obj['y2']) / 2], float)
