"""
utils/geometry.py — Shared ray-geometry and spatial utilities.

Functions here are used by both the gaze pipeline (gaze_processing.py,
gaze_pipeline.py) and the phenomena pipeline (phenomena_tracking.py).
Centralising them avoids cross-module coupling and exposes previously
private helpers through a neutral import path.
"""

import numpy as np

from ms.constants import RAY_EXT_LENGTH as _RAY_EXT_LENGTH


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


_cone_trig_cache: dict = {}

def _cone_trig(angle_deg):
    """Return (half_rad, cos_thresh, cos_half, sin_half) cached by angle."""
    cached = _cone_trig_cache.get(angle_deg)
    if cached is not None:
        return cached
    half_rad = float(np.radians(angle_deg))
    cos_thresh = float(np.cos(half_rad))
    sin_half = float(np.sin(half_rad))
    _cone_trig_cache[angle_deg] = (half_rad, cos_thresh, cos_thresh, sin_half)
    return _cone_trig_cache[angle_deg]


def ray_hits_cone(origin, direction, x1, y1, x2, y2, cone_half_angle_deg: float,
                  ray_length: float = None) -> bool:
    """
    Return True if any part of the bounding box intersects the gaze cone
    defined by `origin`, unit `direction`, and the given half-angle.

    When *ray_length* is provided the cone is bounded to that distance
    from the origin (matching visual rendering).  When ``None`` the cone
    extends to ``RAY_EXT_LENGTH`` (backward-compatible default).

    Three exhaustive cases cover all bbox-cone intersections:
      1. The gaze origin is inside the bbox.
      2. Any corner of the bbox lies inside the cone (and within range).
      3. Either boundary ray of the cone intersects the bbox
         (catches the case where the cone straddles the bbox without any
         corner falling inside the cone).
    """
    _hr, cos_thresh, c, s = _cone_trig(cone_half_angle_deg)
    far = ray_length if ray_length is not None else _RAY_EXT_LENGTH

    ox, oy = origin[0], origin[1]
    dx_dir, dy_dir = direction[0], direction[1]

    if x1 <= ox <= x2 and y1 <= oy <= y2:
        return True

    # Check if any corner of the bbox lies inside the cone
    for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        tx, ty = px - ox, py - oy
        dist_sq = tx * tx + ty * ty
        if dist_sq < 1e-12:
            return True
        if dist_sq > far * far:
            continue
        dot = tx * dx_dir + ty * dy_dir
        if dot <= 0:
            continue
        # Compare dot/dist >= cos_thresh  =>  dot^2 >= cos_thresh^2 * dist_sq
        if dot * dot >= cos_thresh * cos_thresh * dist_sq:
            return True

    # Check if either boundary ray of the cone intersects the bbox
    left_dir  = np.array([ c * dx_dir - s * dy_dir,  s * dx_dir + c * dy_dir])
    right_dir = np.array([ c * dx_dir + s * dy_dir, -s * dx_dir + c * dy_dir])
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


def bbox_diagonal(obj) -> float:
    """Diagonal length of a detection bounding-box in pixels."""
    return float(np.sqrt((obj['x2'] - obj['x1'])**2 +
                         (obj['y2'] - obj['y1'])**2))
