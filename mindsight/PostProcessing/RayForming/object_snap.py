"""
RayForming/object_snap.py — Consolidated adaptive snap, tip snap, and smooth snap.

Moved from ``GazeTracking/gaze_processing.py``.  Contains the unified snap
scoring function, temporal hysteresis state, EMA smoothing, and tip-snapping
between gaze rays.
"""
from __future__ import annotations

import numpy as np

from mindsight.utils.geometry import (
    bbox_center,
    bbox_diagonal,
    pitch_yaw_to_2d,
    ray_hits_box,
    sample_depth_patch,
)


# ══════════════════════════════════════════════════════════════════════════════
# Smooth snap tracker
# ══════════════════════════════════════════════════════════════════════════════

class SmoothSnapTracker:
    """EMA-based smoother for adaptive-snap endpoints.

    Instead of jumping instantly to a snap target, smoothly interpolates the
    ray endpoint toward the target over multiple frames.  State is maintained
    per face track ID.

    Parameters
    ----------
    alpha : float
        EMA rate -- lower values produce smoother/slower transitions,
        higher values are more responsive.  Default 0.20.
    """

    def __init__(self, alpha: float = 0.20):
        self.alpha = alpha
        self._state: dict[int, np.ndarray] = {}

    def update(self, tid: int, endpoint: np.ndarray) -> np.ndarray:
        """Return the smoothed endpoint for track *tid*."""
        endpoint = np.asarray(endpoint, float)
        prev = self._state.get(tid)
        if prev is not None:
            smoothed = prev + self.alpha * (endpoint - prev)
        else:
            smoothed = endpoint.copy()
        self._state[tid] = smoothed.copy()
        return smoothed

    def prune(self, active_tids: set):
        """Remove state for tracks no longer present."""
        for tid in list(self._state):
            if tid not in active_tids:
                del self._state[tid]


# ══════════════════════════════════════════════════════════════════════════════
# Snap temporal state
# ══════════════════════════════════════════════════════════════════════════════

class SnapTemporalState:
    """Lightweight per-face state for temporal consistency in snap scoring.

    Tracks the previous snap target per face and handles the release countdown
    (hold the last snap for a few frames when no new match is found) and
    optional engage delay (require consistent match before snapping for the
    first time).

    Parameters
    ----------
    release_frames : int
        Frames of consecutive no-match before releasing the held snap.
    engage_frames : int
        Frames of consistent match required before engaging snap for the
        first time (0 = instant engage).
    grid_px : int
        Grid cell size for quantising object centres to stable keys.
    """

    def __init__(self, release_frames: int = 5, engage_frames: int = 0,
                 grid_px: int = 40):
        self.release_frames = release_frames
        self.engage_frames = engage_frames
        self.grid_px = grid_px
        self._state: dict = {}

    def key_for(self, center) -> tuple:
        """Grid-quantised key for an object centre."""
        g = self.grid_px
        return (int(center[0]) // g, int(center[1]) // g)

    def prev_target_key(self, face_idx: int):
        """Return the grid key of the previous snap target (or None)."""
        s = self._state.get(face_idx)
        return s['snap_key'] if s else None

    def update(self, face_idx: int, snap_center, found: bool,
               gaze_conf: float = None) -> tuple:
        """Update temporal state after scoring and return the filtered result.

        Parameters
        ----------
        face_idx    : stable face track ID
        snap_center : np.array centre of best candidate (or fallback)
        found       : True if snap_score found a quality-passing match
        gaze_conf   : 0-1 confidence; low values accelerate release

        Returns
        -------
        (filtered_center_or_None, did_snap: bool)
        """
        s = self._state.setdefault(face_idx, dict(
            snap_key=None, snap_ctr=None,
            no_match=0, match_key=None, match_n=0,
        ))

        if gaze_conf is not None:
            gc = max(0.0, min(1.0, gaze_conf))
            eff_release = max(1, int(self.release_frames * (0.3 + 0.7 * gc)))
        else:
            eff_release = self.release_frames

        if not found or snap_center is None:
            s['match_key'] = None
            s['match_n'] = 0
            if s['snap_key'] is not None:
                s['no_match'] += 1
                if s['no_match'] >= eff_release:
                    s['snap_key'] = s['snap_ctr'] = None
                    s['no_match'] = 0
            return s['snap_ctr'], s['snap_ctr'] is not None

        s['no_match'] = 0
        nk = self.key_for(snap_center)

        if nk == s['snap_key']:
            s['snap_ctr'] = snap_center
            s['match_key'] = None
            s['match_n'] = 0
            return snap_center, True

        if self.engage_frames > 0 and s['snap_key'] is None:
            if nk == s['match_key']:
                s['match_n'] += 1
            else:
                s['match_key'] = nk
                s['match_n'] = 1
            if s['match_n'] >= self.engage_frames:
                s['snap_key'] = nk
                s['snap_ctr'] = snap_center
                s['match_key'] = None
                s['match_n'] = 0
                return snap_center, True
            return None, False

        s['snap_key'] = nk
        s['snap_ctr'] = snap_center
        s['match_key'] = None
        s['match_n'] = 0
        return snap_center, True


# ══════════════════════════════════════════════════════════════════════════════
# Snap scoring
# ══════════════════════════════════════════════════════════════════════════════

def snap_score(origin, direction, objects, fallback, *,
               snap_dist=150.0, gaze_conf=None, bbox_scale=0.0,
               w_dist=1.0, w_angle=0.8, w_size=0.0,
               w_intersect=0.5, w_temporal=0.3,
               gate_angle_deg=60.0, head_blend=0.3,
               quality_thresh=0.8, face_center=None,
               prev_target_key=None, frame_diag=None,
               _key_fn=None,
               depth_map=None, w_depth=0.0, gaze_endpoint=None,
               gaze_sample_radius=2):
    """Unified snap scoring with angular plausibility and temporal consistency.

    Scores each candidate object using a weighted combination of:

    - **Distance** (``w_dist``): normalised perpendicular distance to bbox boundary.
    - **Angle** (``w_angle``): quadratic penalty based on angular deviation from a
      blended gaze+head direction.
    - **Size** (``w_size``): reward for larger objects (subtracted from score).
    - **Intersection** (``w_intersect``): bonus when ray passes through bbox.
    - **Temporal** (``w_temporal``): bonus for the previous frame's snap target.

    Eligibility gates (all must pass):
      1. ``t > 0`` -- object in front of face
      2. ``min_dist <= eff_snap_dist`` -- within pixel distance
      3. ``angle <= gate_angle`` -- hard angular cutoff

    Quality gate: best score must be < ``quality_thresh``, otherwise no match.

    Returns ``(object_centre, found_bool, matched_obj, score)``.
    """
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float)

    if gaze_conf is not None:
        snap_dist = snap_dist * (0.2 + 0.8 * max(0.0, min(1.0, gaze_conf)))

    gate_rad = np.radians(gate_angle_deg)
    cos_gate = np.cos(gate_rad)

    if face_center is not None:
        face_center = np.asarray(face_center, float)

    best_score, best_ctr, best_obj = float('inf'), None, None
    for obj in objects:
        ctr = bbox_center(obj)
        diag = bbox_diagonal(obj)
        half_diag = diag / 2.0

        t = float(np.dot(ctr - origin, direction))
        if t <= 0:
            continue

        proj_pt = origin + t * direction
        perp = float(np.linalg.norm(ctr - proj_pt))

        far_end = origin + direction * (t + diag)
        hits = ray_hits_box(origin, far_end,
                            obj['x1'], obj['y1'], obj['x2'], obj['y2'])
        min_dist = 0.0 if hits else max(0.0, perp - half_diag)

        eff_snap = snap_dist + half_diag * bbox_scale
        if min_dist > eff_snap:
            continue

        d_to_obj = ctr - origin
        d_to_obj_norm = np.linalg.norm(d_to_obj)
        if d_to_obj_norm > 1e-6:
            d_to_obj = d_to_obj / d_to_obj_norm
        else:
            continue

        if face_center is not None and head_blend > 0:
            d_head = ctr - face_center
            dh_norm = np.linalg.norm(d_head)
            if dh_norm > 1e-6:
                d_head = d_head / dh_norm
                d_blend = (1.0 - head_blend) * direction + head_blend * d_head
                db_norm = np.linalg.norm(d_blend)
                d_blend = d_blend / db_norm if db_norm > 1e-6 else direction
            else:
                d_blend = direction
        else:
            d_blend = direction

        cos_angle = float(np.dot(d_blend, d_to_obj))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        if cos_angle < cos_gate:
            continue

        angle = np.arccos(cos_angle)

        d_factor = min_dist / eff_snap if eff_snap > 0 else 0.0
        a_factor = (angle / gate_rad) ** 2 if gate_rad > 0 else 0.0
        i_factor = 1.0 if hits else 0.0
        s_factor = (diag / frame_diag) if frame_diag and frame_diag > 0 else 0.0

        t_factor = 0.0
        if prev_target_key is not None and _key_fn is not None:
            if _key_fn(ctr) == prev_target_key:
                t_factor = 1.0

        depth_factor = 0.0
        if depth_map is not None and w_depth > 0 and gaze_endpoint is not None:
            d_range = float(depth_map.max() - depth_map.min())
            if d_range > 1e-4:
                d_at_gaze = sample_depth_patch(
                    depth_map, gaze_endpoint[0], gaze_endpoint[1],
                    radius=gaze_sample_radius)
                obj_depth = obj.get('depth_median', 0.5)
                depth_factor = abs(d_at_gaze - obj_depth) / d_range

        score = (w_dist * d_factor + w_angle * a_factor
                 + w_depth * depth_factor
                 - w_size * s_factor - w_intersect * i_factor
                 - w_temporal * t_factor)

        if score < best_score:
            best_score, best_ctr, best_obj = score, ctr, obj

    if best_ctr is not None and best_score < quality_thresh:
        return best_ctr, True, best_obj, best_score
    return np.asarray(fallback, float), False, None, float('inf')


# ══════════════════════════════════════════════════════════════════════════════
# Tip snapping
# ══════════════════════════════════════════════════════════════════════════════

def apply_tip_snapping(persons_gaze, ray_snapped, ray_extended, gaze_eng,
                       gaze_cfg, *, face_track_ids=None,
                       smooth_snap_tracker=None):
    """Apply tip-snapping between gaze rays for per-face backends.

    Snaps each unsnapped ray endpoint to the tip bounding-box of another
    person's ray when the rays converge within ``gaze_cfg.snap_dist``.

    Parameters
    ----------
    persons_gaze  : list of (origin, ray_end, angles)
    ray_snapped   : list[bool] per-person snap flag
    ray_extended  : list[bool] per-person extension flag
    gaze_eng      : active gaze backend (checked for ``mode`` attribute)
    gaze_cfg      : config with gaze_tips, snap_mode/adaptive_ray, tip_radius, snap_dist
    face_track_ids : list[int] or None -- stable track IDs for smooth snap keying
    smooth_snap_tracker : SmoothSnapTracker or None

    Returns
    -------
    persons_gaze, ray_snapped, ray_extended  (updated lists)
    """
    is_per_face = getattr(gaze_eng, "mode", "per_face") == "per_face"
    # Support both legacy GazeConfig (adaptive_ray) and RayFormingConfig (snap_mode)
    snap_mode = getattr(gaze_cfg, 'snap_mode', getattr(gaze_cfg, 'adaptive_ray', 'off'))
    tip_radius = getattr(gaze_cfg, 'tip_radius', 80)
    gaze_tips = getattr(gaze_cfg, 'gaze_tips', False)
    if not (gaze_tips and snap_mode != "off"
            and is_per_face and len(persons_gaze) > 1):
        return persons_gaze, ray_snapped, ray_extended

    smooth_snap_str = getattr(gaze_cfg, 'smooth_snap', 'off')
    do_smooth = (smooth_snap_tracker is not None
                 and smooth_snap_str in ("gaze_tips", "all"))

    tip_dist = getattr(gaze_cfg, 'snap_tip_dist', -1.0)
    if tip_dist < 0:
        tip_dist = getattr(gaze_cfg, 'snap_dist', 150.0)
    tip_quality = getattr(gaze_cfg, 'snap_tip_quality', -1.0)
    if tip_quality < 0:
        tip_quality = getattr(gaze_cfg, 'snap_quality_thresh', 0.8)

    tips = [{'x1': re[0] - tip_radius, 'y1': re[1] - tip_radius,
             'x2': re[0] + tip_radius, 'y2': re[1] + tip_radius, '_owner': fi}
            for fi, (_, re, _) in enumerate(persons_gaze)]
    new_gaze, new_snap, new_ext = [], [], []
    for fi, (ori, re, ang) in enumerate(persons_gaze):
        if ray_snapped[fi] or ray_extended[fi]:
            new_gaze.append((ori, re, ang))
            new_snap.append(ray_snapped[fi])
            new_ext.append(ray_extended[fi])
            continue
        d = pitch_yaw_to_2d(*ang)
        tip_ctr, sn, _, _ = snap_score(
            ori, d, [t for t in tips if t['_owner'] != fi], re,
            snap_dist=tip_dist,
            bbox_scale=getattr(gaze_cfg, 'snap_bbox_scale', 0.0),
            w_dist=getattr(gaze_cfg, 'snap_w_dist', 1.0),
            w_angle=getattr(gaze_cfg, 'snap_w_angle', 0.8),
            w_size=getattr(gaze_cfg, 'snap_w_size', 0.0),
            w_intersect=getattr(gaze_cfg, 'snap_w_intersect', 0.5),
            w_temporal=0.0,
            gate_angle_deg=getattr(gaze_cfg, 'snap_gate_angle', 60.0),
            head_blend=getattr(gaze_cfg, 'snap_head_blend', 0.3),
            quality_thresh=tip_quality,
            face_center=ori)
        if sn:
            if snap_mode == "snap":
                ne, s, e = tip_ctr, True, False
            else:
                t_proj = float(np.dot(tip_ctr - ori, d))
                ne, s, e = ((ori + d * t_proj), False, True) if t_proj > 0 else (re, False, False)
        else:
            ne, s, e = re, False, False
        if do_smooth:
            tid = face_track_ids[fi] if face_track_ids else fi
            ne = smooth_snap_tracker.update(tid, ne)
        new_gaze.append((ori, ne, ang))
        new_snap.append(s)
        new_ext.append(e)
    return new_gaze, new_snap, new_ext


# ══════════════════════════════════════════════════════════════════════════════
# Object snap orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class ObjectSnap:
    """Manages all snap state (temporal, smooth) for a pipeline instance.

    Wraps snap_score, SnapTemporalState, and SmoothSnapTracker into a single
    callable interface.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.temporal = SnapTemporalState(
            release_frames=cfg.snap_release_frames,
            engage_frames=cfg.snap_engage_frames,
        ) if cfg.snap_release_frames > 0 or cfg.snap_engage_frames > 0 else None
        self.smooth = SmoothSnapTracker(alpha=cfg.smooth_snap_alpha)

    def snap_ray(self, origin, endpoint, direction, gaze_conf, face_width,
                 objects, face_objs, track_id, fi_loc, bbox, frame_diag,
                 depth_map=None, smooth_mode="off"):
        """Run snap scoring and return (final_endpoint, snapped, extended)."""
        cfg = self.cfg
        snap_mode = cfg.snap_mode
        c = np.asarray(origin, float)
        d = np.asarray(direction, float)
        fb = np.asarray(endpoint, float)

        # Determine adaptive targets based on obj_snap_targets config
        other_faces = [fo for fo in face_objs if fo.get('_face_idx') != fi_loc]
        if cfg.obj_snap_targets == "off":
            adaptive_targets = []
        elif cfg.obj_snap_targets == "faces_only":
            adaptive_targets = other_faces
        else:
            adaptive_targets = list(objects) + other_faces

        snap, extended = False, False
        end = fb

        if snap_mode != "off" and adaptive_targets:
            bx1, by1, bx2, by2 = bbox
            face_ctr = np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])

            prev_key = None
            key_fn = None
            if self.temporal is not None:
                prev_key = self.temporal.prev_target_key(track_id)
                key_fn = self.temporal.key_for

            _w_depth = (cfg.snap_w_depth
                        if cfg.depth_aware_scoring and depth_map is not None
                        else 0.0)

            raw_ctr, raw_found, _, _ = snap_score(
                c, d, adaptive_targets, fb,
                snap_dist=cfg.snap_dist,
                gaze_conf=gaze_conf,
                bbox_scale=cfg.snap_bbox_scale,
                w_dist=cfg.snap_w_dist,
                w_angle=cfg.snap_w_angle,
                w_size=cfg.snap_w_size,
                w_intersect=cfg.snap_w_intersect,
                w_temporal=cfg.snap_w_temporal,
                gate_angle_deg=cfg.snap_gate_angle,
                head_blend=cfg.snap_head_blend,
                quality_thresh=cfg.snap_quality_thresh,
                face_center=face_ctr,
                prev_target_key=prev_key,
                frame_diag=frame_diag,
                _key_fn=key_fn,
                depth_map=depth_map,
                w_depth=_w_depth,
                gaze_endpoint=fb,
                gaze_sample_radius=cfg.gaze_sample_radius)

            if self.temporal is not None:
                obj_ctr, did_snap = self.temporal.update(
                    track_id, raw_ctr, raw_found, gaze_conf=gaze_conf)
                if obj_ctr is None or not did_snap:
                    end = fb
                elif snap_mode == "snap":
                    end, snap = obj_ctr, True
                else:
                    t = float(np.dot(obj_ctr - c, d))
                    end, extended = ((c + d * t), True) if t > 0 else (fb, False)
            else:
                if raw_found:
                    if snap_mode == "snap":
                        end, snap = raw_ctr, True
                    else:
                        t = float(np.dot(raw_ctr - c, d))
                        end, extended = ((c + d * t), True) if t > 0 else (fb, False)
                else:
                    end = fb
        else:
            if self.temporal is not None:
                self.temporal.update(track_id, None, False, gaze_conf=gaze_conf)

        # Smooth snap (objects)
        if smooth_mode in ("objects", "all"):
            end = self.smooth.update(track_id, end)

        return end, snap, extended
