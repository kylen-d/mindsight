"""
GazeTracking/gaze_processing.py — Generic gaze utilities shared across all backends.

Contains backend-agnostic tools for temporal smoothing, track re-identification,
fixation lock-on, snap hysteresis, adaptive-snap, ray geometry, and eye-landmark
extraction.  Also provides the three coordinator-level post-processing steps
(tip-snapping, lock-on, ray–bbox intersection) used by gaze_pipeline.py.

Plugins can extend this module's functionality by subclassing ``GazeToolkit``
and overriding or adding methods.  The coordinator passes the active toolkit
instance to the pipeline so plugin-specific tools are available alongside the
generic ones.

Example — plugin extending the toolkit::

    from GazeTracking.gaze_processing import GazeToolkit

    class MyToolkit(GazeToolkit):
        def my_custom_filter(self, persons_gaze, ...):
            ...
"""
from __future__ import annotations

import cv2
import numpy as np

from ms.constants import SMOOTH_ALPHA
from ms.utils.geometry import (
    bbox_center,
    bbox_diagonal,
    pitch_yaw_to_2d,
    ray_hits_box,
    ray_hits_cone,
)

# ══════════════════════════════════════════════════════════════════════════════
# Eye-landmark helper
# ══════════════════════════════════════════════════════════════════════════════

def _get_eye_center(face_dict, inv_scale=1.0):
    """
    Extract the midpoint of the two eye centres from RetinaFace keypoints.

    RetinaFace (via uniface) returns 5 keypoints in 'kps' as [[x,y], ...] with
    order: left_eye, right_eye, nose, left_mouth, right_mouth.

    Returns a float numpy array (x, y) scaled by inv_scale, or None if the
    keypoints are not present / cannot be parsed.
    """
    kps = face_dict.get("kps")
    if kps is not None and len(kps) >= 2:
        try:
            le = np.array(kps[0][:2], float) * inv_scale
            re = np.array(kps[1][:2], float) * inv_scale
            return (le + re) / 2.0
        except (IndexError, TypeError):
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Temporal smoothing & track re-identification
# ══════════════════════════════════════════════════════════════════════════════

class GazeSmootherReID:
    """Per-face temporal EMA smoother with colour histogram re-identification.

    Tracks detected faces across frames using positional distance weighted by
    face-crop colour histogram similarity.  When two faces are at similar
    distances from a track, the histogram similarity breaks the tie, preventing
    track-ID swaps when people cross paths.

    Match score: ``positional_distance * (1 + hist_weight * bhattacharyya_dist)``

    ``hist_weight=0`` disables histogram matching and falls back to
    positional-only behaviour.

    ``grace_frames > 0`` enables a re-identification grace period: when a track
    goes unmatched it is held in a dead-track buffer for that many frames.
    If a new detection appears within max_dist of a dead track before the
    buffer expires, the original track ID is revived instead of minting a
    new one.  This keeps person IDs stable across brief face occlusions.
    """

    def __init__(self, alpha=SMOOTH_ALPHA, max_dist=200, hist_weight=0.35,
                 hist_bins=16, grace_frames=0):
        self.alpha, self.max_dist = alpha, max_dist
        self.hist_weight          = hist_weight
        self.hist_bins            = hist_bins
        self.grace_frames         = grace_frames
        self._tracks, self._nid   = {}, 0
        self._dead: dict          = {}   # tid -> {c, p, y, h, _dropped}
        self._frame: int          = 0

    # -- Histogram helpers ----------------------------------------------------

    @staticmethod
    def _histogram(crop, bins):
        if crop is None or crop.size == 0:
            return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.ndim == 3 else crop
        h_h  = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten().astype(float)
        h_s  = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten().astype(float)
        hist = np.concatenate([h_h, h_s])
        s    = hist.sum()
        return hist / s if s > 0 else hist

    @staticmethod
    def _bhattacharyya(h1, h2):
        if h1 is None or h2 is None:
            return 0.0
        bc = float(np.sum(np.sqrt(np.clip(h1 * h2, 0, None))))
        return float(-np.log(bc + 1e-10))

    # -- Scoring & track management -------------------------------------------

    def _score(self, center, crop, track) -> float:
        """Return a match score (lower = better) between a detection and a track."""
        pos_d = float(np.linalg.norm(center - track['c']))
        new_hist = self._histogram(crop, self.hist_bins)
        hd = self._bhattacharyya(new_hist, track.get('h'))
        return pos_d * (1.0 + self.hist_weight * min(hd, 5.0))

    def _update_track(self, track, center, pitch, yaw, crop):
        """Update an existing track with smoothed values + histogram EMA."""
        sp = self.alpha * pitch + (1 - self.alpha) * track['p']
        sy = self.alpha * yaw   + (1 - self.alpha) * track['y']
        track.update(c=center, p=sp, y=sy)
        new_hist = self._histogram(crop, self.hist_bins)
        old_h = track.get('h')
        track['h'] = (0.7 * old_h + 0.3 * new_hist
                       if old_h is not None and new_hist is not None else new_hist)
        return sp, sy

    def _new_track(self, center, pitch, yaw, crop) -> dict:
        """Create the state dict for a brand-new track."""
        return dict(c=center, p=pitch, y=yaw,
                    h=self._histogram(crop, self.hist_bins))

    # -- Global-motion compensation -------------------------------------------

    def _estimate_global_shift(self, current_centers):
        """Median displacement between current detections and live tracks.

        When the camera moves, all faces shift by roughly the same amount.
        Subtracting this from current centers before matching prevents the
        positional distance from exceeding max_dist due to camera jitter.
        """
        if len(current_centers) < 2 or len(self._tracks) < 2:
            return np.zeros(2)
        disps = []
        for cc in current_centers:
            best_d, best_disp = float('inf'), np.zeros(2)
            for t in self._tracks.values():
                disp = cc - t['c']
                d = float(np.linalg.norm(disp))
                if d < best_d:
                    best_d, best_disp = d, disp
            disps.append(best_disp)
        return np.median(disps, axis=0)

    # -- Main update loop -----------------------------------------------------

    def _best_match(self, center, crop, pool):
        """Find the best-scoring track in *pool* within max_dist."""
        best_id, best_score = None, float('inf')
        for tid, t in pool.items():
            s = self._score(center, crop, t)
            if s < self.max_dist and s < best_score:
                best_score, best_id = s, tid
        return best_id

    def update(self, faces):
        """
        faces: [(center, pitch, yaw, crop)]
        Returns: [(smooth_pitch, smooth_yaw, track_id)]
        """
        self._frame += 1

        # Expire stale dead tracks first
        if self.grace_frames > 0:
            expired = [tid for tid, t in self._dead.items()
                       if self._frame - t['_dropped'] > self.grace_frames]
            for tid in expired:
                del self._dead[tid]

        # Compensate for global camera motion before matching
        raw_centers = [np.asarray(entry[0], float) for entry in faces]
        shift = self._estimate_global_shift(raw_centers)

        unmatched, result = set(self._tracks), []
        for entry in faces:
            center, pitch, yaw = entry[0], entry[1], entry[2]
            crop = entry[3] if len(entry) > 3 else None
            c_raw = np.asarray(center, float)
            c_match = c_raw - shift  # shift-compensated for matching only

            # 1. Try to match a live track
            bid = self._best_match(c_match, crop, self._tracks)
            if bid is not None:
                sp, sy = self._update_track(self._tracks[bid], c_raw, pitch, yaw, crop)
                unmatched.discard(bid)
                result.append((sp, sy, bid))
                continue

            # 2. No live match — try to revive a dead track (grace period)
            if self.grace_frames > 0 and self._dead:
                did = self._best_match(c_match, crop, self._dead)
                if did is not None:
                    revived = self._dead.pop(did)
                    sp, sy = self._update_track(revived, c_raw, pitch, yaw, crop)
                    self._tracks[did] = revived
                    result.append((sp, sy, did))
                    continue

            # 3. Genuinely new face — allocate a fresh ID
            nid = self._nid; self._nid += 1
            self._tracks[nid] = self._new_track(c_raw, pitch, yaw, crop)
            result.append((pitch, yaw, nid))

        # Move unmatched live tracks to the dead-track buffer (or drop immediately)
        for tid in unmatched:
            if self.grace_frames > 0:
                self._dead[tid] = {**self._tracks[tid], '_dropped': self._frame}
            del self._tracks[tid]
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Snap hysteresis
# ══════════════════════════════════════════════════════════════════════════════

class SnapHysteresisTracker:
    """
    Prevents rapid flipping between adaptive-snap targets when raw gaze is jittery.

    Once a face latches onto an object, a different object must be the nearest-valid
    snap target for `switch_frames` consecutive frames before the snap changes.
    Objects are identified by quantising their centre to a `grid_px`-pixel grid so
    minor bounding-box jitter does not count as a target change.
    When no valid snap target exists, the current snap is held for `release_frames`
    before being released.
    """

    def __init__(self, switch_frames: int = 8, release_frames: int = 5, grid_px: int = 40):
        self.switch_frames  = switch_frames
        self.release_frames = release_frames
        self.grid_px        = grid_px
        self._state: dict   = {}

    def _key(self, center):
        g = self.grid_px
        return (int(center[0]) // g, int(center[1]) // g)

    def update(self, face_idx: int, raw_snap_center, raw_snap: bool,
               gaze_conf: float = None) -> tuple:
        """
        face_idx       : stable integer face index within this frame batch
        raw_snap_center: np.array centre of snap candidate, or None / fallback
        raw_snap       : True if adaptive_snap found a valid candidate
        gaze_conf      : 0-1 gaze confidence; low values accelerate release

        Returns (filtered_snap_center_or_None, did_snap: bool).
        """
        s = self._state.setdefault(face_idx, dict(
            snap_key=None, snap_ctr=None,
            cand_key=None, cand_ctr=None, cand_n=0, release_n=0,
        ))

        # Low confidence → faster release (fewer frames needed to detach).
        # High confidence → hold the snap longer.
        if gaze_conf is not None:
            gc = max(0.0, min(1.0, gaze_conf))
            eff_release = max(1, int(self.release_frames * (0.3 + 0.7 * gc)))
        else:
            eff_release = self.release_frames

        if not raw_snap or raw_snap_center is None:
            s['cand_key'] = s['cand_ctr'] = None; s['cand_n'] = 0
            if s['snap_key'] is not None:
                s['release_n'] += 1
                if s['release_n'] >= eff_release:
                    s['snap_key'] = s['snap_ctr'] = None; s['release_n'] = 0
            return s['snap_ctr'], s['snap_ctr'] is not None

        s['release_n'] = 0
        nk = self._key(raw_snap_center)

        if s['snap_key'] is None:
            s['snap_key'] = nk; s['snap_ctr'] = raw_snap_center
            s['cand_key'] = None; s['cand_n'] = 0
            return raw_snap_center, True

        if nk == s['snap_key']:
            s['snap_ctr'] = raw_snap_center
            s['cand_key'] = None; s['cand_n'] = 0
            return raw_snap_center, True

        if nk == s['cand_key']:
            s['cand_n'] += 1
            s['cand_ctr'] = raw_snap_center
        else:
            s['cand_key'] = nk; s['cand_ctr'] = raw_snap_center; s['cand_n'] = 1

        if s['cand_n'] >= self.switch_frames:
            s['snap_key'] = s['cand_key']; s['snap_ctr'] = s['cand_ctr']
            s['cand_key'] = None; s['cand_n'] = 0

        return s['snap_ctr'], s['snap_ctr'] is not None


# ══════════════════════════════════════════════════════════════════════════════
# Fixation lock-on
# ══════════════════════════════════════════════════════════════════════════════

class GazeLockTracker:
    """Fixation lock-on: snaps gaze ray to object after dwell_frames of sustained attention."""

    def __init__(self, dwell_frames=15, release_frames=10, lock_dist=100, max_face_dist=120):
        self.dwell, self.release           = dwell_frames, release_frames
        self.lock_dist, self.max_face_dist = lock_dist, max_face_dist
        self._tracks, self._nid            = {}, 0

    @staticmethod
    def _ray_pt_dist(origin, udir, pt):
        v = pt - origin
        t = float(np.dot(v, udir))
        return float(np.linalg.norm(v if t < 0 else pt - (origin + t * udir)))

    def _find_track(self, center):
        if not self._tracks:
            return None
        bid, bd = min(
            ((tid, float(np.linalg.norm(center - t['c']))) for tid, t in self._tracks.items()),
            key=lambda x: x[1])
        return bid if bd < self.max_face_dist else None

    def update(self, persons_gaze, objects):
        used, results = set(), []
        for origin, ray_end, _ in persons_gaze:
            c    = np.asarray(origin, float)
            dv   = np.asarray(ray_end, float) - c
            dl   = np.linalg.norm(dv)
            udir = dv / dl if dl > 1e-6 else np.array([0., 1.])

            tid = self._find_track(c)
            if tid is None:
                tid = self._nid; self._nid += 1
                self._tracks[tid] = dict(c=c.copy(), dwell={}, locked=None, rc=0)
            t = self._tracks[tid]
            t['c'] = c.copy()
            used.add(tid)

            obj_ctrs = [bbox_center(o) for o in objects]
            near = {oi for oi, ctr in enumerate(obj_ctrs)
                    if self._ray_pt_dist(c, udir, ctr) < self.lock_dist}

            for oi in list(t['dwell']):
                t['dwell'][oi] = t['dwell'][oi] + 1 if oi in near else max(0, t['dwell'][oi] - 2)
                if t['dwell'][oi] == 0:
                    del t['dwell'][oi]
            for oi in near - set(t['dwell']):
                t['dwell'][oi] = 1

            locked = t['locked']
            if locked is not None:
                if locked in near:
                    t['rc'] = 0
                else:
                    t['rc'] += 1
                    if t['rc'] >= self.release:
                        t['locked'] = None; t['rc'] = 0; locked = None

            if locked is None:
                best = max(
                    ((oi, cnt) for oi, cnt in t['dwell'].items() if cnt >= self.dwell),
                    key=lambda x: x[1], default=(None, 0))[0]
                if best is not None:
                    t['locked'] = locked = best; t['rc'] = 0

            frac = min(max(t['dwell'].values(), default=0) / self.dwell, 1.0)
            if locked is not None and locked < len(objects):
                obj     = objects[locked]
                snapped = bbox_center(obj)
                results.append((snapped, locked, frac))
            else:
                results.append((np.asarray(ray_end, float), None, frac))

        for tid in list(self._tracks):
            if tid not in used:
                del self._tracks[tid]
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive snap & face-as-object helpers
# ══════════════════════════════════════════════════════════════════════════════

def _faces_as_objects(face_bboxes_list):
    """Convert a list of (x1,y1,x2,y2) face bboxes to Detection objects so they
    can be treated as gaze targets by adaptive_snap and the hit-detection loop."""
    from ObjectDetection.detection import Detection
    return [Detection(class_name='face', cls_id=-1, conf=1.0,
                      x1=x1, y1=y1, x2=x2, y2=y2, _face_idx=fi)
            for fi, (x1, y1, x2, y2) in enumerate(face_bboxes_list)]


def adaptive_snap(origin, direction, objects, fallback, snap_dist=150,
                  gaze_conf=None, bbox_scale=0.0,
                  w_dist=1.0, w_size=0.0, w_intersect=0.5):
    """Find the best-matching object for the gaze ray.

    Uses a two-tier selection:

    **Tier 1 — Intersected objects** (ray passes through the bounding box):
    These always beat non-intersected objects.  Among intersected objects the
    closest along the ray direction (minimum *t*) wins — the participant is
    most likely looking at the first thing in their line of sight.

    **Tier 2 — Nearby objects** (ray passes within *snap_dist* of the bbox
    boundary): ranked by perpendicular distance to the bbox boundary.

    When *gaze_conf* (0-1) is provided the base snap radius is scaled by
    confidence so high-confidence gaze snaps more readily.

    Returns (object_centre, found_bool, matched_obj_or_None).  The object
    centre is always returned (not the projected point) so
    SnapHysteresisTracker has a stable coordinate to key on.
    """
    if gaze_conf is not None:
        # Scale: conf 1.0 → full snap_dist, conf 0.0 → 20% of snap_dist
        snap_dist = snap_dist * (0.2 + 0.8 * max(0.0, min(1.0, gaze_conf)))

    best_score, best_ctr, best_obj = float('inf'), None, None
    for obj in objects:
        ctr  = bbox_center(obj)
        diag = bbox_diagonal(obj)
        half_diag = diag / 2.0

        # Parametric projection onto ray
        t = float(np.dot(ctr - origin, direction))
        if t <= 0:
            continue

        # Perpendicular distance from ray to centre
        perp = float(np.linalg.norm(ctr - (origin + t * direction)))

        # Does the ray intersect the bounding box?
        far_end = origin + direction * (t + diag)
        hits = ray_hits_box(origin, far_end,
                            obj['x1'], obj['y1'], obj['x2'], obj['y2'])

        # Distance from ray to bbox boundary (0 when ray intersects)
        min_dist = 0.0 if hits else max(0.0, perp - half_diag)

        # Eligibility gate
        eff_snap = snap_dist + half_diag * bbox_scale
        if min_dist > eff_snap:
            continue

        # Two-tier scoring (lower is better):
        #   Tier 1 (hits):     score in [-1e6, 0)  — sorted by t (closest first)
        #   Tier 2 (no hits):  score in [0, +inf)   — sorted by min_dist
        if hits:
            score = -1e6 + t
        else:
            score = min_dist

        if score < best_score:
            best_score, best_ctr, best_obj = score, ctr, obj

    if best_ctr is not None:
        return best_ctr, True, best_obj
    return np.asarray(fallback, float), False, None


# ══════════════════════════════════════════════════════════════════════════════
# Coordinator-level post-processing steps
# ══════════════════════════════════════════════════════════════════════════════

def apply_tip_snapping(persons_gaze, ray_snapped, ray_extended, gaze_eng, gaze_cfg):
    """Apply tip-snapping between gaze rays for per-face backends.

    Snaps each unsnapped ray endpoint to the tip bounding-box of another
    person's ray when the rays converge within ``gaze_cfg.snap_dist``.

    Parameters
    ----------
    persons_gaze  : list of (origin, ray_end, angles)
    ray_snapped   : list[bool] per-person snap flag
    ray_extended  : list[bool] per-person extension flag
    gaze_eng      : active gaze backend (checked for ``mode`` attribute)
    gaze_cfg      : GazeConfig with gaze_tips, adaptive_ray, tip_radius, snap_dist

    Returns
    -------
    persons_gaze, ray_snapped, ray_extended  (updated lists)
    """
    is_per_face = getattr(gaze_eng, "mode", "per_face") == "per_face"
    tip_radius  = gaze_cfg.tip_radius
    if not (gaze_cfg.gaze_tips and gaze_cfg.adaptive_ray != "off"
            and is_per_face and len(persons_gaze) > 1):
        return persons_gaze, ray_snapped, ray_extended

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
        tip_ctr, sn, _ = adaptive_snap(
            ori, d, [t for t in tips if t['_owner'] != fi], re,
            gaze_cfg.snap_dist,
            bbox_scale=gaze_cfg.snap_bbox_scale,
            w_dist=gaze_cfg.snap_w_dist,
            w_size=gaze_cfg.snap_w_size,
            w_intersect=gaze_cfg.snap_w_intersect)
        if sn:
            if gaze_cfg.adaptive_ray == "snap":
                ne, s, e = tip_ctr, True, False
            else:
                t_proj = float(np.dot(tip_ctr - ori, d))
                ne, s, e = ((ori + d * t_proj), False, True) if t_proj > 0 else (re, False, False)
        else:
            ne, s, e = re, False, False
        new_gaze.append((ori, ne, ang))
        new_snap.append(s)
        new_ext.append(e)
    return new_gaze, new_snap, new_ext


def apply_lock_on(persons_gaze, locker, objects):
    """Apply fixation lock-on and return updated persons_gaze and lock_info.

    Parameters
    ----------
    persons_gaze : list of (origin, ray_end, angles)
    locker       : GazeLockTracker instance or None
    objects      : non-person detection list

    Returns
    -------
    persons_gaze : list of (origin, ray_end, angles) — ray_end snapped when locked
    lock_info    : list of (obj_idx_or_None, frac)
    """
    lock_info = [(None, 0.0)] * len(persons_gaze)
    if locker and persons_gaze:
        lr = locker.update(persons_gaze, objects)
        persons_gaze = [(o, se, a) for (o, _, a), (se, _, _) in zip(persons_gaze, lr)]
        lock_info    = [(oi, frac) for (_, oi, frac) in lr]
    return persons_gaze, lock_info


def compute_ray_intersections(persons_gaze, face_confs, face_track_ids,
                              face_objs, objects, gaze_cfg):
    """Compute ray–bbox (or cone) intersections with confidence gating.

    Parameters
    ----------
    persons_gaze    : list of (origin, ray_end, angles)
    face_confs      : list[float] per-face gaze confidence
    face_track_ids  : list[int] stable track IDs (used in hit_events)
    face_objs       : list[Detection] face-as-object targets
    objects         : non-person detection list
    gaze_cfg        : GazeConfig with hit_conf_gate, detect_extend, gaze_cone_angle

    Returns
    -------
    all_targets : list[dict]  objects + face_objs
    hits        : set of (face_list_idx, target_idx) pairs
    hit_events  : list[dict] per-hit records with face_idx = track ID
    """
    hit_conf_gate   = gaze_cfg.hit_conf_gate
    scope           = gaze_cfg.detect_extend_scope
    detect_extend   = gaze_cfg.detect_extend if scope in ('objects', 'both') else 0.0
    gaze_cone_angle = gaze_cfg.gaze_cone_angle
    gaze_tips       = gaze_cfg.gaze_tips
    tip_radius      = gaze_cfg.tip_radius
    fwd_thresh_rad  = np.radians(gaze_cfg.forward_gaze_threshold)
    all_targets = objects + face_objs
    hits, hit_events = set(), []
    for fi, (origin, ray_end, angles) in enumerate(persons_gaze):
        if hit_conf_gate > 0.0 and fi < len(face_confs) and face_confs[fi] < hit_conf_gate:
            continue
        # Skip hit-detection for forward-gaze stub rays
        if fwd_thresh_rad > 0 and angles:
            if abs(angles[0]) < fwd_thresh_rad and abs(angles[1]) < fwd_thresh_rad:
                continue
        o_arr  = np.asarray(origin, float)
        re_arr = np.asarray(ray_end, float)
        dv     = re_arr - o_arr
        dl     = np.linalg.norm(dv)
        udir   = dv / dl if dl > 1e-6 else np.array([0., 1.])
        detect_range = dl + detect_extend          # visual length + optional extension

        # Detection endpoint for ray mode
        if detect_extend > 0:
            det_end = o_arr + udir * detect_range
        else:
            det_end = re_arr                       # exact visual endpoint

        for oi, obj in enumerate(all_targets):
            if obj.get('_face_idx') == fi:
                continue
            if gaze_cone_angle > 0.0:
                hit = ray_hits_cone(o_arr, udir,
                                    obj['x1'], obj['y1'], obj['x2'], obj['y2'],
                                    gaze_cone_angle, ray_length=detect_range)
            else:
                hit = ray_hits_box(o_arr, det_end,
                                   obj['x1'], obj['y1'], obj['x2'], obj['y2'])
            # Gaze-tip hit: object overlaps the tip circle at the ray endpoint
            if not hit and gaze_tips:
                cx = np.clip(re_arr[0], obj['x1'], obj['x2'])
                cy = np.clip(re_arr[1], obj['y1'], obj['y2'])
                hit = (cx - re_arr[0])**2 + (cy - re_arr[1])**2 <= tip_radius**2
            if hit:
                hits.add((fi, oi))
                hit_events.append(dict(
                    face_idx=face_track_ids[fi] if face_track_ids else fi,
                    object=obj['class_name'],
                    object_conf=obj['conf'],
                    bbox=(obj['x1'], obj['y1'], obj['x2'], obj['y2'])))
    return all_targets, hits, hit_events


# ══════════════════════════════════════════════════════════════════════════════
# Extensible toolkit for plugins
# ══════════════════════════════════════════════════════════════════════════════

class GazeToolkit:
    """
    Extensible collection of gaze processing tools.

    Plugins can subclass ``GazeToolkit`` to override default tool behaviour or
    add new plugin-specific tools.  The active toolkit instance is available
    to pipeline functions through the gaze engine.

    Example — custom smoothing in a plugin::

        from GazeTracking.gaze_processing import GazeToolkit, GazeSmootherReID

        class MyToolkit(GazeToolkit):
            def create_smoother(self, fps, grace_seconds=1.0):
                # Use a custom alpha for smoother temporal response
                grace = max(0, int(round(grace_seconds * fps)))
                return GazeSmootherReID(alpha=0.5, grace_frames=grace)
    """

    def create_smoother(self, fps: float, grace_seconds: float = 1.0) -> GazeSmootherReID:
        """Create a temporal smoother with default parameters."""
        grace = max(0, int(round(grace_seconds * fps)))
        return GazeSmootherReID(grace_frames=grace)

    def create_locker(self, dwell_frames: int = 15,
                      lock_dist: int = 100) -> GazeLockTracker:
        """Create a fixation lock-on tracker."""
        return GazeLockTracker(dwell_frames=dwell_frames, lock_dist=lock_dist)

    def create_snap_hysteresis(self, switch_frames: int = 8) -> SnapHysteresisTracker:
        """Create a snap hysteresis tracker."""
        return SnapHysteresisTracker(switch_frames=switch_frames)

    def get_eye_center(self, face_dict, inv_scale=1.0):
        """Extract eye centre from face keypoints (overridable)."""
        return _get_eye_center(face_dict, inv_scale)

    def compute_adaptive_snap(self, origin, direction, objects, fallback,
                              snap_dist=150, gaze_conf=None, **kwargs):
        """Compute adaptive snap (overridable)."""
        return adaptive_snap(origin, direction, objects, fallback,
                             snap_dist, gaze_conf, **kwargs)

    def faces_as_objects(self, face_bboxes_list):
        """Convert face bboxes to Detection objects (overridable)."""
        return _faces_as_objects(face_bboxes_list)


# ══════════════════════════════════════════════════════════════════════════════
# CLI argument registration (generic / backend-agnostic flags only)
# ══════════════════════════════════════════════════════════════════════════════

def add_arguments(parser) -> None:
    """Register generic gaze-tracking CLI flags onto *parser*.

    Backend-specific flags (e.g. ``--mgaze-model``, ``--mgaze-arch``) are
    registered by each plugin's ``add_arguments()`` method.
    """
    parser.add_argument("--ray-length", type=float, default=1.0,
                        help="Gaze ray-length multiplier, default 1.0")
    parser.add_argument("--conf-ray", action="store_true",
                        help="Dynamically adjust gaze ray-length based on gaze confidence value")
    parser.add_argument("--gaze-tips", action="store_true",
                        help="Adds circular bounding-box to tip of gaze-rays, used to determine "
                             "intersection between gaze-rays. Set radius with --tip-radius (default 80).")
    parser.add_argument("--tip-radius", type=int, default=80,
                        help="Pixel radius for --gaze-tips (default 80)")
    parser.add_argument("--adaptive-ray", type=str, default="off",
                        choices=["off", "extend", "snap"],
                        help="Adaptive ray mode: 'off' disables, 'extend' freely extends "
                             "the ray toward the nearest object, 'snap' locks the endpoint "
                             "to the object centre (default: off).")
    parser.add_argument("--snap-dist", type=float, default=150.0)
    parser.add_argument("--snap-bbox-scale", type=float, default=0.0,
                        help="Fraction of bbox half-diagonal added to snap radius (default 0.0)")
    parser.add_argument("--snap-w-dist", type=float, default=1.0,
                        help="Snap scoring weight for normalized distance (default 1.0)")
    parser.add_argument("--snap-w-size", type=float, default=0.0,
                        help="Snap scoring weight for angular size reward (default 0.0)")
    parser.add_argument("--snap-w-intersect", type=float, default=0.5,
                        help="Snap scoring bonus for ray-bbox intersection (default 0.5)")
    parser.add_argument("--hit-conf-gate", type=float, default=0.0, metavar="F",
                        help="Minimum per-face gaze confidence for ray-object hit detection. "
                             "0.0 = disabled (default).")
    parser.add_argument("--detect-extend", type=float, default=0.0, metavar="PX",
                        help="Extend gaze-object detection N pixels past the visual "
                             "ray/cone endpoint. 0 = detection matches visual exactly "
                             "(default: %(default)s).")
    parser.add_argument("--detect-extend-scope", type=str, default="objects",
                        choices=["objects", "phenomena", "both"],
                        help="Scope for --detect-extend: 'objects' extends only "
                             "ray-object hit detection, 'phenomena' extends only "
                             "phenomena tracking (mutual gaze, social ref), 'both' "
                             "extends both (default: objects).")
    parser.add_argument("--gaze-cone", type=float, default=0.0, metavar="DEGREES",
                        help="Replaces standard gaze vectors with vision cones of a specified "
                             "angle in degrees (disabled by default).")
    parser.add_argument("--gaze-lock", action="store_true", default=False,
                        help="Enable fixation lock-on (default: off).")
    parser.add_argument("--dwell-frames", type=int, default=15)
    parser.add_argument("--lock-dist", type=int, default=100)
    parser.add_argument("--gaze-debug", action="store_true")
    parser.add_argument("--snap-switch-frames", type=int, default=8, metavar="N",
                        help="Frames before adaptive-snap hysteresis switches target (default 8).")
    parser.add_argument("--reid-grace-seconds", type=float, default=1.0, metavar="S",
                        help="Seconds a lost face track stays in the re-ID buffer (default 1.0).")
    parser.add_argument("--forward-gaze-threshold", type=float, default=5.0, metavar="DEG",
                        help="Pitch/yaw angles below this (degrees) are treated as looking "
                             "forward at the camera. Set to 0 to disable (default 5.0).")
