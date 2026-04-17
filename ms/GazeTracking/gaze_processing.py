"""
GazeTracking/gaze_processing.py — Generic gaze utilities shared across all backends.

Contains backend-agnostic tools for temporal smoothing, track re-identification,
and eye-landmark extraction.  Also re-exports the post-processing tools
(snap scoring, fixation lock-on, tip-snapping, ray-bbox intersection) that
now live in ``ms.PostProcessing.RayForming`` for backward compatibility.

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

# ══════════════════════════════════════════════════════════════════════════════
# Backward-compat re-exports from ms.PostProcessing.RayForming
# ══════════════════════════════════════════════════════════════════════════════
# These were originally defined in this file and have been moved to the
# unified RayForming module.  Re-exported here so existing imports continue
# to work.
from ms.PostProcessing.RayForming.object_snap import (  # noqa: F401, E402
    SmoothSnapTracker,
    SnapTemporalState,
    snap_score,
    apply_tip_snapping,
)
from ms.PostProcessing.RayForming.fixation import (     # noqa: F401, E402
    GazeLockTracker,
    apply_lock_on,
)
from ms.PostProcessing.RayForming.hit_detection import ( # noqa: F401, E402
    compute_ray_intersections,
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
# Face-as-object helper
# ══════════════════════════════════════════════════════════════════════════════

def _faces_as_objects(face_bboxes_list):
    """Convert a list of (x1,y1,x2,y2) face bboxes to Detection objects so they
    can be treated as gaze targets by adaptive_snap and the hit-detection loop."""
    from ms.ObjectDetection.detection import Detection
    return [Detection(class_name='face', cls_id=-1, conf=1.0,
                      x1=x1, y1=y1, x2=x2, y2=y2, _face_idx=fi)
            for fi, (x1, y1, x2, y2) in enumerate(face_bboxes_list)]


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

    def create_snap_temporal(self, release_frames: int = 5,
                             engage_frames: int = 0) -> SnapTemporalState:
        """Create a snap temporal state tracker."""
        return SnapTemporalState(release_frames=release_frames,
                                 engage_frames=engage_frames)

    def get_eye_center(self, face_dict, inv_scale=1.0):
        """Extract eye centre from face keypoints (overridable)."""
        return _get_eye_center(face_dict, inv_scale)

    def compute_snap_score(self, origin, direction, objects, fallback,
                           **kwargs):
        """Compute snap score (overridable)."""
        return snap_score(origin, direction, objects, fallback, **kwargs)

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
                        help="Snap scoring weight for normalized distance penalty (default 1.0)")
    parser.add_argument("--snap-w-angle", type=float, default=0.8,
                        help="Snap scoring weight for angular deviation penalty (default 0.8)")
    parser.add_argument("--snap-w-size", type=float, default=0.0,
                        help="Snap scoring weight for object size reward (default 0.0)")
    parser.add_argument("--snap-w-intersect", type=float, default=0.5,
                        help="Snap scoring bonus for ray-bbox intersection (default 0.5)")
    parser.add_argument("--snap-w-temporal", type=float, default=0.3,
                        help="Snap scoring bonus for previous-frame target stickiness (default 0.3)")
    parser.add_argument("--snap-gate-angle", type=float, default=60.0, metavar="DEG",
                        help="Hard angular cutoff in degrees: objects beyond this angle "
                             "from the blended gaze+head direction are never snap candidates "
                             "(default 60.0).")
    parser.add_argument("--snap-head-blend", type=float, default=0.3, metavar="F",
                        help="Blend factor for angular scoring: 0=pure gaze direction, "
                             "1=pure head orientation (default 0.3).")
    parser.add_argument("--snap-quality-thresh", type=float, default=0.8, metavar="F",
                        help="Maximum score to accept a snap match. Higher values are more "
                             "permissive. Set lower to reject poor matches (default 0.8).")
    parser.add_argument("--snap-tip-dist", type=float, default=-1.0, metavar="PX",
                        help="Tip-snap distance threshold. -1 = use --snap-dist (default -1).")
    parser.add_argument("--snap-tip-quality", type=float, default=-1.0, metavar="F",
                        help="Tip-snap quality threshold. -1 = use --snap-quality-thresh "
                             "(default -1).")
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
    parser.add_argument("--snap-release-frames", type=int, default=5, metavar="N",
                        help="Frames of no-match before releasing the held snap target (default 5).")
    parser.add_argument("--snap-engage-frames", type=int, default=0, metavar="N",
                        help="Frames of consistent match required before engaging snap "
                             "for the first time. 0 = instant engage (default 0).")
    parser.add_argument("--reid-grace-seconds", type=float, default=1.0, metavar="S",
                        help="Seconds a lost face track stays in the re-ID buffer (default 1.0).")
    parser.add_argument("--forward-gaze-threshold", type=float, default=5.0, metavar="DEG",
                        help="Pitch/yaw angles below this (degrees) are treated as looking "
                             "forward at the camera. Set to 0 to disable (default 5.0).")
    parser.add_argument("--smooth-snap", type=str, default="off",
                        choices=["off", "objects", "gaze_tips", "all"],
                        help="Smooth snap mode: smoothly interpolate the ray toward "
                             "snap targets instead of jumping instantly. 'objects' = "
                             "smooth object snaps only, 'gaze_tips' = smooth gaze-tip "
                             "snaps only, 'all' = both (default: off).")
    parser.add_argument("--smooth-snap-alpha", type=float, default=0.20, metavar="F",
                        help="EMA rate for smooth snap: lower = smoother/slower, "
                             "higher = faster/more responsive (default: 0.20).")

    # ── Ray Forming (Gazelle blend) flags ──────────────────────────────────
    rf = parser.add_argument_group("Ray Forming (Gazelle blend)")
    rf.add_argument("--rf-gazelle-model", default=None, metavar="PATH",
                    help="Path to a Gaze-LLE checkpoint (.pt) for Gazelle "
                         "blend ray forming. Used alongside a pitch/yaw "
                         "backend (L2CS, MGaze, etc.) to periodically correct "
                         "rays with Gaze-LLE heatmaps.")
    rf.add_argument("--rf-gazelle-name",
                    default="gazelle_dinov2_vitb14",
                    choices=sorted([
                        "gazelle_dinov2_vitb14", "gazelle_dinov2_vitl14",
                        "gazelle_dinov2_vitb14_inout",
                        "gazelle_dinov2_vitl14_inout",
                    ]),
                    metavar="NAME",
                    help="Gaze-LLE model variant for ray forming "
                         "(default: gazelle_dinov2_vitb14).")
    rf.add_argument("--rf-gazelle-interval", type=int, default=30, metavar="N",
                    help="Run Gaze-LLE heatmap inference every N frames "
                         "(default: 30).")
    rf.add_argument("--blend-strength", type=float, default=1.0, metavar="F",
                    help="Gazelle blend strength (sets both direction and "
                         "length if not individually specified). 0.0 = pure "
                         "pitch/yaw, 1.0 = full fusion (default: 1.0).")
    rf.add_argument("--direction-blend", type=float, default=None, metavar="F",
                    help="Direction blend strength. Overrides --blend-strength "
                         "for direction only (default: uses --blend-strength).")
    rf.add_argument("--length-blend", type=float, default=None, metavar="F",
                    help="Length/reach blend strength. Overrides "
                         "--blend-strength for length only "
                         "(default: uses --blend-strength).")
    rf.add_argument("--length-only", action="store_true", default=False,
                    help="Gazelle influences ray length only, not direction.")
    rf.add_argument("--direction-decay", type=float, default=0.30, metavar="F",
                    help="Ray direction EMA response rate. Higher = direction "
                         "follows belief centroid faster. Lower = smoother "
                         "direction transitions (default: 0.30).")
    rf.add_argument("--length-decay", type=float, default=0.15, metavar="F",
                    help="Ray length/reach EMA response rate. Lower = ray "
                         "reach persists longer between Gazelle updates. "
                         "Set lower than --direction-decay for stable reach "
                         "(default: 0.15).")
    rf.add_argument("--diffusion-sigma", type=float, default=0.40, metavar="F",
                    help="Per-frame Gaussian blur sigma on the belief map. "
                         "Controls how fast Gazelle correction confidence "
                         "decays between updates. 0 = no decay (default: 0.40).")
    rf.add_argument("--blend-conf-scale", type=float, default=0.70, metavar="F",
                    help="How much gaze confidence tightens the PY prior. "
                         "Higher = confident gaze estimates steer belief "
                         "more strongly (default: 0.70).")
    rf.add_argument("--belief-min-peak", type=float, default=0.05, metavar="F",
                    help="Minimum Gaze-LLE heatmap peak to accept "
                         "(default: 0.05).")
    rf.add_argument("--inout-threshold", type=float, default=0.5, metavar="F",
                    help="Suppress Gaze-LLE heatmap when in/out score is below "
                         "this threshold (default: 0.5).")
    rf.add_argument("--depth-ray-length", action="store_true", default=False,
                    help="Use depth map to scale ray length based on scene "
                         "geometry (default: off).")
    rf.add_argument("--depth-length-min", type=float, default=0.5, metavar="F",
                    help="Ray length multiplier at depth=0 (nearest) "
                         "(default: 0.5).")
    rf.add_argument("--depth-length-max", type=float, default=3.0, metavar="F",
                    help="Ray length multiplier at depth=1 (farthest) "
                         "(default: 3.0).")
    rf.add_argument("--depth-belief-boost", type=float, default=0.0, metavar="F",
                    help="How much depth agreement boosts Gaze-LLE heatmap "
                         "confidence in the belief update (default: 0.0).")
