"""
GazeTracking/gaze_processing.py — Generic gaze utilities shared across all backends.

Contains backend-agnostic tools for temporal smoothing, track re-identification,
and eye-landmark extraction.  The post-processing tools (snap scoring, fixation
lock-on, tip-snapping, ray-bbox intersection) live in
``mindsight.PostProcessing.RayForming``.

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

from mindsight.constants import SMOOTH_ALPHA
from mindsight.PostProcessing.RayForming.fixation import GazeLockTracker
from mindsight.PostProcessing.RayForming.object_snap import SnapTemporalState, snap_score

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


def normalize_face_dicts(raw, inv_scale=1.0):
    """Map RetinaFace detection dicts onto the internal face shape (W3X).

    uniface 1.1.0 emits ``{'bbox': [x1,y1,x2,y2], 'confidence': float,
    'landmarks': [[x,y]*5]}`` -- but the pipeline convention (and every
    downstream reader) expects the detection score appended as a 5th bbox
    element and the keypoints under ``'kps'``.  Before this adapter the
    mismatch silently disabled the eye-centre origin and the
    EYE_CONF_THRESH gate (kps was always None, score always 1.0).

    Dicts already carrying ``'kps'`` or a scored bbox (tests, plugins,
    non-uniface detectors) pass through with scaling only.
    """
    out = []
    for f in raw:
        bbox = [float(c) * inv_scale for c in f["bbox"][:4]]
        score = f["bbox"][4] if len(f["bbox"]) > 4 else f.get("confidence")
        kps = f.get("kps")
        if kps is None:
            kps = f.get("landmarks")
        nf = dict(f)
        nf["bbox"] = bbox + ([float(score)] if score is not None else [])
        nf["kps"] = ([[float(k[0]) * inv_scale, float(k[1]) * inv_scale]
                      for k in kps] if kps is not None else None)
        out.append(nf)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Per-face estimate reuse (perceptual no-change gate)
# ══════════════════════════════════════════════════════════════════════════════

class MGazeReuseCache:
    """Skip per-face gaze estimation on visually-unchanged crops (v1.1 W2.2).

    The per-face model runs every frame for every face regardless of
    ``skip_frames`` -- the dominant pipeline cost on multi-face footage.  A
    naive every-Nth skip would poison the blend scheduler's fixation history
    with artificial zero-velocity samples, so this cache only reuses when the
    face crop is *visually unchanged* (mean absolute difference of 32x32
    grayscale thumbnails at or below ``eps``): in that regime the model output
    genuinely would not move, and a static crop reading as a fixation is
    truthful.  Entries are matched frame-to-frame by bbox IoU, so face-order
    churn simply misses the cache rather than mixing faces up.

    ``eps <= 0`` disables reuse entirely (the default).
    """

    _THUMB_SIZE = (32, 32)
    _MIN_IOU = 0.5

    def __init__(self, eps: float):
        self.eps = float(eps)
        self.hits = 0
        self.misses = 0
        self._prev: list[dict] = []
        self._next: list[dict] = []

    @staticmethod
    def _thumb(crop) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, MGazeReuseCache._THUMB_SIZE,
                          interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def _iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(area_a + area_b - inter)

    def estimate(self, bbox, crop, estimator):
        """Return ``estimator(crop)``, reusing the previous frame's result
        when the same face region is visually unchanged."""
        if self.eps <= 0:
            return estimator(crop)
        thumb = self._thumb(crop)
        result = None
        for entry in self._prev:
            if self._iou(bbox, entry['bbox']) < self._MIN_IOU:
                continue
            if float(np.mean(np.abs(thumb - entry['thumb']))) <= self.eps:
                result = entry['result']
                self.hits += 1
                break
        if result is None:
            result = estimator(crop)
            self.misses += 1
        self._next.append({'bbox': bbox, 'thumb': thumb, 'result': result})
        return result

    def end_frame(self) -> None:
        """Roll this frame's entries into the match set for the next frame."""
        self._prev = self._next
        self._next = []


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

    ``embed_fn`` + ``embed_sim > 0`` (v1.1 W3X, ``--face-reid-sim``) add
    embedding-verified revival: dead tracks carry a face embedding
    (computed at creation, EMA-refreshed periodically) and a new detection
    is matched against them by cosine similarity FIRST -- anywhere in the
    frame, not just within max_dist -- reviving only above ``embed_sim``.
    When no dead track passes, the positional revival above still applies,
    so the feature is strictly additive.  ``embed_fn(crop, kps_local)``
    must return a unit-norm 1-D vector or None (kps are crop-relative
    5-point landmarks; entries without landmarks skip embedding work).
    """

    _EMBED_REFRESH = 30   # frames between per-track embedding refreshes

    def __init__(self, alpha=SMOOTH_ALPHA, max_dist=200, hist_weight=0.35,
                 hist_bins=16, grace_frames=0, embed_fn=None, embed_sim=0.0):
        self.alpha, self.max_dist = alpha, max_dist
        self.hist_weight          = hist_weight
        self.hist_bins            = hist_bins
        self.grace_frames         = grace_frames
        self.embed_fn             = embed_fn
        self.embed_sim            = float(embed_sim)
        self._tracks, self._nid   = {}, 0
        self._dead: dict          = {}   # tid -> {c, p, y, h, _dropped}
        self._frame: int          = 0

    @property
    def _embed_active(self) -> bool:
        return self.embed_fn is not None and self.embed_sim > 0.0

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

    def _new_track(self, center, pitch, yaw, crop, embed=None) -> dict:
        """Create the state dict for a brand-new track."""
        return dict(c=center, p=pitch, y=yaw,
                    h=self._histogram(crop, self.hist_bins),
                    e=embed, ef=self._frame)

    # -- Embedding helpers (W3X; inert unless embed_fn + embed_sim set) -------

    def _embed(self, crop, kps):
        """Compute a unit embedding for a detection, or None."""
        if not self._embed_active or crop is None or kps is None:
            return None
        try:
            e = self.embed_fn(crop, kps)
        except Exception:
            return None
        if e is None:
            return None
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(e))
        return e / n if n > 0 else None

    def _refresh_embedding(self, track, crop, kps):
        """EMA-refresh a live track's embedding every _EMBED_REFRESH frames."""
        if not self._embed_active:
            return
        if (track.get('e') is not None
                and self._frame - track.get('ef', 0) < self._EMBED_REFRESH):
            return
        e_new = self._embed(crop, kps)
        if e_new is None:
            return
        e_old = track.get('e')
        if e_old is not None:
            mixed = 0.7 * e_old + 0.3 * e_new
            n = float(np.linalg.norm(mixed))
            e_new = mixed / n if n > 0 else e_new
        track['e'] = e_new
        track['ef'] = self._frame

    def _revive_by_embedding(self, det_embed):
        """Best dead-track ID by cosine similarity >= embed_sim, else None."""
        if det_embed is None:
            return None
        best_tid, best_sim = None, self.embed_sim
        for tid, t in self._dead.items():
            e = t.get('e')
            if e is None:
                continue
            sim = float(np.dot(det_embed, e))
            if sim >= best_sim:
                best_sim, best_tid = sim, tid
        return best_tid

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
        faces: [(center, pitch, yaw, crop)] -- an optional 5th element
        carries crop-relative 5-point landmarks for embedding ReID (W3X).
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
            kps = entry[4] if len(entry) > 4 else None
            c_raw = np.asarray(center, float)
            c_match = c_raw - shift  # shift-compensated for matching only

            # 1. Try to match a live track
            bid = self._best_match(c_match, crop, self._tracks)
            if bid is not None:
                sp, sy = self._update_track(self._tracks[bid], c_raw, pitch, yaw, crop)
                self._refresh_embedding(self._tracks[bid], crop, kps)
                unmatched.discard(bid)
                result.append((sp, sy, bid))
                continue

            # 2. No live match — try to revive a dead track (grace period).
            # Embedding identity wins over position when enabled (W3X):
            # a person may re-enter anywhere in the frame.
            if self.grace_frames > 0 and self._dead:
                det_embed = (self._embed(crop, kps)
                             if self._embed_active else None)
                did = self._revive_by_embedding(det_embed)
                if did is None:
                    did = self._best_match(c_match, crop, self._dead)
                if did is not None:
                    revived = self._dead.pop(did)
                    sp, sy = self._update_track(revived, c_raw, pitch, yaw, crop)
                    if det_embed is not None:
                        revived['e'] = det_embed
                        revived['ef'] = self._frame
                    self._tracks[did] = revived
                    result.append((sp, sy, did))
                    continue

            # 3. Genuinely new face — allocate a fresh ID
            nid = self._nid; self._nid += 1
            self._tracks[nid] = self._new_track(
                c_raw, pitch, yaw, crop, embed=self._embed(crop, kps))
            result.append((pitch, yaw, nid))

        # Move unmatched live tracks to the dead-track buffer (or drop immediately)
        for tid in unmatched:
            if self.grace_frames > 0:
                self._dead[tid] = {**self._tracks[tid], '_dropped': self._frame}
            del self._tracks[tid]
        return result


def create_face_embedder():
    """Build the ``embed_fn`` used by embedding-verified ReID (W3X).

    Wraps uniface's ArcFace recognizer: the crop is aligned from its
    crop-relative 5-point landmarks and embedded to a unit vector.  The
    ONNX weights auto-download to ~/.uniface on first use (same mechanism
    as the RetinaFace detector weights).  Note the upstream provenance:
    the w600k ArcFace models come from the InsightFace model zoo, which
    marks them for non-commercial research use -- see THIRD_PARTY_LICENSES.
    """
    from uniface import ArcFace
    model = ArcFace()

    def embed(crop, kps):
        if crop is None or getattr(crop, 'size', 0) == 0 or kps is None:
            return None
        if len(kps) < 5:
            return None
        emb = model.get_normalized_embedding(
            crop, np.asarray(kps, dtype=np.float32))
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    return embed


# ══════════════════════════════════════════════════════════════════════════════
# Face-as-object helper
# ══════════════════════════════════════════════════════════════════════════════

def _faces_as_objects(face_bboxes_list):
    """Convert a list of (x1,y1,x2,y2) face bboxes to Detection objects so they
    can be treated as gaze targets by adaptive_snap and the hit-detection loop."""
    from mindsight.ObjectDetection.detection import Detection
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
