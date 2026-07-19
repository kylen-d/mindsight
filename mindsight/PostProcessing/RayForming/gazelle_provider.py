"""
RayForming/gazelle_provider.py -- Gaze-LLE model loading + fixation-aware
inference scheduling.

Owns the Gaze-LLE model instance, the HeatmapCache, and the
InferenceScheduler.  Each frame:
  1. The pipeline calls ``observe_face(...)`` per face with (track_id,
     py_dir, py_conf) -- this happens inside run_ray_forming, AFTER
     step() has run for the frame.  The scheduler's fire decision
     therefore uses the PREVIOUS frame's fixation likelihoods -- a
     deliberate one-frame lag that is harmless because fixations span
     many frames, and which keeps the provider's step() call site
     unchanged from the legacy pipeline ordering.
  2. ``step()`` asks the scheduler whether to fire; if yes, runs
     Gaze-LLE over the batch of active faces and stores heatmaps in the
     cache with per-track ``wanted`` flags set by the scheduler.

See docs/superpowers/specs/2026-05-15-gazelle-blend-redesign-design.md.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mindsight.PostProcessing.RayForming.heatmap_cache import HeatmapCache
from mindsight.PostProcessing.RayForming.inference_scheduler import InferenceScheduler
from mindsight.PostProcessing.RayForming.ray_config import resolve_min_call_gap


class GazelleReuseGate:
    """Perceptual refire suppression for Gaze-LLE (v1.1 W3X; off unless
    ``--rf-reuse-eps`` > 0).

    At every REAL forward pass the gate stores a grayscale thumbnail of the
    frame plus each track's bbox.  When the scheduler next asks to fire, the
    call is skipped iff the frame thumbnail is visually unchanged (mean
    absolute difference <= eps vs the LAST REAL call -- drift accumulates
    into the diff, so skipping self-limits) AND every wanting track still
    has a cached heatmap and a stable bbox (IoU >= 0.5 vs fire time).  The
    cached heatmaps are then re-anchored (age reset, wanted re-flagged), so
    to the blender a reused accept is indistinguishable from a fresh fire --
    truthful for the same reason as the W2.2 MGazeReuseCache: on an
    unchanged input the model output genuinely would not move.
    """

    _THUMB_SIZE = (64, 64)
    _MIN_IOU = 0.5

    def __init__(self, eps: float):
        self.eps = float(eps)
        self.hits = 0
        self.misses = 0
        self._thumb: np.ndarray | None = None
        self._bboxes: dict[int, tuple] = {}

    @staticmethod
    def _make_thumb(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, GazelleReuseGate._THUMB_SIZE,
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

    def try_reuse(self, frame: np.ndarray, wanting_tids: set[int],
                  bbox_by_tid: dict[int, tuple],
                  cache: HeatmapCache) -> bool:
        """Re-anchor cached heatmaps instead of firing, when safe.

        Returns True (and refreshes the cache entries for *wanting_tids*)
        only when the scene and every wanting face are visually unchanged
        since the last real call; the caller must still record_accepted().
        """
        if self._thumb is None:
            self.misses += 1
            return False
        thumb = self._make_thumb(frame)
        if float(np.mean(np.abs(thumb - self._thumb))) > self.eps:
            self.misses += 1
            return False
        for tid in wanting_tids:
            now = bbox_by_tid.get(tid)
            then = self._bboxes.get(tid)
            if now is None or then is None or self._iou(now, then) < self._MIN_IOU:
                self.misses += 1
                return False
            if cache.get(tid)[0] is None:
                self.misses += 1
                return False
        for tid in wanting_tids:
            hm, _age, inout, _wanted = cache.get(tid)
            cache.update(tid, hm, inout_score=inout, wanted=True)
        self.hits += 1
        return True

    def record_fire(self, frame: np.ndarray,
                    bbox_by_tid: dict[int, tuple]) -> None:
        """Snapshot the scene state at a real forward pass."""
        self._thumb = self._make_thumb(frame)
        self._bboxes = dict(bbox_by_tid)

    def prune(self, active_tids: set[int]) -> None:
        """Drop stored bboxes for tracks that no longer exist."""
        for tid in list(self._bboxes):
            if tid not in active_tids:
                self._bboxes.pop(tid, None)


def _resolve_gazelle_name(ns, ckpt_path: Path) -> str:
    """Resolve the Gaze-LLE architecture variant to construct (v1.1 W3.1).

    When in/out gating is requested (``rf_inout_gate > 0``), the user left
    ``--rf-gazelle-name`` untyped, and the checkpoint actually carries the
    in/out head parameters, upgrade to the ``_inout`` architecture -- same
    weights, numerically identical heatmaps, plus the in/out output the
    gate needs.  (The shipped default checkpoint IS upstream's ``_inout``
    file; without the upgrade its head loads and is discarded.)  An
    explicitly typed name always wins, and with the gate at 0 (default)
    the 1.0.0 construction is reproduced exactly.
    """
    gz_name = getattr(ns, 'rf_gazelle_name', 'gazelle_dinov2_vitb14')
    gate = getattr(ns, 'rf_inout_gate', 0.0) or 0.0
    explicit = 'rf_gazelle_name' in getattr(ns, '_explicit_cli', frozenset())
    if gate <= 0 or explicit or gz_name.endswith('_inout'):
        return gz_name
    try:
        import torch
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        has_head = any(k.startswith('inout_head.') for k in sd.keys())
    except Exception as exc:  # unreadable checkpoint: let the engine report it
        print(f"Note: could not inspect checkpoint for in/out head ({exc})")
        return gz_name
    if has_head:
        upgraded = gz_name + '_inout'
        print(f"Gaze-LLE checkpoint carries an in/out head: constructing "
              f"{upgraded} (in/out gate {gate:.2f})")
        return upgraded
    print("Note: --rf-inout-gate set but the checkpoint has no in/out head; "
          "gating is inert (in/out score stays 1.0)")
    return gz_name


class GazelleProvider:
    """Manages a Gaze-LLE model + a fixation-aware inference scheduler."""

    def __init__(self, gazelle_engine, *,
                 v_threshold: float, d_threshold: float, min_call_gap: int,
                 onset_samples: int = 0, onset_gap: int = 0,
                 reuse_eps: float = 0.0,
                 length_engine=None, length_refresh_gap: int = 0):
        self._engine = gazelle_engine
        # v1.1 W3Y cheap length channel (inert while gap is 0): a second,
        # normally half-precision engine whose heatmaps refresh ray LENGTH
        # only.  May be the main engine itself (shared) when a separate
        # fp16 copy buys nothing (CPU device, or main engine already fp16).
        self._length_engine = length_engine if length_refresh_gap > 0 else None
        self._length_refresh_gap = int(length_refresh_gap)
        # tid -> (heatmap, inout_score) from the latest length-only pass;
        # consumed (popped) by ray_pipeline on the following blend update.
        self._length_refresh: dict[int, tuple[np.ndarray, float]] = {}
        # Retained so reset() can rebuild the scheduler/cache without the ns.
        self._v_threshold = float(v_threshold)
        self._d_threshold = float(d_threshold)
        self._min_call_gap = int(min_call_gap)
        self._onset_samples = int(onset_samples)
        self._onset_gap = int(onset_gap)
        self._reuse_eps = float(reuse_eps)
        self.heatmap_cache = HeatmapCache()
        self._scheduler = InferenceScheduler(
            v_threshold=v_threshold,
            d_threshold=d_threshold,
            min_call_gap=min_call_gap,
            onset_samples=onset_samples,
            onset_gap=onset_gap,
            length_refresh_gap=length_refresh_gap,
        )
        self.reuse_gate = (GazelleReuseGate(reuse_eps)
                           if reuse_eps > 0 else None)

    def reset(self) -> None:
        """Drop all per-run scheduling + heatmap-cache state, keeping the
        loaded Gaze-LLE model.

        Called between videos in a project batch (SP3.1 Q4/D9) so a video's
        inference schedule, per-track fixation history, and belief anchors
        never depend on the previous video's end state -- notably the global
        call-gap counter and the per-track buffers that video b would
        otherwise inherit (track-IDs restart at 0 each video).  The multi-GB
        model weights are NOT reloaded; only the cheap scheduler + cache are
        rebuilt from the thresholds captured at construction.
        """
        self.heatmap_cache = HeatmapCache()
        self._length_refresh = {}
        self._scheduler = InferenceScheduler(
            v_threshold=self._v_threshold,
            d_threshold=self._d_threshold,
            min_call_gap=self._min_call_gap,
            onset_samples=self._onset_samples,
            onset_gap=self._onset_gap,
            length_refresh_gap=self._length_refresh_gap,
        )
        self.reuse_gate = (GazelleReuseGate(self._reuse_eps)
                           if self._reuse_eps > 0 else None)

    @classmethod
    def from_namespace(cls, ns, device: str = "auto") -> "GazelleProvider | None":
        gazelle_ckpt = getattr(ns, 'rf_gazelle_model', None)
        if not gazelle_ckpt:
            return None

        from mindsight.weights import resolve_weight

        ckpt_path = Path(resolve_weight("Gazelle", str(gazelle_ckpt)))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Gaze-LLE checkpoint not found: {ckpt_path}")

        # v1.1 W3Z: an .onnx checkpoint selects the onnxruntime engine
        # (PINTO0309 gazelle-dinov3 exports -- DINOv3/distilled backbones;
        # atto measured ~8x faster per call than the torch vitb14 engine).
        # Same raw_heatmaps/_last_inout contract, so everything downstream
        # (scheduler, blender, length channel) is untouched.
        if ckpt_path.suffix.lower() == ".onnx":
            from Plugins.GazeTracking.Gazelle.gazelle_onnx_engine import (
                GazelleOnnxEngine,
            )
            gz_name = ckpt_path.name
            # The global --device picks the execution provider (cuda -> CUDA
            # EP, mps -> CoreML EP for the ViT static exports, cpu -> CPU);
            # unavailable/failing providers fall back to CPU with a note.
            engine = GazelleOnnxEngine(ckpt_path, device=device)
        else:
            from Plugins.GazeTracking.Gazelle.gazelle_backend import (
                GazeEstimationGazelle,
            )
            gz_name = _resolve_gazelle_name(ns, ckpt_path)
            engine = GazeEstimationGazelle(
                gz_name, ckpt_path,
                inout_threshold=0.5,   # engine-internal; scheduler gates separately
                skip_frames=0,
                # v1.1 W2.3: plumbed through (previously hardcoded False, so the
                # --gazelle-* flags only reached the standalone backend). Both
                # default off -- fp16 is never byte-identical to fp32.
                use_fp16=getattr(ns, 'rf_gazelle_fp16', False),
                use_compile=getattr(ns, 'rf_gazelle_compile', False),
                device=device,
            )

        v_thresh = getattr(ns, 'fixation_v_threshold', 0.04)
        d_thresh = getattr(ns, 'fixation_d_threshold', 0.15)
        gap = resolve_min_call_gap(ns)

        # v1.1 W3Y cheap length channel.  A separate persistent fp16 sibling
        # engine (same checkpoint; casting per call would cost more than it
        # saves) is built ONLY on CUDA, where half precision genuinely runs
        # faster (tensor cores) for its ~179MB (vitb14).  Measured on MPS,
        # fp16 is NOT faster per call (87.1 vs 87.7 ms solo, 2026-07-17), so
        # on MPS/CPU -- or when the main engine is already fp16 -- the main
        # engine is shared: same per-call cost, zero extra memory, exact
        # output.  The channel's value there is pure scheduling (periodic
        # length refreshes between fixation-gated corrections).
        len_gap = getattr(ns, 'rf_len_refresh_gap', 10) or 0
        length_engine = None
        if len_gap > 0:
            main_is_fp16 = getattr(engine, '_use_fp16', False)
            is_cuda = getattr(engine, 'device', None) is not None and \
                engine.device.type == "cuda"
            if main_is_fp16 or not is_cuda:
                length_engine = engine
                print(f"Gaze-LLE length-refresh channel: every {len_gap} "
                      f"frames, length-only (sharing the main engine -- no "
                      f"separate fp16 copy pays off on this device)")
            else:
                length_engine = GazeEstimationGazelle(
                    gz_name, ckpt_path,
                    inout_threshold=0.5,
                    skip_frames=0,
                    use_fp16=True,
                    use_compile=False,
                    device=device,
                )
                print(f"Gaze-LLE length-refresh channel: fp16 engine loaded "
                      f"(every {len_gap} frames, length-only)")

        print(f"Gaze-LLE model loaded: {gz_name}")
        return cls(engine,
                   v_threshold=v_thresh,
                   d_threshold=d_thresh,
                   min_call_gap=gap,
                   # v1.1 W3X fire-decision knobs; all inert at defaults.
                   onset_samples=getattr(ns, 'rf_onset_samples', 3) or 0,
                   onset_gap=getattr(ns, 'rf_onset_gap', 5) or 0,
                   reuse_eps=getattr(ns, 'rf_reuse_eps', 0.0) or 0.0,
                   length_engine=length_engine,
                   length_refresh_gap=len_gap)

    def observe_face(self, *, track_id: int, py_dir: np.ndarray,
                     py_conf: float) -> None:
        """Feed one face's PY observation into the scheduler for this frame."""
        self._scheduler.observe(track_id=track_id, py_dir=py_dir, py_conf=py_conf)

    def step(self, frame: np.ndarray, face_bboxes: list[tuple],
             face_track_ids: list[int]) -> None:
        """Per-frame: consult scheduler, fire if warranted, cache heatmaps."""
        active_tids = set(face_track_ids)
        self.heatmap_cache.age_all(active_tids)

        should_fire, wanting_tids = self._scheduler.tick()
        fired = False
        if should_fire and face_bboxes:
            bbox_by_tid = dict(zip(face_track_ids, face_bboxes))
            reused = (self.reuse_gate is not None
                      and self.reuse_gate.try_reuse(
                          frame, wanting_tids, bbox_by_tid,
                          self.heatmap_cache))
            if not reused:
                heatmaps = self._engine.raw_heatmaps(frame, face_bboxes)
                inout_arr = getattr(self._engine, '_last_inout', None)
                for fi, tid in enumerate(face_track_ids):
                    if fi < heatmaps.shape[0]:
                        inout = float(inout_arr[fi]) if (
                            inout_arr is not None and fi < len(inout_arr)) else 1.0
                        self.heatmap_cache.update(
                            tid, heatmaps[fi], inout_score=inout,
                            wanted=(tid in wanting_tids))
                if self.reuse_gate is not None:
                    self.reuse_gate.record_fire(frame, bbox_by_tid)
            self._scheduler.record_accepted(wanting_tids)
            fired = True

        # Cheap length-only channel (v1.1 W3Y): when the full-precision
        # channel stayed quiet this frame, a counter-gated fp16 pass may
        # refresh ray length for every current face.  Results go to the
        # side dict, NOT the heatmap cache -- they must never become
        # belief-map accepts.
        if (not fired and self._length_engine is not None and face_bboxes
                and self._scheduler.tick_length_refresh()):
            heatmaps = self._length_engine.raw_heatmaps(frame, face_bboxes)
            inout_arr = getattr(self._length_engine, '_last_inout', None)
            for fi, tid in enumerate(face_track_ids):
                if fi < heatmaps.shape[0]:
                    inout = float(inout_arr[fi]) if (
                        inout_arr is not None and fi < len(inout_arr)) else 1.0
                    self._length_refresh[tid] = (heatmaps[fi], inout)
            self._scheduler.record_length_refresh()

        # Prune scheduler state for disappeared tracks, then advance the
        # frame counters (also clears the observed-this-frame set).
        stale = self._scheduler.tracked_tids - active_tids
        if stale:
            self._scheduler.forget(stale)
        for tid in list(self._length_refresh):
            if tid not in active_tids:
                self._length_refresh.pop(tid, None)
        if self.reuse_gate is not None:
            self.reuse_gate.prune(active_tids)
        self._scheduler.advance_frame()

    def likelihood(self, track_id: int) -> float:
        """Fixation_likelihood for a track, for the blender's trust input."""
        return self._scheduler.likelihood(track_id)

    def pop_length_refresh(self, track_id: int):
        """Consume a pending length-only heatmap for a track.

        Returns ``(heatmap, inout_score)`` or ``None``.  Popped exactly
        once -- ray_pipeline calls this every blend update, so a result is
        applied on the first frame the track is seen after the pass.
        """
        return self._length_refresh.pop(track_id, None)

    @property
    def engine(self):
        return self._engine
