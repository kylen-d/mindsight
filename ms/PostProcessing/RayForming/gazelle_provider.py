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

import numpy as np

from ms.PostProcessing.RayForming.heatmap_cache import HeatmapCache
from ms.PostProcessing.RayForming.inference_scheduler import InferenceScheduler
from ms.PostProcessing.RayForming.ray_config import resolve_min_call_gap


class GazelleProvider:
    """Manages a Gaze-LLE model + a fixation-aware inference scheduler."""

    def __init__(self, gazelle_engine, *,
                 v_threshold: float, d_threshold: float, min_call_gap: int):
        self._engine = gazelle_engine
        self.heatmap_cache = HeatmapCache()
        self._scheduler = InferenceScheduler(
            v_threshold=v_threshold,
            d_threshold=d_threshold,
            min_call_gap=min_call_gap,
        )

    @classmethod
    def from_namespace(cls, ns, device: str = "auto") -> "GazelleProvider | None":
        gazelle_ckpt = getattr(ns, 'rf_gazelle_model', None)
        if not gazelle_ckpt:
            return None

        from Plugins.GazeTracking.Gazelle.gazelle_backend import (
            GazeEstimationGazelle,
        )
        from ms.weights import resolve_weight

        ckpt_path = Path(resolve_weight("Gazelle", str(gazelle_ckpt)))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Gaze-LLE checkpoint not found: {ckpt_path}")

        gz_name = getattr(ns, 'rf_gazelle_name', 'gazelle_dinov2_vitb14')
        engine = GazeEstimationGazelle(
            gz_name, ckpt_path,
            inout_threshold=0.5,       # engine-internal; scheduler gates separately
            skip_frames=0,
            use_fp16=False,
            use_compile=False,
            device=device,
        )

        v_thresh = getattr(ns, 'fixation_v_threshold', 0.02)
        d_thresh = getattr(ns, 'fixation_d_threshold', 0.10)
        gap = resolve_min_call_gap(ns)

        print(f"Gaze-LLE model loaded: {gz_name}")
        return cls(engine,
                   v_threshold=v_thresh,
                   d_threshold=d_thresh,
                   min_call_gap=gap)

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
        if should_fire and face_bboxes:
            heatmaps = self._engine.raw_heatmaps(frame, face_bboxes)
            inout_arr = getattr(self._engine, '_last_inout', None)
            for fi, tid in enumerate(face_track_ids):
                if fi < heatmaps.shape[0]:
                    inout = float(inout_arr[fi]) if (
                        inout_arr is not None and fi < len(inout_arr)) else 1.0
                    self.heatmap_cache.update(
                        tid, heatmaps[fi], inout_score=inout,
                        wanted=(tid in wanting_tids))
            self._scheduler.record_accepted(wanting_tids)

        # Prune scheduler state for disappeared tracks, then advance the
        # frame counters (also clears the observed-this-frame set).
        stale = self._scheduler.tracked_tids - active_tids
        if stale:
            self._scheduler.forget(stale)
        self._scheduler.advance_frame()

    def likelihood(self, track_id: int) -> float:
        """Fixation_likelihood for a track, for the blender's trust input."""
        return self._scheduler.likelihood(track_id)

    @property
    def engine(self):
        return self._engine
