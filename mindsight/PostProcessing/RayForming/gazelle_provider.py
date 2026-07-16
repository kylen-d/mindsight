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

from mindsight.PostProcessing.RayForming.heatmap_cache import HeatmapCache
from mindsight.PostProcessing.RayForming.inference_scheduler import InferenceScheduler
from mindsight.PostProcessing.RayForming.ray_config import resolve_min_call_gap


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
                 v_threshold: float, d_threshold: float, min_call_gap: int):
        self._engine = gazelle_engine
        # Retained so reset() can rebuild the scheduler/cache without the ns.
        self._v_threshold = float(v_threshold)
        self._d_threshold = float(d_threshold)
        self._min_call_gap = int(min_call_gap)
        self.heatmap_cache = HeatmapCache()
        self._scheduler = InferenceScheduler(
            v_threshold=v_threshold,
            d_threshold=d_threshold,
            min_call_gap=min_call_gap,
        )

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
        self._scheduler = InferenceScheduler(
            v_threshold=self._v_threshold,
            d_threshold=self._d_threshold,
            min_call_gap=self._min_call_gap,
        )

    @classmethod
    def from_namespace(cls, ns, device: str = "auto") -> "GazelleProvider | None":
        gazelle_ckpt = getattr(ns, 'rf_gazelle_model', None)
        if not gazelle_ckpt:
            return None

        from Plugins.GazeTracking.Gazelle.gazelle_backend import (
            GazeEstimationGazelle,
        )
        from mindsight.weights import resolve_weight

        ckpt_path = Path(resolve_weight("Gazelle", str(gazelle_ckpt)))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Gaze-LLE checkpoint not found: {ckpt_path}")

        gz_name = _resolve_gazelle_name(ns, ckpt_path)
        engine = GazeEstimationGazelle(
            gz_name, ckpt_path,
            inout_threshold=0.5,       # engine-internal; scheduler gates separately
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
