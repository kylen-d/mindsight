"""
RayForming/gazelle_provider.py — Gaze-LLE model loading and heatmap inference.

Handles Gazelle model instantiation, periodic heatmap inference scheduling,
and heatmap cache management as a core RayForming component.  Replaces the
model loading previously embedded in the GazelleSnap plugin.

Usage::

    provider = GazelleProvider.from_namespace(args, device="auto")
    if provider is not None:
        # Each frame: provider.step() runs Gazelle when scheduled
        provider.step(frame, face_bboxes, face_track_ids)
        # Heatmaps are stored in provider.heatmap_cache
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ms.PostProcessing.RayForming.heatmap_cache import HeatmapCache


class GazelleProvider:
    """Manages a Gaze-LLE model and periodic heatmap inference.

    Parameters
    ----------
    gazelle_engine : GazeEstimationGazelle instance with ``raw_heatmaps()`` method.
    interval       : Frames between Gaze-LLE inferences.
    """

    def __init__(self, gazelle_engine, *, interval: int = 30):
        self._engine = gazelle_engine
        self._interval = max(1, interval)
        self._frame_counter = 0
        self.heatmap_cache = HeatmapCache()

    @classmethod
    def from_namespace(cls, ns, device: str = "auto") -> GazelleProvider | None:
        """Create a GazelleProvider from CLI args, or return None if not configured.

        Looks for ``--gazelle-model`` (core flag) or ``--gs-gazelle-model``
        (legacy GazelleSnap flag) to determine whether to load a Gazelle model.
        """
        # Check core flag first, fall back to legacy GazelleSnap flag.
        # NOTE: uses rf_gazelle_model (not gazelle_model) to avoid
        # collision with the standalone Gazelle plugin's --gazelle-model.
        gazelle_ckpt = getattr(ns, 'rf_gazelle_model', None)
        if gazelle_ckpt is None:
            gazelle_ckpt = getattr(ns, 'gs_gazelle_model', None)
        if not gazelle_ckpt:
            return None

        from Plugins.GazeTracking.Gazelle.gazelle_backend import (
            GazeEstimationGazelle,
        )
        from ms.weights import resolve_weight

        ckpt_path = Path(resolve_weight("Gazelle", str(gazelle_ckpt)))
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Gaze-LLE checkpoint not found: {ckpt_path}"
            )

        # Model name: core flag or legacy flag
        gz_name = getattr(ns, 'rf_gazelle_name',
                   getattr(ns, 'gs_gazelle_name', 'gazelle_dinov2_vitb14'))
        inout_thresh = getattr(ns, 'inout_threshold', 0.5)

        engine = GazeEstimationGazelle(
            gz_name, ckpt_path,
            inout_threshold=inout_thresh,
            skip_frames=0,
            use_fp16=False,
            use_compile=False,
            device=device,
        )

        # Interval: core flag or legacy flag
        interval = getattr(ns, 'rf_gazelle_interval',
                    getattr(ns, 'gs_snap_interval', 30))

        print(f"Gaze-LLE model loaded: {gz_name}")
        return cls(engine, interval=interval)

    def step(self, frame: np.ndarray, face_bboxes: list[tuple],
             face_track_ids: list[int]) -> None:
        """Run one frame of the inference schedule.

        Calls Gaze-LLE every ``interval`` frames, caches the resulting
        heatmaps, and ages all entries.

        Parameters
        ----------
        frame          : BGR numpy array at display resolution.
        face_bboxes    : list of (x1, y1, x2, y2) in pixels.
        face_track_ids : stable track IDs matching the bboxes.
        """
        # Age all entries FIRST, then update with fresh heatmaps.
        # This ensures freshly updated entries have age=0 when the
        # ray forming pipeline reads them this frame.
        active_tids = set(face_track_ids)
        self.heatmap_cache.age_all(active_tids)

        run_now = (
            face_bboxes
            and self._frame_counter % self._interval == 0
        )

        if run_now:
            heatmaps = self._engine.raw_heatmaps(frame, face_bboxes)
            for fi, tid in enumerate(face_track_ids):
                if fi < heatmaps.shape[0]:
                    # Extract in/out score if the model supports it
                    inout = 1.0
                    inout_arr = getattr(self._engine, '_last_inout', None)
                    if inout_arr is not None and fi < len(inout_arr):
                        inout = float(inout_arr[fi])
                    self.heatmap_cache.update(tid, heatmaps[fi], inout)

        self._frame_counter += 1

    @property
    def interval(self) -> int:
        return self._interval

    @property
    def engine(self):
        """The underlying GazeEstimationGazelle instance."""
        return self._engine
