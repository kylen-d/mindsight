"""
RayForming/heatmap_cache.py -- Per-track Gaze-LLE heatmap storage and aging.

Manages cached 64x64 heatmaps from Gaze-LLE inference, tracks their age
(frames since last refresh), stores per-track in/out scores from
``_inout`` model variants, and carries a per-track ``wanted`` flag set by
the InferenceScheduler at fire time.

The ``wanted`` flag lets the GazeLLEBlender decide whether to APPLY a
fresh heatmap to its belief map -- only faces that were fixating when the
inference fired should accept the anchor; the others get the heatmap
cached (still available for debug overlays) but ignore it.
"""
from __future__ import annotations

import numpy as np


class HeatmapCache:
    """Per-track Gaze-LLE heatmap cache with age tracking.

    Parameters
    ----------
    max_age : int
        Heatmaps older than this are automatically pruned.  Set to 0 to
        disable automatic age-based pruning (only inactive-track pruning).
    """

    def __init__(self, max_age: int = 0):
        self._heatmaps: dict[int, np.ndarray] = {}
        self._ages: dict[int, int] = {}
        self._inout_scores: dict[int, float] = {}
        self._wanted: dict[int, bool] = {}
        self._max_age = max_age

    def update(self, track_id: int, heatmap: np.ndarray,
               inout_score: float = 1.0, wanted: bool = True) -> None:
        """Store a fresh heatmap for *track_id* and reset its age to 0.

        Parameters
        ----------
        wanted : bool
            True if the InferenceScheduler flagged this track as wanting
            the inference (blender should accept).  False if the track
            was included in the batch only because Gaze-LLE runs
            per-batch and this track happened to also be in-frame -- the
            heatmap is cached for debug overlays but the blender must
            ignore it.
        """
        self._heatmaps[track_id] = heatmap
        self._ages[track_id] = 0
        self._inout_scores[track_id] = inout_score
        self._wanted[track_id] = wanted

    def get(self, track_id: int) -> tuple[np.ndarray | None, int, float, bool]:
        """Return ``(heatmap, age, inout_score, wanted)`` or ``(None, -1, 0.0, False)``."""
        hm = self._heatmaps.get(track_id)
        if hm is None:
            return None, -1, 0.0, False
        return (hm,
                self._ages.get(track_id, 0),
                self._inout_scores.get(track_id, 1.0),
                self._wanted.get(track_id, False))

    def age_all(self, active_tids: set[int]) -> None:
        """Increment age for all entries and prune tracks no longer active."""
        for tid in list(self._ages):
            self._ages[tid] += 1
            # A cached heatmap that's not brand-new is no longer wanted --
            # the "wanted" flag applies only to the fire event itself.
            if self._ages[tid] > 0:
                self._wanted[tid] = False

        for tid in list(self._heatmaps):
            if tid not in active_tids:
                self._heatmaps.pop(tid, None)
                self._ages.pop(tid, None)
                self._inout_scores.pop(tid, None)
                self._wanted.pop(tid, None)
                continue
            if self._max_age > 0 and self._ages.get(tid, 0) > self._max_age:
                self._heatmaps.pop(tid, None)
                self._ages.pop(tid, None)
                self._inout_scores.pop(tid, None)
                self._wanted.pop(tid, None)

    @property
    def track_ids(self) -> set[int]:
        """Currently cached track IDs."""
        return set(self._heatmaps)
