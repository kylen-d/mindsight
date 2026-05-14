"""
RayForming/heatmap_cache.py — Per-track Gaze-LLE heatmap storage and aging.

Manages cached 64x64 heatmaps from periodic Gaze-LLE inference, tracks their
age (frames since last refresh), and stores per-track in/out scores from
Gaze-LLE ``_inout`` model variants.
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
        self._max_age = max_age

    def update(self, track_id: int, heatmap: np.ndarray,
               inout_score: float = 1.0) -> None:
        """Store a fresh heatmap for *track_id* and reset its age to 0."""
        self._heatmaps[track_id] = heatmap
        self._ages[track_id] = 0
        self._inout_scores[track_id] = inout_score

    def get(self, track_id: int) -> tuple[np.ndarray | None, int, float]:
        """Return ``(heatmap, age, inout_score)`` or ``(None, -1, 0.0)``."""
        hm = self._heatmaps.get(track_id)
        if hm is None:
            return None, -1, 0.0
        return hm, self._ages.get(track_id, 0), self._inout_scores.get(track_id, 1.0)

    def age_all(self, active_tids: set[int]) -> None:
        """Increment age for all entries and prune tracks no longer active."""
        # Increment ages
        for tid in list(self._ages):
            self._ages[tid] += 1

        # Prune inactive tracks
        for tid in list(self._heatmaps):
            if tid not in active_tids:
                self._heatmaps.pop(tid, None)
                self._ages.pop(tid, None)
                self._inout_scores.pop(tid, None)
                continue
            # Optional max-age pruning
            if self._max_age > 0 and self._ages.get(tid, 0) > self._max_age:
                self._heatmaps.pop(tid, None)
                self._ages.pop(tid, None)
                self._inout_scores.pop(tid, None)

    @property
    def track_ids(self) -> set[int]:
        """Currently cached track IDs."""
        return set(self._heatmaps)
