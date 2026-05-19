"""
RayForming/inference_scheduler.py -- Fixation-aware Gaze-LLE scheduler.

Owns per-face PYHistoryBuffer + FixationDetector state, plus global and
per-face rate-limit counters.  Each frame, ``tick()`` returns
``(should_fire, wanting_tids)`` -- whether to run a Gaze-LLE call this
frame, and which face tracks want the result applied to their belief.

The caller pattern is:

    for each face this frame:
        scheduler.observe(track_id, py_dir, py_conf)
    should_fire, wanting_tids = scheduler.tick()
    if should_fire:
        heatmaps = gaze_lle_engine.run(active_faces)   # batched
        # For each face, blender.update(...) knows whether to APPLY the
        # cached heatmap based on wanting_tids membership.
        scheduler.record_accepted(wanting_tids)
    scheduler.forget(inactive_tids=faces_that_disappeared)
    scheduler.advance_frame()

Internal thresholds (P_ACCEPT, MIN_FACE_REFRESH, PY_CONF_FLOOR) are
NOT user-tunable -- they set the shape of the response, and per the
design spec (section 8) their values are internal constants, not
tuning axes.
"""
from __future__ import annotations

import numpy as np

from ms.PostProcessing.RayForming.py_history import PYHistoryBuffer
from ms.PostProcessing.RayForming.fixation_detector import FixationDetector


P_ACCEPT: float = 0.7
"""fixation_likelihood threshold at which a face is deemed to want inference."""

MIN_FACE_REFRESH: int = 5
"""Minimum frames between accepted inferences for a single face."""

PY_CONF_FLOOR: float = 0.5
"""Minimum PY gaze_conf for a face to be considered a valid target
for inference.  Prevents anchoring on garbage face crops.
"""

PY_HISTORY_SIZE: int = 10


class InferenceScheduler:
    """Per-face fixation gating + global rate limit for Gaze-LLE calls."""

    def __init__(self, *, v_threshold: float, d_threshold: float,
                 min_call_gap: int):
        self.v_threshold = float(v_threshold)
        self.d_threshold = float(d_threshold)
        self.min_call_gap = int(min_call_gap)

        self._buffers: dict[int, PYHistoryBuffer] = {}
        self._detectors: dict[int, FixationDetector] = {}
        self._likelihoods: dict[int, float] = {}
        self._py_confs: dict[int, float] = {}
        self._frames_since_last_accept: dict[int, int] = {}
        self._has_latched: dict[int, bool] = {}
        self._frames_since_last_global_call: int = 10**9  # allow first fire

    def observe(self, *, track_id: int, py_dir: np.ndarray,
                py_conf: float) -> None:
        """Push one PY observation for a face this frame."""
        if track_id not in self._buffers:
            self._buffers[track_id] = PYHistoryBuffer(size=PY_HISTORY_SIZE)
            self._detectors[track_id] = FixationDetector(
                v_threshold=self.v_threshold, d_threshold=self.d_threshold)
            self._frames_since_last_accept[track_id] = 10**9  # bootstrap
            self._has_latched[track_id] = False
        self._buffers[track_id].push(py_dir)
        self._likelihoods[track_id] = self._detectors[track_id].update(
            self._buffers[track_id])
        self._py_confs[track_id] = float(py_conf)

    def tick(self) -> tuple[bool, set[int]]:
        """End-of-frame decision: fire this frame? which faces want it?"""
        wanting: set[int] = set()
        for tid, likelihood in self._likelihoods.items():
            if likelihood < P_ACCEPT:
                continue
            if self._py_confs.get(tid, 0.0) < PY_CONF_FLOOR:
                continue
            # Bootstrap: never latched -> want ASAP.  Otherwise enforce
            # per-face min-refresh timer.
            if self._has_latched[tid]:
                if self._frames_since_last_accept[tid] < MIN_FACE_REFRESH:
                    continue
            wanting.add(tid)

        should_fire = bool(wanting) and (
            self._frames_since_last_global_call >= self.min_call_gap)
        return should_fire, wanting if should_fire else set()

    def record_accepted(self, wanting: set[int]) -> None:
        """Caller notifies which faces just got their fresh inference.

        Also resets the global call-gap counter -- callers only invoke
        this immediately after actually firing a Gaze-LLE call.
        """
        for tid in wanting:
            self._frames_since_last_accept[tid] = 0
            self._has_latched[tid] = True
        self._frames_since_last_global_call = 0

    def advance_frame(self) -> None:
        """Increment all rate-limit counters by one frame.  Call once per
        frame AFTER tick()/record_accepted()."""
        for tid in list(self._frames_since_last_accept):
            self._frames_since_last_accept[tid] += 1
        self._frames_since_last_global_call += 1

    def forget(self, inactive_tids: set[int]) -> None:
        """Drop state for tracks that no longer exist."""
        for tid in inactive_tids:
            self._buffers.pop(tid, None)
            self._detectors.pop(tid, None)
            self._likelihoods.pop(tid, None)
            self._py_confs.pop(tid, None)
            self._frames_since_last_accept.pop(tid, None)
            self._has_latched.pop(tid, None)

    @property
    def tracked_tids(self) -> set[int]:
        """Track IDs the scheduler currently holds state for.

        Used by the provider to prune disappeared tracks without
        reaching into private state.
        """
        return set(self._buffers)

    def likelihood(self, track_id: int) -> float:
        """Read the last-computed fixation_likelihood for a track.

        The blender needs this to weight belief vs PY at the output.
        Returns 0.0 for unknown tracks.
        """
        return float(self._likelihoods.get(track_id, 0.0))
