"""
Plugins/Phenomena/EyeMovement/classifiers.py -- I-VT eye movement classifier.

Implements Identification by Velocity Threshold (I-VT) to classify gaze
samples into fixation, saccade, and smooth pursuit.  Includes post-processing
for segment merging and minimum-duration rejection.
"""

from __future__ import annotations

import collections
from enum import Enum

import numpy as np


class EyeState(Enum):
    FIXATION = "fixation"
    SACCADE = "saccade"
    SMOOTH_PURSUIT = "pursuit"


class IVTClassifier:
    """
    Per-participant I-VT classifier with median-filtered velocity and
    post-processing for segment merging.

    Parameters
    ----------
    saccade_threshold  : Velocity above this -> saccade (px/frame).
    fixation_threshold : Velocity below this -> fixation (px/frame).
    min_fixation_frames : Reject fixation segments shorter than this.
    min_saccade_frames  : Reject saccade segments shorter than this (default 1).
    velocity_window     : Median filter window for velocity smoothing.
    min_fixation_gap    : Merge adjacent fixation segments separated by
                          fewer than this many non-fixation frames.
    """

    def __init__(
        self,
        saccade_threshold: float = 30.0,
        fixation_threshold: float = 10.0,
        min_fixation_frames: int = 4,
        min_saccade_frames: int = 1,
        velocity_window: int = 3,
        min_fixation_gap: int = 2,
    ) -> None:
        self.saccade_thresh = saccade_threshold
        self.fixation_thresh = fixation_threshold
        self.min_fix_frames = min_fixation_frames
        self.min_sac_frames = min_saccade_frames
        self.vel_window = max(1, velocity_window)
        self.min_fix_gap = min_fixation_gap

        # Rolling velocity buffer for median filtering
        self._vel_buf: collections.deque = collections.deque(
            maxlen=self.vel_window
        )
        self._prev_pos: np.ndarray | None = None
        self._state = EyeState.FIXATION
        self._state_frames = 0

        # Current segment tracking
        self._segment_start: int = 0
        self._segment_peak_vel: float = 0.0

        # Completed events
        self.events: list[dict] = []

    def reset(self) -> None:
        self._vel_buf.clear()
        self._prev_pos = None
        self._state = EyeState.FIXATION
        self._state_frames = 0
        self._segment_start = 0
        self._segment_peak_vel = 0.0

    def classify(self, position: np.ndarray, frame_no: int,
                 skip: bool = False) -> EyeState:
        """
        Classify one frame's eye movement.

        Parameters
        ----------
        position : 2D position (x, y) in pixels.
        frame_no : Current frame number.
        skip     : If True, skip this frame (e.g. ray_snapped artifact).

        Returns
        -------
        Current EyeState classification.
        """
        pos = np.asarray(position, dtype=np.float64)

        if skip or self._prev_pos is None:
            self._prev_pos = pos
            return self._state

        # Compute instantaneous velocity
        displacement = pos - self._prev_pos
        velocity = float(np.linalg.norm(displacement))
        self._prev_pos = pos

        # Median filter
        self._vel_buf.append(velocity)
        filtered_vel = float(np.median(list(self._vel_buf)))

        # Classify
        if filtered_vel > self.saccade_thresh:
            new_state = EyeState.SACCADE
        elif filtered_vel < self.fixation_thresh:
            new_state = EyeState.FIXATION
        else:
            new_state = EyeState.SMOOTH_PURSUIT

        # State transition -- emit event for completed segment
        if new_state != self._state:
            self._emit_segment(frame_no, filtered_vel)
            self._state = new_state
            self._state_frames = 1
            self._segment_start = frame_no
            self._segment_peak_vel = filtered_vel
        else:
            self._state_frames += 1
            self._segment_peak_vel = max(self._segment_peak_vel, filtered_vel)

        return self._state

    def _emit_segment(self, end_frame: int, last_vel: float) -> None:
        """Record the just-finished segment as an event."""
        duration = self._state_frames

        # Reject short segments
        if self._state == EyeState.FIXATION and duration < self.min_fix_frames:
            return
        if self._state == EyeState.SACCADE and duration < self.min_sac_frames:
            return

        self.events.append({
            'type': self._state.value,
            'start_frame': self._segment_start,
            'end_frame': end_frame - 1,
            'duration_frames': duration,
            'peak_velocity': self._segment_peak_vel,
        })

    def finalize(self, total_frames: int) -> None:
        """Emit the last open segment and run post-processing."""
        if self._state_frames > 0:
            self._emit_segment(total_frames, 0.0)

        self._merge_fixation_gaps()

    def _merge_fixation_gaps(self) -> None:
        """Merge adjacent fixation events separated by short gaps."""
        if len(self.events) < 3:
            return

        merged = [self.events[0]]
        for ev in self.events[1:]:
            prev = merged[-1]
            if (prev['type'] == 'fixation' and ev['type'] == 'fixation' and
                    ev['start_frame'] - prev['end_frame'] <= self.min_fix_gap):
                # Merge
                prev['end_frame'] = ev['end_frame']
                prev['duration_frames'] = prev['end_frame'] - prev['start_frame'] + 1
                prev['peak_velocity'] = max(prev['peak_velocity'], ev['peak_velocity'])
            else:
                merged.append(ev)
        self.events = merged

    @property
    def state(self) -> EyeState:
        return self._state

    def summary_stats(self) -> dict:
        """Compute summary statistics from recorded events."""
        fix_events = [e for e in self.events if e['type'] == 'fixation']
        sac_events = [e for e in self.events if e['type'] == 'saccade']

        fix_durations = [e['duration_frames'] for e in fix_events]
        sac_vels = [e['peak_velocity'] for e in sac_events]

        total_frames = sum(e['duration_frames'] for e in self.events) or 1

        return {
            'fixation_count': len(fix_events),
            'saccade_count': len(sac_events),
            'mean_fixation_duration': float(np.mean(fix_durations)) if fix_durations else 0.0,
            'mean_saccade_velocity': float(np.mean(sac_vels)) if sac_vels else 0.0,
            'fixation_pct': sum(fix_durations) / total_frames * 100,
            'saccade_pct': sum(e['duration_frames'] for e in sac_events) / total_frames * 100,
        }
