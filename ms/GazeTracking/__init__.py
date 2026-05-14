"""
GazeTracking — Gaze estimation and ray-intersection pipeline stage.

Coordinates face-level gaze estimation (via pluggable backends), temporal
smoothing with ReID, snap-to-object scoring, and dwell-based gaze locking.
"""

from .gaze_factory import create_gaze_engine
from .gaze_pipeline import run_gaze_step
from .gaze_processing import (
    GazeLockTracker,
    GazeSmootherReID,
    GazeToolkit,
    SnapTemporalState,
)
from .pitchyaw_pipeline import run_pitchyaw_pipeline

__all__ = [
    "run_gaze_step",
    "run_pitchyaw_pipeline",
    "GazeSmootherReID",
    "GazeLockTracker",
    "SnapTemporalState",
    "GazeToolkit",
    "create_gaze_engine",
]
