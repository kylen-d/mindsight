"""
GazeTracking — Gaze estimation and ray-intersection pipeline stage.

Coordinates face-level gaze estimation (via pluggable backends), temporal
smoothing with ReID, snap-to-object hysteresis, and dwell-based gaze locking.
"""

from .gaze_pipeline import run_gaze_step
from .gaze_processing import (
    GazeSmootherReID,
    GazeLockTracker,
    SnapHysteresisTracker,
    GazeToolkit,
)
from .gaze_factory import create_gaze_engine

__all__ = [
    "run_gaze_step",
    "GazeSmootherReID",
    "GazeLockTracker",
    "SnapHysteresisTracker",
    "GazeToolkit",
    "create_gaze_engine",
]
