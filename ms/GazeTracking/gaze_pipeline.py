"""[SP1.5 shim] moved to mindsight.GazeTracking.gaze_pipeline; delete in SP1.6."""
import sys
import mindsight.GazeTracking.gaze_pipeline
sys.modules[__name__] = mindsight.GazeTracking.gaze_pipeline
