"""[SP1.5 shim] moved to mindsight.GazeTracking.pitchyaw_pipeline; delete in SP1.6."""
import sys
import mindsight.GazeTracking.pitchyaw_pipeline
sys.modules[__name__] = mindsight.GazeTracking.pitchyaw_pipeline
