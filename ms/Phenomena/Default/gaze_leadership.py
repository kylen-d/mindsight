"""[SP1.5 shim] moved to mindsight.Phenomena.Default.gaze_leadership; delete in SP1.6."""
import sys
import mindsight.Phenomena.Default.gaze_leadership
sys.modules[__name__] = mindsight.Phenomena.Default.gaze_leadership
