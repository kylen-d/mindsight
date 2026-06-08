"""[SP1.5 shim] moved to mindsight.Phenomena.Default.gaze_aversion; delete in SP1.6."""
import sys
import mindsight.Phenomena.Default.gaze_aversion
sys.modules[__name__] = mindsight.Phenomena.Default.gaze_aversion
