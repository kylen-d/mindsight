"""[SP1.5 shim] moved to mindsight.Phenomena.Default.mutual_gaze; delete in SP1.6."""
import sys
import mindsight.Phenomena.Default.mutual_gaze
sys.modules[__name__] = mindsight.Phenomena.Default.mutual_gaze
