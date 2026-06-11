"""[SP1.5 shim] moved to mindsight.GUI.gaze_tab.gaze_tab; delete in SP1.6."""
import sys
import mindsight.GUI.gaze_tab.gaze_tab
sys.modules[__name__] = mindsight.GUI.gaze_tab.gaze_tab
