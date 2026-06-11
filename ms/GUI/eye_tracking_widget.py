"""[SP1.5 shim] moved to mindsight.GUI.eye_tracking_widget; delete in SP1.6."""
import sys
import mindsight.GUI.eye_tracking_widget
sys.modules[__name__] = mindsight.GUI.eye_tracking_widget
