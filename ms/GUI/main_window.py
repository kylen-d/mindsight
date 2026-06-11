"""[SP1.5 shim] moved to mindsight.GUI.main_window; delete in SP1.6."""
import sys
import mindsight.GUI.main_window
sys.modules[__name__] = mindsight.GUI.main_window
