"""[SP1.5 shim] moved to mindsight.GUI.phenomena_panel; delete in SP1.6."""
import sys
import mindsight.GUI.phenomena_panel
sys.modules[__name__] = mindsight.GUI.phenomena_panel
