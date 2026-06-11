"""[SP1.5 shim] moved to mindsight.GUI.plugin_panel; delete in SP1.6."""
import sys
import mindsight.GUI.plugin_panel
sys.modules[__name__] = mindsight.GUI.plugin_panel
