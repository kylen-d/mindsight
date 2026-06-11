"""[SP1.5 shim] moved to mindsight.GUI.settings_manager; delete in SP1.6."""
import sys
import mindsight.GUI.settings_manager
sys.modules[__name__] = mindsight.GUI.settings_manager
