"""[SP1.5 shim] moved to mindsight.GUI.project_tab.project_tab; delete in SP1.6."""
import sys
import mindsight.GUI.project_tab.project_tab
sys.modules[__name__] = mindsight.GUI.project_tab.project_tab
