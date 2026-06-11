"""[SP1.5 shim] moved to mindsight.GUI.project_tab.conditions_section; delete in SP1.6."""
import sys
import mindsight.GUI.project_tab.conditions_section
sys.modules[__name__] = mindsight.GUI.project_tab.conditions_section
