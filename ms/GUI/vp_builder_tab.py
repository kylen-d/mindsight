"""[SP1.5 shim] moved to mindsight.GUI.vp_builder_tab; delete in SP1.6."""
import sys
import mindsight.GUI.vp_builder_tab
sys.modules[__name__] = mindsight.GUI.vp_builder_tab
