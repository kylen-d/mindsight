"""[SP1.5 shim] moved to mindsight.GUI.live_dashboard_bridge; delete in SP1.6."""
import sys
import mindsight.GUI.live_dashboard_bridge
sys.modules[__name__] = mindsight.GUI.live_dashboard_bridge
