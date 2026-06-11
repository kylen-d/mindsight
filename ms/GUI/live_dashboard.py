"""[SP1.5 shim] moved to mindsight.GUI.live_dashboard; delete in SP1.6."""
import sys
import mindsight.GUI.live_dashboard
sys.modules[__name__] = mindsight.GUI.live_dashboard
