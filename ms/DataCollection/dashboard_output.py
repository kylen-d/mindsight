"""[SP1.5 shim] moved to mindsight.DataCollection.dashboard_output; delete in SP1.6."""
import sys
import mindsight.DataCollection.dashboard_output
sys.modules[__name__] = mindsight.DataCollection.dashboard_output
