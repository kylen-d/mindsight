"""[SP1.5 shim] moved to mindsight.DataCollection.dashboard_matplotlib; delete in SP1.6."""
import sys
import mindsight.DataCollection.dashboard_matplotlib
sys.modules[__name__] = mindsight.DataCollection.dashboard_matplotlib
