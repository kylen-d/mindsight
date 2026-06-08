"""[SP1.5 shim] moved to mindsight.DataCollection.chart_output; delete in SP1.6."""
import sys
import mindsight.DataCollection.chart_output
sys.modules[__name__] = mindsight.DataCollection.chart_output
