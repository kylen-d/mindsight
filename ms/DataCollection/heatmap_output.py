"""[SP1.5 shim] moved to mindsight.DataCollection.heatmap_output; delete in SP1.6."""
import sys
import mindsight.DataCollection.heatmap_output
sys.modules[__name__] = mindsight.DataCollection.heatmap_output
