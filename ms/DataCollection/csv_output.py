"""[SP1.5 shim] moved to mindsight.DataCollection.csv_output; delete in SP1.6."""
import sys
import mindsight.DataCollection.csv_output
sys.modules[__name__] = mindsight.DataCollection.csv_output
