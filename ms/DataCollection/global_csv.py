"""[SP1.5 shim] moved to mindsight.DataCollection.global_csv; delete in SP1.6."""
import sys
import mindsight.DataCollection.global_csv
sys.modules[__name__] = mindsight.DataCollection.global_csv
