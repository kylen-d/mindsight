"""[SP1.5 shim] moved to mindsight.DataCollection.data_pipeline; delete in SP1.6."""
import sys
import mindsight.DataCollection.data_pipeline
sys.modules[__name__] = mindsight.DataCollection.data_pipeline
