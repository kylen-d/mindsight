"""[SP1.5 shim] moved to mindsight.pipeline; delete in SP1.6."""
import sys
import mindsight.pipeline
sys.modules[__name__] = mindsight.pipeline
