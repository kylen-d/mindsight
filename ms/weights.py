"""[SP1.5 shim] moved to mindsight.weights; delete in SP1.6."""
import sys
import mindsight.weights
sys.modules[__name__] = mindsight.weights
