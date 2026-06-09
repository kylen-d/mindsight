"""[SP1.5 shim] moved to mindsight.config; delete in SP1.6."""
import sys
import mindsight.config
sys.modules[__name__] = mindsight.config
