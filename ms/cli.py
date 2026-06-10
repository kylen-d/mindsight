"""[SP1.5 shim] moved to mindsight.cli; delete in SP1.6."""
import sys
import mindsight.cli
sys.modules[__name__] = mindsight.cli
