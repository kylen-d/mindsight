"""[SP1.5 shim] moved to mindsight.constants; delete in SP1.6."""
import sys
import mindsight.constants
sys.modules[__name__] = mindsight.constants
