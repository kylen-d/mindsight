"""[SP1.5 shim] moved to mindsight.utils.device; delete in SP1.6."""
import sys
import mindsight.utils.device
sys.modules[__name__] = mindsight.utils.device
