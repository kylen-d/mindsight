"""[SP1.5 shim] moved to mindsight.utils.one_euro; delete in SP1.6."""
import sys
import mindsight.utils.one_euro
sys.modules[__name__] = mindsight.utils.one_euro
