"""[SP1.5 shim] moved to mindsight.config_compat; delete in SP1.6."""
import sys
import mindsight.config_compat
sys.modules[__name__] = mindsight.config_compat
