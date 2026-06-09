"""[SP1.5 shim] moved to mindsight.cli_flags; delete in SP1.6."""
import sys
import mindsight.cli_flags
sys.modules[__name__] = mindsight.cli_flags
