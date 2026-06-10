"""[SP1.5 shim] moved to mindsight.project_runner; delete in SP1.6."""
import sys
import mindsight.project_runner
sys.modules[__name__] = mindsight.project_runner
