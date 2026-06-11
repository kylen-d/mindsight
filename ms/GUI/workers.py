"""[SP1.5 shim] moved to mindsight.GUI.workers; delete in SP1.6."""
import sys
import mindsight.GUI.workers
sys.modules[__name__] = mindsight.GUI.workers
