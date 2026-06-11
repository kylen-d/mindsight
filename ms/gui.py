"""[SP1.5 shim] moved to mindsight.gui; delete in SP1.6."""
import sys
import mindsight.gui
sys.modules[__name__] = mindsight.gui
