"""[SP1.5 shim] moved to mindsight.GUI.widgets; delete in SP1.6."""
import sys
import mindsight.GUI.widgets
sys.modules[__name__] = mindsight.GUI.widgets
