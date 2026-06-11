"""[SP1.5 shim] moved to mindsight.GUI.arg_introspector; delete in SP1.6."""
import sys
import mindsight.GUI.arg_introspector
sys.modules[__name__] = mindsight.GUI.arg_introspector
