"""[SP1.5 shim] moved to mindsight.utils.geometry; delete in SP1.6."""
import sys
import mindsight.utils.geometry
sys.modules[__name__] = mindsight.utils.geometry
