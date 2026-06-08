"""[SP1.5 shim] moved to mindsight.Phenomena.phenomena_tracking; delete in SP1.6."""
import sys
import mindsight.Phenomena.phenomena_tracking
sys.modules[__name__] = mindsight.Phenomena.phenomena_tracking
