"""[SP1.5 shim] moved to mindsight.Phenomena.helpers; delete in SP1.6."""
import sys
import mindsight.Phenomena.helpers
sys.modules[__name__] = mindsight.Phenomena.helpers
