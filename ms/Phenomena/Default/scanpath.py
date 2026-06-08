"""[SP1.5 shim] moved to mindsight.Phenomena.Default.scanpath; delete in SP1.6."""
import sys
import mindsight.Phenomena.Default.scanpath
sys.modules[__name__] = mindsight.Phenomena.Default.scanpath
