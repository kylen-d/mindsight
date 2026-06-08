"""[SP1.5 shim] moved to mindsight.io.sources; delete in SP1.6."""
import sys
import mindsight.io.sources
sys.modules[__name__] = mindsight.io.sources
