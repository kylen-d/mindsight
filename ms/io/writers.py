"""[SP1.5 shim] moved to mindsight.io.writers; delete in SP1.6."""
import sys
import mindsight.io.writers
sys.modules[__name__] = mindsight.io.writers
