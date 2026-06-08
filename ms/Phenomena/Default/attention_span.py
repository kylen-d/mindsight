"""[SP1.5 shim] moved to mindsight.Phenomena.Default.attention_span; delete in SP1.6."""
import sys
import mindsight.Phenomena.Default.attention_span
sys.modules[__name__] = mindsight.Phenomena.Default.attention_span
