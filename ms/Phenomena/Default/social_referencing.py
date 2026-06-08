"""[SP1.5 shim] moved to mindsight.Phenomena.Default.social_referencing; delete in SP1.6."""
import sys
import mindsight.Phenomena.Default.social_referencing
sys.modules[__name__] = mindsight.Phenomena.Default.social_referencing
