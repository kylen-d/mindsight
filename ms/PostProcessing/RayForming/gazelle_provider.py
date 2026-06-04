"""[SP1.5 shim] moved to mindsight.PostProcessing.RayForming.gazelle_provider; delete in SP1.6."""
import sys
import mindsight.PostProcessing.RayForming.gazelle_provider
sys.modules[__name__] = mindsight.PostProcessing.RayForming.gazelle_provider
