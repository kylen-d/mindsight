"""[SP1.5 shim] moved to mindsight.PostProcessing.RayForming.heatmap_cache; delete in SP1.6."""
import sys
import mindsight.PostProcessing.RayForming.heatmap_cache
sys.modules[__name__] = mindsight.PostProcessing.RayForming.heatmap_cache
