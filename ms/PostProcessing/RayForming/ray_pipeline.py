"""[SP1.5 shim] moved to mindsight.PostProcessing.RayForming.ray_pipeline; delete in SP1.6."""
import sys
import mindsight.PostProcessing.RayForming.ray_pipeline
sys.modules[__name__] = mindsight.PostProcessing.RayForming.ray_pipeline
