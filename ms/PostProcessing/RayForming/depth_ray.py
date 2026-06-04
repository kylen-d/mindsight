"""[SP1.5 shim] moved to mindsight.PostProcessing.RayForming.depth_ray; delete in SP1.6."""
import sys
import mindsight.PostProcessing.RayForming.depth_ray
sys.modules[__name__] = mindsight.PostProcessing.RayForming.depth_ray
