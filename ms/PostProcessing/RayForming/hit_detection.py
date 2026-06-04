"""[SP1.5 shim] moved to mindsight.PostProcessing.RayForming.hit_detection; delete in SP1.6."""
import sys
import mindsight.PostProcessing.RayForming.hit_detection
sys.modules[__name__] = mindsight.PostProcessing.RayForming.hit_detection
