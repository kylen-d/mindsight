"""[SP1.5 shim] moved to mindsight.PostProcessing.RayForming.inference_scheduler; delete in SP1.6."""
import sys
import mindsight.PostProcessing.RayForming.inference_scheduler
sys.modules[__name__] = mindsight.PostProcessing.RayForming.inference_scheduler
