"""[SP1.5 shim] moved to mindsight.PostProcessing.RayForming.fixation_detector; delete in SP1.6."""
import sys
import mindsight.PostProcessing.RayForming.fixation_detector
sys.modules[__name__] = mindsight.PostProcessing.RayForming.fixation_detector
