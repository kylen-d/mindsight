"""[SP1.5 shim] moved to mindsight.PostProcessing.RayForming.object_snap; delete in SP1.6."""
import sys
import mindsight.PostProcessing.RayForming.object_snap
sys.modules[__name__] = mindsight.PostProcessing.RayForming.object_snap
