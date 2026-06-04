"""[SP1.5 shim] moved to mindsight.PostProcessing.RayForming.gazelle_blender; delete in SP1.6."""
import sys
import mindsight.PostProcessing.RayForming.gazelle_blender
sys.modules[__name__] = mindsight.PostProcessing.RayForming.gazelle_blender
