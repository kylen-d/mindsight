"""[SP1.5 shim] moved to mindsight.DepthEstimation.depth_backend; delete in SP1.6."""
import sys
import mindsight.DepthEstimation.depth_backend
sys.modules[__name__] = mindsight.DepthEstimation.depth_backend
