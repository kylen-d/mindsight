"""
mindsight.DepthEstimation -- Monocular depth estimation pipeline stage.

Provides a ``DepthBackend`` protocol and a lightweight MiDaS implementation
for injecting per-frame depth maps into the MindSight pipeline.
"""

from mindsight.DepthEstimation.depth_backend import DepthBackend, create_depth_backend
from mindsight.DepthEstimation.depth_pipeline import run_depth_step

__all__ = ["DepthBackend", "create_depth_backend", "run_depth_step"]
