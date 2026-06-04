"""
PostProcessing.RayForming — Unified ray forming pipeline.

Converts raw gaze estimates (pitch/yaw angles + optional Gaze-LLE heatmaps)
into finalized gaze rays through a chain of belief blending, depth adjustment,
object snapping, fixation lock-on, and hit detection.

Public API
----------
    from mindsight.PostProcessing.RayForming import (
        run_ray_forming,
        RayFormingConfig,
        GazeLLEBlender,
        HeatmapCache,
        ObjectSnap,
        GazeLockTracker,
        compute_ray_intersections,
    )
"""
from mindsight.PostProcessing.RayForming.ray_config import RayFormingConfig
from mindsight.PostProcessing.RayForming.heatmap_cache import HeatmapCache
from mindsight.PostProcessing.RayForming.gazelle_blender import GazeLLEBlender
from mindsight.PostProcessing.RayForming.object_snap import (
    ObjectSnap,
    SmoothSnapTracker,
    SnapTemporalState,
    snap_score,
    apply_tip_snapping,
)
from mindsight.PostProcessing.RayForming.fixation import GazeLockTracker, apply_lock_on
from mindsight.PostProcessing.RayForming.hit_detection import compute_ray_intersections
from mindsight.PostProcessing.RayForming.gazelle_provider import GazelleProvider
from mindsight.PostProcessing.RayForming.ray_pipeline import run_ray_forming, RawGaze

__all__ = [
    "run_ray_forming",
    "RawGaze",
    "RayFormingConfig",
    "GazeLLEBlender",
    "HeatmapCache",
    "ObjectSnap",
    "SmoothSnapTracker",
    "SnapTemporalState",
    "snap_score",
    "apply_tip_snapping",
    "GazeLockTracker",
    "apply_lock_on",
    "compute_ray_intersections",
    "GazelleProvider",
]
