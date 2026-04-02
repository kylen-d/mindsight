"""
pipeline_config.py — Dataclass-based configuration for the MindSight pipeline.

Groups the many parameters of ``process_frame()`` and ``run()`` into a small
number of opaque config objects so that adding a new parameter no longer
requires touching every call site.

Includes ``FrameContext``, the mutable per-frame data carrier that flows
through all pipeline stages.  Each stage reads what it needs and writes its
results; plugins access only the keys they care about via ``**kwargs``.

See also ``Phenomena/phenomena_config.py`` for phenomena-specific config.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# Per-frame pipeline context
# ══════════════════════════════════════════════════════════════════════════════

class FrameContext:
    """Mutable context passed through all pipeline stages.

    Each stage reads what it needs and writes its results.
    Plugins access only the keys they care about.

    Supports dict-like access::

        ctx = FrameContext(frame=img, frame_no=42)
        ctx['objects'] = detected_objects
        objs = ctx.get('objects', [])
        if 'hits' in ctx: ...
    """

    __slots__ = ('data',)

    def __init__(self, frame=None, frame_no: int = 0, **kwargs):
        self.data: dict = {'frame': frame, 'frame_no': frame_no, **kwargs}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, mapping):
        self.data.update(mapping)

    def as_kwargs(self) -> dict:
        """Return a shallow copy of the context data for ``**kwargs`` unpacking."""
        return dict(self.data)


# ══════════════════════════════════════════════════════════════════════════════
# Gaze configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GazeConfig:
    """All gaze-estimation and ray-intersection tuning parameters."""

    ray_length: float = 1.0
    adaptive_ray: bool = False
    snap_dist: float = 150.0
    adaptive_snap_mode: bool = False
    conf_ray: bool = False
    gaze_tips: bool = False
    tip_radius: int = 80
    gaze_cone_angle: float = 0.0
    ja_conf_gate: float = 0.0
    ja_quorum: float = 1.0
    gaze_debug: bool = False

    @classmethod
    def from_namespace(cls, ns) -> GazeConfig:
        """Construct from an ``argparse.Namespace``."""
        return cls(
            ray_length=ns.ray_length,
            adaptive_ray=ns.adaptive_ray,
            snap_dist=ns.snap_dist,
            adaptive_snap_mode=ns.adaptive_snap,
            conf_ray=ns.conf_ray,
            gaze_tips=ns.gaze_tips,
            tip_radius=ns.tip_radius,
            gaze_cone_angle=ns.gaze_cone,
            ja_conf_gate=ns.ja_conf_gate,
            ja_quorum=ns.ja_quorum,
            gaze_debug=ns.gaze_debug,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Detection configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionConfig:
    """Object-detection parameters passed through to YOLO."""

    conf: float = 0.35
    class_ids: list | None = None
    blacklist: set = field(default_factory=set)
    detect_scale: float = 1.0

    @classmethod
    def from_namespace(cls, ns, class_ids, blacklist) -> DetectionConfig:
        """Construct from an ``argparse.Namespace`` plus resolved IDs/blacklist."""
        return cls(
            conf=ns.conf,
            class_ids=class_ids,
            blacklist=blacklist,
            detect_scale=ns.detect_scale,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tracker construction configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackerConfig:
    """Parameters used by ``run()`` to construct per-run tracker instances."""

    gaze_lock: bool = False
    dwell_frames: int = 15
    lock_dist: int = 100
    skip_frames: int = 1
    obj_persistence: int = 0
    snap_switch_frames: int = 8
    reid_grace_seconds: float = 1.0

    @classmethod
    def from_namespace(cls, ns) -> TrackerConfig:
        """Construct from an ``argparse.Namespace``."""
        return cls(
            gaze_lock=ns.gaze_lock,
            dwell_frames=ns.dwell_frames,
            lock_dist=ns.lock_dist,
            skip_frames=ns.skip_frames,
            obj_persistence=ns.obj_persistence,
            snap_switch_frames=ns.snap_switch_frames,
            reid_grace_seconds=ns.reid_grace_seconds,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Output configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OutputConfig:
    """Paths and flags controlling run-loop outputs (video, CSV, heatmaps)."""

    save: bool | str | None = None
    log_path: str | None = None
    summary_path: str | None = None
    heatmap_path: str | None = None

    @classmethod
    def from_namespace(cls, ns) -> OutputConfig:
        """Construct from an ``argparse.Namespace``."""
        return cls(
            save=ns.save,
            log_path=ns.log,
            summary_path=ns.summary,
            heatmap_path=ns.heatmap,
        )
