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
from pathlib import Path
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# Participant-ID display helper
# ══════════════════════════════════════════════════════════════════════════════

def resolve_display_pid(track_id: int,
                        pid_map: Optional[dict[int, str]] = None) -> str:
    """Return the display label for an internal track ID.

    If *pid_map* is provided and contains *track_id*, the custom label is
    returned.  Otherwise falls back to ``"P{track_id}"``.
    """
    if pid_map and track_id in pid_map:
        return pid_map[track_id]
    return f"P{track_id}"


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
    adaptive_ray: str = "off"          # "off", "extend", or "snap"
    snap_dist: float = 150.0
    snap_bbox_scale: float = 0.0      # fraction of bbox half-diag added to snap radius
    snap_w_dist: float = 1.0          # scoring weight: normalized distance penalty
    snap_w_size: float = 0.0          # scoring weight: angular size reward (off by default)
    snap_w_intersect: float = 0.5     # scoring bonus for ray-bbox intersection
    conf_ray: bool = False
    gaze_tips: bool = False
    tip_radius: int = 80
    gaze_cone_angle: float = 0.0
    hit_conf_gate: float = 0.0
    detect_extend: float = 0.0            # extra pixels past visual ray (0 = visual parity)
    detect_extend_scope: str = "objects"   # "objects", "phenomena", or "both"
    ja_quorum: float = 1.0
    gaze_debug: bool = False
    forward_gaze_threshold: float = 5.0

    @classmethod
    def from_namespace(cls, ns) -> GazeConfig:
        """Construct from an ``argparse.Namespace``."""
        return cls(
            ray_length=ns.ray_length,
            adaptive_ray=ns.adaptive_ray,
            snap_dist=ns.snap_dist,
            snap_bbox_scale=getattr(ns, 'snap_bbox_scale', 0.0),
            snap_w_dist=getattr(ns, 'snap_w_dist', 1.0),
            snap_w_size=getattr(ns, 'snap_w_size', 0.0),
            snap_w_intersect=getattr(ns, 'snap_w_intersect', 0.5),
            conf_ray=ns.conf_ray,
            gaze_tips=ns.gaze_tips,
            tip_radius=ns.tip_radius,
            gaze_cone_angle=ns.gaze_cone,
            hit_conf_gate=getattr(ns, 'hit_conf_gate', 0.0),
            detect_extend=getattr(ns, 'detect_extend', 0.0),
            detect_extend_scope=getattr(ns, 'detect_extend_scope', 'objects'),
            ja_quorum=ns.ja_quorum,
            gaze_debug=ns.gaze_debug,
            forward_gaze_threshold=ns.forward_gaze_threshold,
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
    reid_max_dist: int = 200

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
            reid_max_dist=getattr(ns, 'reid_max_dist', 200),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Output configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AuxStreamConfig:
    """Configuration for a single auxiliary video stream mapped to a participant.

    Auxiliary streams are optional per-participant video feeds (e.g. a
    dedicated eye-tracking camera or a first-person-view camera) that are
    frame-synchronised with the main source.  They are exposed in
    ``FrameContext['aux_frames']`` for consumption by plugins but are
    **not** processed by any built-in pipeline stage.
    """

    pid: str            # participant label (e.g. "S70")
    stream_type: str    # purpose tag (e.g. "eye_camera", "first_person_view")
    source: str         # file path or device index string


@dataclass
class OutputConfig:
    """Paths and flags controlling run-loop outputs (video, CSV, heatmaps)."""

    save: bool | str | None = None
    log_path: str | None = None
    summary_path: str | None = None
    heatmap_path: str | None = None
    charts_path: bool | str | None = None
    pid_map: dict[int, str] | None = None
    aux_streams: list[AuxStreamConfig] | None = None
    anonymize: str | None = None          # "blur" or "black"; None = disabled
    anonymize_padding: float = 0.3        # fraction of bbox size added as margin
    video_name: str | None = None         # source video stem (project mode only)
    conditions: str | None = None         # pipe-delimited tags (project mode only)

    @classmethod
    def from_namespace(cls, ns) -> OutputConfig:
        """Construct from an ``argparse.Namespace``."""
        return cls(
            save=ns.save,
            log_path=ns.log,
            summary_path=ns.summary,
            heatmap_path=ns.heatmap,
            charts_path=getattr(ns, 'charts', None),
            pid_map=getattr(ns, 'pid_map', None),
            aux_streams=getattr(ns, 'aux_streams', None),
            anonymize=getattr(ns, 'anonymize', None),
            anonymize_padding=getattr(ns, 'anonymize_padding', 0.3),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Project-level configuration (loaded from project.yaml)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectOutputConfig:
    """Controls where project-level outputs are written."""

    directory: str | None = None          # output root (absolute or relative to project)

    def resolve_root(self, project: Path) -> Path:
        """Return the resolved output root directory."""
        if self.directory:
            out = Path(self.directory)
            return out if out.is_absolute() else project / out
        return project / "Outputs"


@dataclass
class ProjectConfig:
    """Project metadata loaded from ``project.yaml``.

    Stores study-level information (condition tags, participant labels,
    output settings) that is orthogonal to pipeline processing parameters
    stored in ``pipeline.yaml``.
    """

    pipeline_path: str | None = None
    conditions: dict[str, list[str]] = field(default_factory=dict)
    participants: dict[str, dict[int, str]] = field(default_factory=dict)
    output: ProjectOutputConfig = field(default_factory=ProjectOutputConfig)
