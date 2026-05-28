"""
config.py -- Unified pydantic-v2 schema for every MindSight pipeline parameter.

This is the single source of truth introduced in SP1.1.  Each section model
mirrors an existing runtime dataclass field-for-field (same names, types, and
defaults) so that ``ms.config_compat.to_dataclasses()`` can reconstruct the
exact objects the pipeline consumes today:

    section       mirrors dataclass
    ---------     -----------------------------------------------------------
    detection     ms.pipeline_config.DetectionConfig
    gaze          ms.pipeline_config.GazeConfig
    tracker       ms.pipeline_config.TrackerConfig
    rayforming    ms.PostProcessing.RayForming.ray_config.RayFormingConfig
    depth         ms.pipeline_config.DepthConfig
    phenomena     ms.Phenomena.phenomena_config.PhenomenaConfig
    output        ms.pipeline_config.OutputConfig
    project       ms.pipeline_config.ProjectConfig

Field metadata: fields whose argparse ``dest`` matches the field name carry
``json_schema_extra={"cli": "--the-flag"}``.  Flags whose dest differs from
the schema field name live in ``ms.config_compat.CLI_ALIASES`` instead, and
flags with no schema home are documented in
``ms.config_compat.EXCLUDED_CLI_FLAGS``.

Several argparse dests feed MORE THAN ONE schema field (e.g. ``ray_length``
populates both ``gaze.ray_length`` and ``rayforming.ray_length``, exactly as
the existing ``from_namespace`` classmethods do).  The canonical ``cli``
metadata lives on the primary owner; the mirrors are listed in
``ms.config_compat.PATH_MIRRORS``.

Weight paths are stored exactly as given -- resolution stays in
``ms.weights.resolve_weight()`` / the model factories.

SP1.2 note: ``PipelineConfig.from_namespace(ns)`` is the bridge for
``Pipeline(cfg)`` while the CLI/GUI still produce argparse namespaces.  It
mirrors the dataclass ``from_namespace`` classmethods EXACTLY and therefore
expects the same pre-processing ``cli._build_from_args`` performs today
(``--no-depth`` flipping ``ns.depth``, ``--aux-stream`` raw entries parsed
into ``ns.aux_streams``, pid-map resolution into ``ns.pid_map``).
"""
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, field_serializer

# Reuse the runtime enum so values compare equal across the schema/dataclass
# boundary, and the runtime legacy-precedence resolver so the schema can never
# drift from the behavior `RayFormingConfig.from_namespace` ships today.
from ms.pipeline_config import VideoType
from ms.PostProcessing.RayForming.ray_config import resolve_min_call_gap

_FORBID = ConfigDict(extra="forbid")


# ══════════════════════════════════════════════════════════════════════════════
# Section models
# ══════════════════════════════════════════════════════════════════════════════

class DetectionSection(BaseModel):
    """Mirrors ms.pipeline_config.DetectionConfig."""

    model_config = _FORBID

    conf: float = Field(0.35, json_schema_extra={"cli": "--conf"})
    # class_ids / blacklist hold RESOLVED values (YOLO class ids and a
    # lowercased name set).  The raw --classes / --blacklist name lists are
    # resolved against the loaded model by create_yolo_detector() inside
    # cli._build_from_args, so those flags are documented exclusions rather
    # than aliases of these fields.
    class_ids: list | None = None
    blacklist: set = Field(default_factory=set)
    detect_scale: float = Field(1.0, json_schema_extra={"cli": "--detect-scale"})
    merge_overlaps: bool = Field(False, json_schema_extra={"cli": "--merge-overlaps"})
    # KNOWN DISCREPANCY (preserved, do not "fix" silently): the
    # DetectionConfig dataclass default is 'dynamic' and the argparse default
    # is ALSO 'dynamic'; the 'filter' below matches the getattr() fallback in
    # DetectionConfig.from_namespace, which SP1.1's plan pinned as the schema
    # default.  On any real CLI run the parser supplies 'dynamic', so this
    # default is only visible to schema users who construct PipelineConfig()
    # directly or via config_compat.load_yaml.  Revisit before SP1.3
    # generates CLI flags from these defaults (see SP1.1 report).
    merge_overlap_strategy: str = Field(
        "filter", json_schema_extra={"cli": "--merge-overlap-strategy"})
    merge_overlap_threshold: float = Field(
        0.7, json_schema_extra={"cli": "--merge-overlap-threshold"})

    @field_serializer("blacklist", when_used="json")
    def _sorted_blacklist(self, v: set) -> list:
        # Sets iterate in a hash-randomized order; serialize sorted so
        # canonical_hash() is deterministic across processes.
        return sorted(v, key=str)


class GazeSection(BaseModel):
    """Mirrors ms.pipeline_config.GazeConfig."""

    model_config = _FORBID

    ray_length: float = Field(1.0, json_schema_extra={"cli": "--ray-length"})
    adaptive_ray: str = Field("off", json_schema_extra={"cli": "--adaptive-ray"})
    snap_dist: float = Field(150.0, json_schema_extra={"cli": "--snap-dist"})
    snap_bbox_scale: float = Field(0.0, json_schema_extra={"cli": "--snap-bbox-scale"})
    snap_w_dist: float = Field(1.0, json_schema_extra={"cli": "--snap-w-dist"})
    snap_w_angle: float = Field(0.8, json_schema_extra={"cli": "--snap-w-angle"})
    snap_w_size: float = Field(0.0, json_schema_extra={"cli": "--snap-w-size"})
    snap_w_intersect: float = Field(0.5, json_schema_extra={"cli": "--snap-w-intersect"})
    snap_w_temporal: float = Field(0.3, json_schema_extra={"cli": "--snap-w-temporal"})
    snap_gate_angle: float = Field(60.0, json_schema_extra={"cli": "--snap-gate-angle"})
    snap_head_blend: float = Field(0.3, json_schema_extra={"cli": "--snap-head-blend"})
    snap_quality_thresh: float = Field(0.8, json_schema_extra={"cli": "--snap-quality-thresh"})
    snap_tip_dist: float = Field(-1.0, json_schema_extra={"cli": "--snap-tip-dist"})
    snap_tip_quality: float = Field(-1.0, json_schema_extra={"cli": "--snap-tip-quality"})
    conf_ray: bool = Field(False, json_schema_extra={"cli": "--conf-ray"})
    gaze_tips: bool = Field(False, json_schema_extra={"cli": "--gaze-tips"})
    tip_radius: int = Field(80, json_schema_extra={"cli": "--tip-radius"})
    gaze_cone_angle: float = 0.0          # dest 'gaze_cone' -> CLI_ALIASES['--gaze-cone']
    hit_conf_gate: float = Field(0.0, json_schema_extra={"cli": "--hit-conf-gate"})
    detect_extend: float = Field(0.0, json_schema_extra={"cli": "--detect-extend"})
    detect_extend_scope: str = Field(
        "objects", json_schema_extra={"cli": "--detect-extend-scope"})
    ja_quorum: float = 1.0                # shared dest; canonical flag on phenomena.ja_quorum
    gaze_debug: bool = Field(False, json_schema_extra={"cli": "--gaze-debug"})
    forward_gaze_threshold: float = Field(
        5.0, json_schema_extra={"cli": "--forward-gaze-threshold"})
    smooth_snap: str = Field("off", json_schema_extra={"cli": "--smooth-snap"})
    smooth_snap_alpha: float = Field(0.20, json_schema_extra={"cli": "--smooth-snap-alpha"})


class TrackerSection(BaseModel):
    """Mirrors ms.pipeline_config.TrackerConfig."""

    model_config = _FORBID

    gaze_lock: bool = Field(False, json_schema_extra={"cli": "--gaze-lock"})
    dwell_frames: int = Field(15, json_schema_extra={"cli": "--dwell-frames"})
    lock_dist: int = Field(100, json_schema_extra={"cli": "--lock-dist"})
    skip_frames: int = Field(1, json_schema_extra={"cli": "--skip-frames"})
    obj_persistence: int = Field(0, json_schema_extra={"cli": "--obj-persistence"})
    snap_release_frames: int = Field(5, json_schema_extra={"cli": "--snap-release-frames"})
    snap_engage_frames: int = Field(0, json_schema_extra={"cli": "--snap-engage-frames"})
    reid_grace_seconds: float = Field(1.0, json_schema_extra={"cli": "--reid-grace-seconds"})
    reid_max_dist: int = 200              # no CLI flag today (dead getattr fallback)


class RayFormingSection(BaseModel):
    """Mirrors ms.PostProcessing.RayForming.ray_config.RayFormingConfig.

    Many fields here are populated from the SAME argparse dest as a gaze /
    tracker / depth field (RayFormingConfig.from_namespace reads the shared
    flags directly).  Canonical ``cli`` metadata lives on the primary owner
    section; the mirroring is recorded in config_compat.PATH_MIRRORS.
    """

    model_config = _FORBID

    # Ray geometry -- mirrors of gaze.* dests
    ray_length: float = 1.0
    conf_ray: bool = False
    forward_gaze_threshold: float = 5.0

    # Gazelle blend scheduler / smoother -- rayforming-owned flags
    fixation_v_threshold: float = Field(
        0.04, json_schema_extra={"cli": "--fixation-v-threshold"})
    fixation_d_threshold: float = Field(
        0.15, json_schema_extra={"cli": "--fixation-d-threshold"})
    # --min-call-gap wins over the legacy --rf-gazelle-interval alias
    # (CLI_ALIASES); both default to None in argparse so "unset" resolves to
    # this schema default of 30 via ray_config.resolve_min_call_gap.
    min_call_gap: int = Field(30, json_schema_extra={"cli": "--min-call-gap"})
    dir_min_cutoff: float = Field(1.0, json_schema_extra={"cli": "--dir-min-cutoff"})
    dir_beta: float = Field(0.5, json_schema_extra={"cli": "--dir-beta"})
    len_min_cutoff: float = Field(1.0, json_schema_extra={"cli": "--len-min-cutoff"})
    len_beta: float = Field(0.3, json_schema_extra={"cli": "--len-beta"})
    len_hold_tau: float = Field(5.0, json_schema_extra={"cli": "--len-hold-tau"})

    # Object snap -- snap_mode reads dest 'adaptive_ray' (mirror of
    # gaze.adaptive_ray); the rest mirror gaze.* / tracker.* dests.
    snap_mode: str = "off"
    snap_dist: float = 150.0
    snap_bbox_scale: float = 0.0
    snap_w_dist: float = 1.0
    snap_w_angle: float = 0.8
    snap_w_size: float = 0.0
    snap_w_intersect: float = 0.5
    snap_w_temporal: float = 0.3
    snap_gate_angle: float = 60.0
    snap_head_blend: float = 0.3
    snap_quality_thresh: float = 0.8
    snap_release_frames: int = 5
    snap_engage_frames: int = 0
    snap_tip_dist: float = -1.0
    snap_tip_quality: float = -1.0
    smooth_snap: str = "off"
    smooth_snap_alpha: float = 0.20
    obj_snap_targets: str = "all"         # no CLI flag today (GUI-set only)

    # Depth integration -- rayforming-owned flags except depth_aware_scoring
    # (mirror of depth.depth_aware_scoring).  NOTE: snap_w_depth and
    # gaze_sample_radius read dests 'snap_w_depth' / 'gaze_sample_radius'
    # which NO current flag sets (--depth-w-depth / --depth-sample-radius
    # populate the depth section's differently-named dests), so these two
    # always sit at their defaults on CLI runs.  Preserved as-is.
    depth_ray_length: bool = Field(False, json_schema_extra={"cli": "--depth-ray-length"})
    depth_length_min: float = Field(0.5, json_schema_extra={"cli": "--depth-length-min"})
    depth_length_max: float = Field(3.0, json_schema_extra={"cli": "--depth-length-max"})
    depth_belief_boost: float = Field(0.0, json_schema_extra={"cli": "--depth-belief-boost"})
    depth_aware_scoring: bool = False
    snap_w_depth: float = 0.0
    gaze_sample_radius: int = 2

    # Hit detection -- mirrors of gaze.* dests
    gaze_tips: bool = False
    tip_radius: int = 80
    gaze_cone_angle: float = 0.0
    hit_conf_gate: float = 0.0
    detect_extend: float = 0.0
    detect_extend_scope: str = "objects"


class DepthSection(BaseModel):
    """Mirrors ms.pipeline_config.DepthConfig.

    All dests here except depth_aware_scoring are prefixed ('depth',
    'depth_backend', ...) while the field names are not, so the flags live in
    CLI_ALIASES rather than as ``cli`` metadata.
    """

    model_config = _FORBID

    enabled: bool = False                 # dest 'depth' -> CLI_ALIASES['--depth']
    backend: str = "midas_small"          # CLI_ALIASES['--depth-backend']
    input_size: int = 384                 # CLI_ALIASES['--depth-input-size']
    skip_frames: int = 1                  # CLI_ALIASES['--depth-skip-frames']
    depth_aware_scoring: bool = Field(
        False, json_schema_extra={"cli": "--depth-aware-scoring"})
    snap_w_depth: float = 0.4             # CLI_ALIASES['--depth-w-depth']
    gaze_sample_radius: int = 2           # CLI_ALIASES['--depth-sample-radius']


class PhenomenaSection(BaseModel):
    """Mirrors ms.Phenomena.phenomena_config.PhenomenaConfig.

    ``--all-phenomena`` is a transient expander (handled inside
    from_namespace, like PhenomenaConfig.from_namespace does); it has no
    schema field and is a documented CLI exclusion.
    """

    model_config = _FORBID

    joint_attention: bool = Field(False, json_schema_extra={"cli": "--joint-attention"})
    ja_window: int = Field(0, json_schema_extra={"cli": "--ja-window"})
    ja_window_thresh: float = Field(0.70, json_schema_extra={"cli": "--ja-window-thresh"})
    ja_quorum: float = Field(1.0, json_schema_extra={"cli": "--ja-quorum"})
    mutual_gaze: bool = Field(False, json_schema_extra={"cli": "--mutual-gaze"})
    social_ref: bool = Field(False, json_schema_extra={"cli": "--social-ref"})
    social_ref_window: int = Field(60, json_schema_extra={"cli": "--social-ref-window"})
    gaze_follow: bool = Field(False, json_schema_extra={"cli": "--gaze-follow"})
    gaze_follow_lag: int = Field(30, json_schema_extra={"cli": "--gaze-follow-lag"})
    gaze_aversion: bool = Field(False, json_schema_extra={"cli": "--gaze-aversion"})
    aversion_window: int = Field(60, json_schema_extra={"cli": "--aversion-window"})
    aversion_conf: float = Field(0.5, json_schema_extra={"cli": "--aversion-conf"})
    scanpath: bool = Field(False, json_schema_extra={"cli": "--scanpath"})
    scanpath_dwell: int = Field(8, json_schema_extra={"cli": "--scanpath-dwell"})
    gaze_leader: bool = Field(False, json_schema_extra={"cli": "--gaze-leader"})
    gaze_leader_tips: bool = Field(False, json_schema_extra={"cli": "--gaze-leader-tips"})
    gaze_leader_tip_lag: int = Field(15, json_schema_extra={"cli": "--gaze-leader-tip-lag"})
    attn_span: bool = Field(False, json_schema_extra={"cli": "--attn-span"})


class AuxStream(BaseModel):
    """Mirrors ms.pipeline_config.AuxStreamConfig."""

    model_config = _FORBID

    source: str
    video_type: VideoType
    stream_label: str
    participants: list[str]
    auto_detect_faces: bool = True


class OutputSection(BaseModel):
    """Mirrors ms.pipeline_config.OutputConfig."""

    model_config = _FORBID

    save: bool | str | None = Field(None, json_schema_extra={"cli": "--save"})
    # log_path/summary_path are annotated str|None on the dataclass, but at
    # runtime they also carry bool True (--summary's nargs='?' const=True and
    # YAML log_csv/summary_csv: true), so the schema accepts bool too.
    log_path: bool | str | None = None    # dest 'log' -> CLI_ALIASES['--log']
    summary_path: bool | str | None = None  # dest 'summary' -> CLI_ALIASES['--summary']
    heatmap_path: bool | str | None = None  # dest 'heatmap' -> CLI_ALIASES['--heatmap']
    charts_path: bool | str | None = None   # dest 'charts' -> CLI_ALIASES['--charts']
    pid_map: dict[int, str] | None = None   # resolved by _build_from_args (no direct flag)
    aux_streams: list[AuxStream] | None = None  # parsed from --aux-stream raw entries
    anonymize: str | None = Field(None, json_schema_extra={"cli": "--anonymize"})
    anonymize_padding: float = Field(0.3, json_schema_extra={"cli": "--anonymize-padding"})
    video_name: str | None = None         # project mode only
    conditions: str | None = None         # project mode only


class ProjectOutputSection(BaseModel):
    """Mirrors ms.pipeline_config.ProjectOutputConfig."""

    model_config = _FORBID

    directory: str | None = None


class ProjectSection(BaseModel):
    """Mirrors ms.pipeline_config.ProjectConfig (loaded from project.yaml)."""

    model_config = _FORBID

    pipeline_path: str | None = None
    conditions: dict[str, list[str]] = Field(default_factory=dict)
    participants: dict[str, dict[int, str]] = Field(default_factory=dict)
    output: ProjectOutputSection = Field(default_factory=ProjectOutputSection)


# ══════════════════════════════════════════════════════════════════════════════
# Root model
# ══════════════════════════════════════════════════════════════════════════════

class PipelineConfig(BaseModel):
    """Unified schema for the full MindSight pipeline configuration."""

    model_config = _FORBID

    detection: DetectionSection = Field(default_factory=DetectionSection)
    gaze: GazeSection = Field(default_factory=GazeSection)
    tracker: TrackerSection = Field(default_factory=TrackerSection)
    rayforming: RayFormingSection = Field(default_factory=RayFormingSection)
    depth: DepthSection = Field(default_factory=DepthSection)
    phenomena: PhenomenaSection = Field(default_factory=PhenomenaSection)
    output: OutputSection = Field(default_factory=OutputSection)
    project: ProjectSection = Field(default_factory=ProjectSection)

    def canonical_hash(self) -> str:
        """sha256 over the sorted-key JSON dump; stable across processes."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_namespace(cls, ns, class_ids: list | None = None,
                       blacklist: set | None = None) -> "PipelineConfig":
        """Build the full schema from a fully-resolved argparse namespace.

        Mirrors the existing dataclass ``from_namespace`` classmethods
        EXACTLY (same dests, same fallback defaults), so that
        ``config_compat.to_dataclasses(PipelineConfig.from_namespace(ns))``
        reproduces what ``cli._build_from_args`` builds today.

        ``class_ids`` / ``blacklist`` are the values resolved by
        ``create_yolo_detector`` -- the schema cannot resolve class names
        itself (that requires the loaded model), so they are passed through
        just as ``DetectionConfig.from_namespace(ns, class_ids, blacklist)``
        receives them.
        """
        g = lambda a, d: getattr(ns, a, d)  # noqa: E731

        detection = DetectionSection(
            conf=g("conf", 0.35),
            class_ids=class_ids,
            blacklist=blacklist if blacklist is not None else set(),
            detect_scale=g("detect_scale", 1.0),
            merge_overlaps=g("merge_overlaps", False),
            merge_overlap_strategy=g("merge_overlap_strategy", "filter"),
            merge_overlap_threshold=g("merge_overlap_threshold", 0.7),
        )
        gaze = GazeSection(
            ray_length=g("ray_length", 1.0),
            adaptive_ray=g("adaptive_ray", "off"),
            snap_dist=g("snap_dist", 150.0),
            snap_bbox_scale=g("snap_bbox_scale", 0.0),
            snap_w_dist=g("snap_w_dist", 1.0),
            snap_w_angle=g("snap_w_angle", 0.8),
            snap_w_size=g("snap_w_size", 0.0),
            snap_w_intersect=g("snap_w_intersect", 0.5),
            snap_w_temporal=g("snap_w_temporal", 0.3),
            snap_gate_angle=g("snap_gate_angle", 60.0),
            snap_head_blend=g("snap_head_blend", 0.3),
            snap_quality_thresh=g("snap_quality_thresh", 0.8),
            snap_tip_dist=g("snap_tip_dist", -1.0),
            snap_tip_quality=g("snap_tip_quality", -1.0),
            conf_ray=g("conf_ray", False),
            gaze_tips=g("gaze_tips", False),
            tip_radius=g("tip_radius", 80),
            gaze_cone_angle=g("gaze_cone", 0.0),
            hit_conf_gate=g("hit_conf_gate", 0.0),
            detect_extend=g("detect_extend", 0.0),
            detect_extend_scope=g("detect_extend_scope", "objects"),
            ja_quorum=g("ja_quorum", 1.0),
            gaze_debug=g("gaze_debug", False),
            forward_gaze_threshold=g("forward_gaze_threshold", 5.0),
            smooth_snap=g("smooth_snap", "off"),
            smooth_snap_alpha=g("smooth_snap_alpha", 0.20),
        )
        tracker = TrackerSection(
            gaze_lock=g("gaze_lock", False),
            dwell_frames=g("dwell_frames", 15),
            lock_dist=g("lock_dist", 100),
            skip_frames=g("skip_frames", 1),
            obj_persistence=g("obj_persistence", 0),
            snap_release_frames=g("snap_release_frames", 5),
            snap_engage_frames=g("snap_engage_frames", 0),
            reid_grace_seconds=g("reid_grace_seconds", 1.0),
            reid_max_dist=g("reid_max_dist", 200),
        )
        rayforming = RayFormingSection(
            ray_length=g("ray_length", 1.0),
            conf_ray=g("conf_ray", False),
            forward_gaze_threshold=g("forward_gaze_threshold", 5.0),
            fixation_v_threshold=g("fixation_v_threshold", 0.04),
            fixation_d_threshold=g("fixation_d_threshold", 0.15),
            min_call_gap=resolve_min_call_gap(ns),
            dir_min_cutoff=g("dir_min_cutoff", 1.0),
            dir_beta=g("dir_beta", 0.5),
            len_min_cutoff=g("len_min_cutoff", 1.0),
            len_beta=g("len_beta", 0.3),
            len_hold_tau=g("len_hold_tau", 5.0),
            snap_mode=g("adaptive_ray", "off"),
            snap_dist=g("snap_dist", 150.0),
            snap_bbox_scale=g("snap_bbox_scale", 0.0),
            snap_w_dist=g("snap_w_dist", 1.0),
            snap_w_angle=g("snap_w_angle", 0.8),
            snap_w_size=g("snap_w_size", 0.0),
            snap_w_intersect=g("snap_w_intersect", 0.5),
            snap_w_temporal=g("snap_w_temporal", 0.3),
            snap_gate_angle=g("snap_gate_angle", 60.0),
            snap_head_blend=g("snap_head_blend", 0.3),
            snap_quality_thresh=g("snap_quality_thresh", 0.8),
            snap_release_frames=g("snap_release_frames", 5),
            snap_engage_frames=g("snap_engage_frames", 0),
            snap_tip_dist=g("snap_tip_dist", -1.0),
            snap_tip_quality=g("snap_tip_quality", -1.0),
            smooth_snap=g("smooth_snap", "off"),
            smooth_snap_alpha=g("smooth_snap_alpha", 0.20),
            obj_snap_targets=g("obj_snap_targets", "all"),
            depth_ray_length=g("depth_ray_length", False),
            depth_length_min=g("depth_length_min", 0.5),
            depth_length_max=g("depth_length_max", 3.0),
            depth_belief_boost=g("depth_belief_boost", 0.0),
            depth_aware_scoring=g("depth_aware_scoring", False),
            snap_w_depth=g("snap_w_depth", 0.0),
            gaze_sample_radius=g("gaze_sample_radius", 2),
            gaze_tips=g("gaze_tips", False),
            tip_radius=g("tip_radius", 80),
            gaze_cone_angle=g("gaze_cone", 0.0),
            hit_conf_gate=g("hit_conf_gate", 0.0),
            detect_extend=g("detect_extend", 0.0),
            detect_extend_scope=g("detect_extend_scope", "objects"),
        )
        depth = DepthSection(
            enabled=g("depth", False),
            backend=g("depth_backend", "midas_small"),
            input_size=g("depth_input_size", 384),
            skip_frames=g("depth_skip_frames", 1),
            depth_aware_scoring=g("depth_aware_scoring", False),
            snap_w_depth=g("depth_w_depth", 0.4),
            gaze_sample_radius=g("depth_sample_radius", 2),
        )
        all_on = g("all_phenomena", False)
        phenomena = PhenomenaSection(
            joint_attention=g("joint_attention", False) or all_on,
            ja_window=g("ja_window", 0),
            ja_window_thresh=g("ja_window_thresh", 0.70),
            ja_quorum=g("ja_quorum", 1.0),
            mutual_gaze=g("mutual_gaze", False) or all_on,
            social_ref=g("social_ref", False) or all_on,
            social_ref_window=g("social_ref_window", 60),
            gaze_follow=g("gaze_follow", False) or all_on,
            gaze_follow_lag=g("gaze_follow_lag", 30),
            gaze_aversion=g("gaze_aversion", False) or all_on,
            aversion_window=g("aversion_window", 60),
            aversion_conf=g("aversion_conf", 0.5),
            scanpath=g("scanpath", False) or all_on,
            scanpath_dwell=g("scanpath_dwell", 8),
            gaze_leader=g("gaze_leader", False) or all_on,
            gaze_leader_tips=g("gaze_leader_tips", False),
            gaze_leader_tip_lag=g("gaze_leader_tip_lag", 15),
            attn_span=g("attn_span", False) or all_on,
        )
        raw_aux = g("aux_streams", None)
        aux_streams = None
        if raw_aux:
            aux_streams = [
                AuxStream(
                    source=a.source,
                    video_type=a.video_type,
                    stream_label=a.stream_label,
                    participants=list(a.participants),
                    auto_detect_faces=a.auto_detect_faces,
                )
                for a in raw_aux
            ]
        output = OutputSection(
            save=g("save", None),
            log_path=g("log", None),
            summary_path=g("summary", None),
            heatmap_path=g("heatmap", None),
            charts_path=g("charts", None),
            pid_map=g("pid_map", None),
            aux_streams=aux_streams,
            anonymize=g("anonymize", None),
            anonymize_padding=g("anonymize_padding", 0.3),
        )
        return cls(
            detection=detection, gaze=gaze, tracker=tracker,
            rayforming=rayforming, depth=depth, phenomena=phenomena,
            output=output, project=ProjectSection(),
        )
