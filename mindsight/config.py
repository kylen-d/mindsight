"""
config.py -- Unified pydantic-v2 schema for every MindSight pipeline parameter.

This is the single source of truth introduced in SP1.1.  Each section model
mirrors an existing runtime dataclass field-for-field (same names, types, and
defaults) so that ``mindsight.config_compat.to_dataclasses()`` can reconstruct the
exact objects the pipeline consumes today:

    section       mirrors dataclass
    ---------     -----------------------------------------------------------
    detection     mindsight.pipeline_config.DetectionConfig
    gaze          mindsight.pipeline_config.GazeConfig
    tracker       mindsight.pipeline_config.TrackerConfig
    rayforming    mindsight.PostProcessing.RayForming.ray_config.RayFormingConfig
    depth         mindsight.pipeline_config.DepthConfig
    phenomena     mindsight.Phenomena.phenomena_config.PhenomenaConfig
    output        mindsight.pipeline_config.OutputConfig
    project       mindsight.pipeline_config.ProjectConfig

Field metadata: fields whose argparse ``dest`` matches the field name carry
``json_schema_extra={"cli": "--the-flag"}``.  Flags whose dest differs from
the schema field name live in ``mindsight.config_compat.CLI_ALIASES`` instead, and
flags with no schema home are documented in
``mindsight.config_compat.EXCLUDED_CLI_FLAGS``.

Several argparse dests feed MORE THAN ONE schema field (e.g. ``ray_length``
populates both ``gaze.ray_length`` and ``rayforming.ray_length``, exactly as
the existing ``from_namespace`` classmethods do).  The canonical ``cli``
metadata lives on the primary owner; the mirrors are listed in
``mindsight.config_compat.PATH_MIRRORS``.

Weight paths are stored exactly as given -- resolution stays in
``mindsight.weights.resolve_weight()`` / the model factories.

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
from mindsight.pipeline_config import VideoType
from mindsight.PostProcessing.RayForming.ray_config import resolve_min_call_gap

_FORBID = ConfigDict(extra="forbid")


# ══════════════════════════════════════════════════════════════════════════════
# Section models
# ══════════════════════════════════════════════════════════════════════════════

class DetectionSection(BaseModel):
    """Mirrors mindsight.pipeline_config.DetectionConfig."""

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
    # Runtime truth is 'dynamic': both the DetectionConfig dataclass default
    # (pipeline_config.py:171) and the argparse default are 'dynamic', so every
    # real CLI/GUI run resolves 'dynamic'.  The lone 'filter' left in the tree
    # is the DEAD getattr() fallback in DetectionConfig.from_namespace
    # (pipeline_config.py:183), which never fires because the parser always
    # supplies the value.  SP1.1 mistakenly pinned this schema default to that
    # dead fallback; corrected here so PipelineConfig() / config_compat.load_yaml
    # match runtime.
    merge_overlap_strategy: str = Field(
        "dynamic", json_schema_extra={"cli": "--merge-overlap-strategy"})
    merge_overlap_threshold: float = Field(
        0.7, json_schema_extra={"cli": "--merge-overlap-threshold"})

    @field_serializer("blacklist", when_used="json")
    def _sorted_blacklist(self, v: set) -> list:
        # Sets iterate in a hash-randomized order; serialize sorted so
        # canonical_hash() is deterministic across processes.
        return sorted(v, key=str)


class GazeSection(BaseModel):
    """Mirrors mindsight.pipeline_config.GazeConfig."""

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
    # v1.1 3.8 flip: eye-midpoint origins by default (eval-validated).
    face_eye_origin: bool = Field(True, json_schema_extra={"cli": "--face-eye-origin"})


class TrackerSection(BaseModel):
    """Mirrors mindsight.pipeline_config.TrackerConfig."""

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
    mgaze_reuse_eps: float = Field(0.0, json_schema_extra={"cli": "--mgaze-reuse-eps"})
    face_reid_sim: float = Field(0.0, json_schema_extra={"cli": "--face-reid-sim"})


class RayFormingSection(BaseModel):
    """Mirrors mindsight.PostProcessing.RayForming.ray_config.RayFormingConfig.

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
    rf_inout_gate: float = Field(0.0, json_schema_extra={"cli": "--rf-inout-gate"})
    rf_reuse_eps: float = Field(0.0, json_schema_extra={"cli": "--rf-reuse-eps"})
    # v1.1 3.8 flips: eval-validated onset defaults (corrections from frame ~3).
    rf_onset_samples: int = Field(3, json_schema_extra={"cli": "--rf-onset-samples"})
    rf_onset_gap: int = Field(5, json_schema_extra={"cli": "--rf-onset-gap"})
    # v1.1 W3Y flip: eval-validated default (70.3px mean / 66% hit rate vs
    # 71.3/64% off, ~+0.6ms/frame). 0 disables.
    rf_len_refresh_gap: int = Field(10, json_schema_extra={"cli": "--rf-len-refresh-gap"})
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
    """Mirrors mindsight.pipeline_config.DepthConfig.

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
    """Mirrors mindsight.Phenomena.phenomena_config.PhenomenaConfig.

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
    """Mirrors mindsight.pipeline_config.AuxStreamConfig."""

    model_config = _FORBID

    source: str
    video_type: VideoType
    stream_label: str
    participants: list[str]
    auto_detect_faces: bool = True


class OutputSection(BaseModel):
    """Mirrors mindsight.pipeline_config.OutputConfig."""

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
    """Mirrors mindsight.pipeline_config.ProjectOutputConfig."""

    model_config = _FORBID

    directory: str | None = None


class ProjectSection(BaseModel):
    """Mirrors mindsight.pipeline_config.ProjectConfig (loaded from project.yaml)."""

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
            merge_overlap_strategy=g("merge_overlap_strategy", "dynamic"),
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
            face_eye_origin=g("face_eye_origin", True),
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
            mgaze_reuse_eps=g("mgaze_reuse_eps", 0.0),
            face_reid_sim=g("face_reid_sim", 0.0),
        )
        rayforming = RayFormingSection(
            ray_length=g("ray_length", 1.0),
            conf_ray=g("conf_ray", False),
            forward_gaze_threshold=g("forward_gaze_threshold", 5.0),
            fixation_v_threshold=g("fixation_v_threshold", 0.04),
            fixation_d_threshold=g("fixation_d_threshold", 0.15),
            min_call_gap=resolve_min_call_gap(ns),
            rf_inout_gate=g("rf_inout_gate", 0.0),
            rf_reuse_eps=g("rf_reuse_eps", 0.0),
            rf_onset_samples=g("rf_onset_samples", 3),
            rf_onset_gap=g("rf_onset_gap", 5),
            rf_len_refresh_gap=g("rf_len_refresh_gap", 10),
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


# ══════════════════════════════════════════════════════════════════════════════
# UI metadata (SP3.1 Batch F, D6)
# ══════════════════════════════════════════════════════════════════════════════
#
# Every schema field carries a ``"ui"`` entry in its ``json_schema_extra``:
# either a dict describing how the field renders on the generated settings
# surface, or ``None`` = deliberately hidden from that surface (mirrors, output
# paths, project section, no-flag internals, and knobs that no current hand
# widget exposes -- keeping the generated surface census-neutral with the live
# GUI).  ``json_schema_extra`` is inert to validation and NOT part of
# ``canonical_hash`` (which hashes ``model_dump`` VALUES), so attaching this
# metadata never moves a config hash -- pinned by
# ``tests/test_config_schema.py::test_ui_metadata_does_not_move_canonical_hash``.
#
# ui dict keys (only those a field needs):
#   group        -- widget-section group key (matches the ui_spec group tree;
#                   NOT the schema section -- e.g. detect_scale lives in the
#                   "performance" group though it is a ``detection`` field).
#   label        -- form-row label (harvested from the live hand widgets).
#   advanced     -- True = deep-tuning tier (hidden behind Show-advanced).
#   min/max/step -- numeric spin ranges (harvested from the hand widgets;
#                   they encode real tuning knowledge).
#   decimals     -- double-spin precision.
#   choices      -- combo values (lowercase schema values, not display labels).
#   toggle_group -- names the checkable group this field OWNS (owner-only).
#   off_value    -- what the owner writes when its group is unchecked (T10).
#   default      -- widget-init override where the hand widget's initial value
#                   differs from the schema default (e.g. ja_window shows 30 in
#                   the GUI though the schema/CLI default is 0; adaptive/smooth
#                   combos show their first ON choice while the group is off).
#
# Tooltips are NOT stored here -- ui_spec pulls them from the FlagSpec help
# table at build time (single source, D6(b)).

_UI: dict[str, dict | None] = {
    # -- Detection (rendered by the hand DetectionSection; kept in the tier
    #    census as basic, but not part of the generated Gaze-Tuning surface). --
    "detection.conf": {"group": "detection", "label": "Conf",
                       "min": 0.05, "max": 0.95, "step": 0.05, "decimals": 2},
    "detection.class_ids": None,
    "detection.blacklist": None,
    "detection.detect_scale": {"group": "performance", "label": "Detect scale",
                               "min": 0.25, "max": 1.0, "step": 0.05,
                               "decimals": 2},
    "detection.merge_overlaps": {"group": "detection", "label": "Merge Overlaps"},
    "detection.merge_overlap_strategy": {
        "group": "detection", "label": "Strategy",
        "choices": ["dynamic", "filter", "merge"]},
    "detection.merge_overlap_threshold": {
        "group": "detection", "label": "Threshold",
        "min": 0.10, "max": 1.00, "step": 0.05, "decimals": 2},

    # -- Gaze: ray geometry -------------------------------------------------
    "gaze.ray_length": {"group": "ray_geometry", "label": "Ray length",
                        "min": 0.2, "max": 5.0, "step": 0.1, "decimals": 1},
    "gaze.conf_ray": {"group": "ray_geometry",
                      "label": "Scale ray length by confidence"},
    "gaze.gaze_cone_angle": {"group": "ray_geometry", "label": "Gaze cone",
                             "min": 0.0, "max": 45.0, "step": 1.0,
                             "decimals": 1},
    "gaze.forward_gaze_threshold": {
        "group": "ray_geometry", "label": "Forward threshold (°)",
        "min": 0.0, "max": 30.0, "step": 0.5, "decimals": 1},
    "gaze.gaze_tips": {"group": "gaze_tips", "label": "Gaze tips (virtual objects)",
                       "toggle_group": "gaze_tips", "off_value": False},
    "gaze.tip_radius": {"group": "gaze_tips", "label": "Tip radius",
                        "min": 20, "max": 300},

    # -- Gaze: adaptive snap ------------------------------------------------
    "gaze.adaptive_ray": {"group": "adaptive_snap", "label": "Mode",
                          "choices": ["extend", "snap"],
                          "toggle_group": "adaptive_snap", "off_value": "off",
                          "default": "extend"},
    "gaze.snap_dist": {"group": "adaptive_snap", "label": "Snap dist (px)",
                       "min": 20, "max": 500, "step": 1, "decimals": 0},
    "gaze.snap_bbox_scale": {"group": "adaptive_snap", "label": "Bbox scale",
                             "advanced": True, "min": 0.0, "max": 2.0,
                             "step": 0.1, "decimals": 2},
    "gaze.snap_w_dist": {"group": "adaptive_snap", "label": "W distance",
                         "advanced": True, "min": 0.0, "max": 3.0,
                         "step": 0.1, "decimals": 2},
    "gaze.snap_w_angle": {"group": "adaptive_snap", "label": "W angle",
                          "advanced": True, "min": 0.0, "max": 3.0,
                          "step": 0.1, "decimals": 2},
    "gaze.snap_w_size": {"group": "adaptive_snap", "label": "W size",
                         "advanced": True, "min": 0.0, "max": 3.0,
                         "step": 0.1, "decimals": 2},
    "gaze.snap_w_intersect": {"group": "adaptive_snap", "label": "W intersect",
                              "advanced": True, "min": 0.0, "max": 3.0,
                              "step": 0.1, "decimals": 2},
    "gaze.snap_w_temporal": {"group": "adaptive_snap", "label": "W temporal",
                             "advanced": True, "min": 0.0, "max": 3.0,
                             "step": 0.1, "decimals": 2},
    "gaze.snap_gate_angle": {"group": "adaptive_snap", "label": "Gate angle (°)",
                             "advanced": True, "min": 10.0, "max": 180.0,
                             "step": 5.0, "decimals": 1},
    "gaze.snap_head_blend": {"group": "adaptive_snap", "label": "Head blend",
                             "advanced": True, "min": 0.0, "max": 1.0,
                             "step": 0.05, "decimals": 2},
    "gaze.snap_quality_thresh": {
        "group": "adaptive_snap", "label": "Quality threshold",
        "advanced": True, "min": 0.1, "max": 3.0, "step": 0.1, "decimals": 2},
    "gaze.snap_tip_dist": {"group": "adaptive_snap", "label": "Tip dist (px)",
                           "advanced": True, "min": -1.0, "max": 500.0,
                           "step": 10.0, "decimals": 1},
    "gaze.snap_tip_quality": {"group": "adaptive_snap", "label": "Tip quality",
                              "advanced": True, "min": -1.0, "max": 3.0,
                              "step": 0.1, "decimals": 2},

    # -- Gaze: ray-forming smoothing ---------------------------------------
    "gaze.smooth_snap": {"group": "smoothing", "label": "Smooth targets",
                         "choices": ["objects", "gaze_tips", "all"],
                         "toggle_group": "smoothing", "off_value": "off",
                         "default": "all"},
    "gaze.smooth_snap_alpha": {"group": "smoothing", "label": "Smooth alpha",
                               "min": 0.01, "max": 1.0, "step": 0.05,
                               "decimals": 2},

    # -- Gaze: hit detection ------------------------------------------------
    "gaze.hit_conf_gate": {"group": "hit_detection", "label": "Hit conf gate",
                           "min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2},
    "gaze.detect_extend": {"group": "hit_detection",
                           "label": "Extend detection (px)",
                           "min": 0, "max": 20000, "step": 50, "decimals": 0},
    "gaze.detect_extend_scope": {"group": "hit_detection", "label": "Extend scope",
                                 "choices": ["objects", "phenomena", "both"]},

    # -- Gaze: misc ---------------------------------------------------------
    "gaze.ja_quorum": None,          # mirror of phenomena.ja_quorum (canonical)
    "gaze.gaze_debug": {"group": "performance",
                        "label": "Show pitch/yaw debug overlay"},
    "gaze.face_eye_origin": {"group": "ray_geometry",
                             "label": "Eye-midpoint ray origin",
                             "advanced": True},

    # -- Tracker ------------------------------------------------------------
    "tracker.gaze_lock": {"group": "fixation", "label": "Fixation Lock-On",
                          "toggle_group": "fixation", "off_value": False},
    "tracker.dwell_frames": {"group": "fixation", "label": "Dwell frames",
                             "min": 1, "max": 120},
    "tracker.lock_dist": {"group": "fixation", "label": "Lock dist (px)",
                          "min": 20, "max": 400},
    "tracker.skip_frames": {"group": "performance", "label": "Skip frames",
                            "min": 1, "max": 10},
    "tracker.mgaze_reuse_eps": {"group": "performance",
                                "label": "Face reuse eps",
                                "min": 0.0, "max": 20.0, "advanced": True},
    "tracker.obj_persistence": {"group": "performance", "label": "Obj persistence",
                                "min": 0, "max": 60},
    "tracker.snap_release_frames": {"group": "adaptive_snap",
                                    "label": "Release frames", "advanced": True,
                                    "min": 1, "max": 30},
    "tracker.snap_engage_frames": {"group": "adaptive_snap",
                                   "label": "Engage frames", "advanced": True,
                                   "min": 0, "max": 30},
    "tracker.reid_grace_seconds": {"group": "performance", "label": "ReID grace (s)",
                                   "advanced": True, "min": 0.0, "max": 10.0,
                                   "step": 0.5, "decimals": 1},
    "tracker.face_reid_sim": {"group": "performance",
                              "label": "ReID embed similarity",
                              "advanced": True, "min": 0.0, "max": 1.0,
                              "step": 0.05, "decimals": 2},
    "tracker.reid_max_dist": None,   # no CLI flag / no widget (dead fallback)

    # -- Ray forming (Gaze-LLE blend scheduler/smoother) -------------------
    "rayforming.ray_length": None,           # mirror of gaze.ray_length
    "rayforming.conf_ray": None,             # mirror of gaze.conf_ray
    "rayforming.forward_gaze_threshold": None,  # mirror
    "rayforming.fixation_v_threshold": {
        "group": "gazelle_blend", "label": "Fixation v-threshold (rad/frame)",
        "advanced": True, "min": 0.001, "max": 0.5, "step": 0.005,
        "decimals": 3},
    "rayforming.fixation_d_threshold": {
        "group": "gazelle_blend", "label": "Fixation d-threshold (rad)",
        "advanced": True, "min": 0.01, "max": 1.5, "step": 0.01, "decimals": 2},
    "rayforming.min_call_gap": {"group": "gazelle_blend",
                                "label": "Min call gap (frames)",
                                "min": 1, "max": 120},
    "rayforming.rf_inout_gate": {"group": "gazelle_blend",
                                 "label": "In/out gate",
                                 "advanced": True, "min": 0.0, "max": 1.0,
                                 "step": 0.05, "decimals": 2},
    "rayforming.rf_reuse_eps": {"group": "gazelle_blend",
                                "label": "Scene reuse eps",
                                "advanced": True, "min": 0.0, "max": 20.0,
                                "step": 0.1, "decimals": 1},
    "rayforming.rf_onset_samples": {"group": "gazelle_blend",
                                    "label": "Onset samples",
                                    "advanced": True, "min": 0, "max": 5},
    "rayforming.rf_onset_gap": {"group": "gazelle_blend",
                                "label": "Onset call gap (frames)",
                                "advanced": True, "min": 0, "max": 120},
    "rayforming.rf_len_refresh_gap": {"group": "gazelle_blend",
                                      "label": "Length refresh gap (frames)",
                                      "advanced": True, "min": 0, "max": 300},
    "rayforming.dir_min_cutoff": {"group": "gazelle_blend",
                                  "label": "Direction min-cutoff (Hz)",
                                  "advanced": True, "min": 0.1, "max": 20.0,
                                  "step": 0.1, "decimals": 2},
    "rayforming.dir_beta": {"group": "gazelle_blend",
                            "label": "Direction responsiveness",
                            "min": 0.0, "max": 5.0, "step": 0.1, "decimals": 2},
    "rayforming.len_min_cutoff": {"group": "gazelle_blend",
                                  "label": "Length min-cutoff (Hz)",
                                  "advanced": True, "min": 0.1, "max": 20.0,
                                  "step": 0.1, "decimals": 2},
    "rayforming.len_beta": {"group": "gazelle_blend",
                            "label": "Length responsiveness",
                            "min": 0.0, "max": 5.0, "step": 0.1, "decimals": 2},
    "rayforming.len_hold_tau": {"group": "gazelle_blend", "label": "Length hold (s)",
                                "min": 0.1, "max": 60.0, "step": 0.5,
                                "decimals": 1},
    "rayforming.snap_mode": None,            # mirror of gaze.adaptive_ray
    "rayforming.snap_dist": None,            # mirror
    "rayforming.snap_bbox_scale": None,      # mirror
    "rayforming.snap_w_dist": None,          # mirror
    "rayforming.snap_w_angle": None,         # mirror
    "rayforming.snap_w_size": None,          # mirror
    "rayforming.snap_w_intersect": None,     # mirror
    "rayforming.snap_w_temporal": None,      # mirror
    "rayforming.snap_gate_angle": None,      # mirror
    "rayforming.snap_head_blend": None,      # mirror
    "rayforming.snap_quality_thresh": None,  # mirror
    "rayforming.snap_release_frames": None,  # mirror
    "rayforming.snap_engage_frames": None,   # mirror
    "rayforming.snap_tip_dist": None,        # mirror
    "rayforming.snap_tip_quality": None,     # mirror
    "rayforming.smooth_snap": None,          # mirror
    "rayforming.smooth_snap_alpha": None,    # mirror
    "rayforming.obj_snap_targets": None,     # no CLI flag / no widget
    # depth-ray-length knobs: real flags but NO current widget -- hidden so the
    # generated surface stays census-neutral with the live GUI (candidates for
    # a future explicit addition; see Batch F retrospective).
    "rayforming.depth_ray_length": None,
    "rayforming.depth_length_min": None,
    "rayforming.depth_length_max": None,
    "rayforming.depth_belief_boost": None,
    "rayforming.depth_aware_scoring": None,  # mirror of depth.depth_aware_scoring
    "rayforming.snap_w_depth": None,         # no dest sets this (reads a dead dest)
    "rayforming.gaze_sample_radius": None,   # no dest sets this
    "rayforming.gaze_tips": None,            # mirror of gaze.gaze_tips
    "rayforming.tip_radius": None,           # mirror
    "rayforming.gaze_cone_angle": None,      # mirror
    "rayforming.hit_conf_gate": None,        # mirror
    "rayforming.detect_extend": None,        # mirror
    "rayforming.detect_extend_scope": None,  # mirror

    # -- Depth --------------------------------------------------------------
    "depth.enabled": {"group": "depth", "label": "Depth Estimation",
                      "toggle_group": "depth", "off_value": False},
    "depth.backend": {"group": "depth", "label": "Backend",
                      "choices": ["midas_small"]},
    "depth.input_size": {"group": "depth", "label": "Input size (px)",
                         "advanced": True, "min": 256, "max": 512, "step": 64},
    "depth.skip_frames": {"group": "depth", "label": "Skip frames",
                          "advanced": True, "min": 1, "max": 10},
    "depth.depth_aware_scoring": {"group": "depth",
                                  "label": "Enable depth-weighted snap scoring",
                                  "advanced": True},
    "depth.snap_w_depth": {"group": "depth", "label": "Depth weight",
                           "advanced": True, "min": 0.0, "max": 2.0,
                           "step": 0.05, "decimals": 2},
    "depth.gaze_sample_radius": None,        # no widget exposes --depth-sample-radius

    # -- Phenomena ----------------------------------------------------------
    "phenomena.joint_attention": {"group": "ja", "label": "Joint Attention",
                                  "toggle_group": "ja", "off_value": False},
    "phenomena.ja_window": {"group": "ja", "label": "Temporal window",
                            "min": 0, "max": 300, "default": 30},
    "phenomena.ja_window_thresh": {"group": "ja", "label": "Window threshold",
                                   "min": 0.0, "max": 1.0, "step": 0.05,
                                   "decimals": 2},
    "phenomena.ja_quorum": {"group": "ja", "label": "Quorum",
                            "min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2},
    "phenomena.mutual_gaze": {"group": "phenomena", "label": "Mutual Gaze"},
    "phenomena.social_ref": {"group": "phenomena", "label": "Social Referencing"},
    "phenomena.social_ref_window": {"group": "phenomena", "label": "window",
                                    "min": 1, "max": 300},
    "phenomena.gaze_follow": {"group": "phenomena", "label": "Gaze Following"},
    "phenomena.gaze_follow_lag": {"group": "phenomena", "label": "lag",
                                  "min": 1, "max": 120},
    "phenomena.gaze_aversion": {"group": "aversion", "label": "Gaze Aversion",
                                "toggle_group": "aversion", "off_value": False},
    "phenomena.aversion_window": {"group": "aversion", "label": "Window",
                                  "min": 1, "max": 300},
    "phenomena.aversion_conf": {"group": "aversion", "label": "Confidence",
                                "min": 0.0, "max": 1.0, "step": 0.05,
                                "decimals": 2},
    "phenomena.scanpath": {"group": "phenomena", "label": "Scanpath"},
    "phenomena.scanpath_dwell": {"group": "phenomena", "label": "dwell",
                                 "min": 1, "max": 60},
    "phenomena.gaze_leader": {"group": "phenomena", "label": "Gaze Leadership"},
    "phenomena.gaze_leader_tips": {"group": "phenomena", "label": "+ Tips"},
    "phenomena.gaze_leader_tip_lag": {"group": "phenomena", "label": "lag",
                                      "min": 1, "max": 120},
    "phenomena.attn_span": {"group": "phenomena", "label": "Attention Span"},

    # -- Output / project: hand-written or non-tunable -> hidden -----------
    "output.save": None,
    "output.log_path": None,
    "output.summary_path": None,
    "output.heatmap_path": None,
    "output.charts_path": None,
    "output.pid_map": None,
    "output.aux_streams": None,
    "output.anonymize": None,
    "output.anonymize_padding": None,
    "output.video_name": None,
    "output.conditions": None,
    "project.pipeline_path": None,
    "project.conditions": None,
    "project.participants": None,
    "project.output": None,
}


def _attach_ui_metadata() -> None:
    """Attach the ``"ui"`` entry from ``_UI`` onto every schema field's
    ``json_schema_extra`` (D6).  Runs once at import; asserts full coverage so a
    new field can never ship untagged.  Inert to validation and canonical_hash.
    """
    tagged: set[str] = set()
    for section, section_field in PipelineConfig.model_fields.items():
        submodel = section_field.annotation
        for fname, field in submodel.model_fields.items():
            path = f"{section}.{fname}"
            if path not in _UI:
                raise RuntimeError(
                    f"schema field {path} has no ui metadata in _UI "
                    f"(every field must be tagged, dict or None)")
            extra = field.json_schema_extra
            if extra is None:
                field.json_schema_extra = {"ui": _UI[path]}
            else:
                extra["ui"] = _UI[path]
            tagged.add(path)
    stale = set(_UI) - tagged
    if stale:
        raise RuntimeError(f"_UI references unknown schema fields: {sorted(stale)}")


_attach_ui_metadata()
