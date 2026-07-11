"""
mindsight.cli_flags -- Schema-generated argparse frontend.

The MindSight CLI parser is built from an ordered ``FlagSpec`` table
(``CORE_FLAGS``) rather than hand-written ``add_argument`` calls.  The table was
generated once from the live legacy parser (scripts/capture_cli_parser_spec.py)
and frozen here -- it carries presentation (help/metavar/nargs/const/choices/
group/order); the VALUES (default + scalar type) come from the pydantic schema
in ``mindsight.config`` at build time, so the schema stays the single source of truth
for defaults.  Two flags (``--min-call-gap`` / ``--rf-gazelle-interval``) carry
an explicit ``default=None`` because ``resolve_min_call_gap`` needs None to
detect "unset"; every other schema-backed flag's default is the schema default.

Plugins keep contributing flags via ``add_arguments(parser)`` (paper contract):
``build_parser`` calls the gaze / object-detection / phenomena registries after
the core table, in the same order the legacy ``_args`` did.  MGaze is a core
backend (not a registry plugin) since SP1.6, so ``build_parser`` calls its
``add_arguments`` explicitly, right after the gaze registry loop, to preserve
the legacy flag order.

``parse_cli`` reproduces the SUPPRESS double-parse that records exactly the
flags the user typed (``ns._explicit_cli``) -- the YAML-precedence mechanism.
"""

import argparse
from dataclasses import dataclass

from Plugins import (
    gaze_registry as _gaze_registry,
)
from Plugins import (
    object_detection_registry as _od_registry,
)
from Plugins import (
    phenomena_registry as _phenomena_registry,
)

# Sentinel: pull this flag's default from its schema field at build time.
_FROM_SCHEMA = object()


@dataclass(frozen=True)
class FlagSpec:
    """One CLI flag.  ``schema_path`` ("section.field") means the default and
    scalar type come from that pydantic field; None means the flag lives
    outside the schema (model wiring / run-loop / meta) and fully specifies its
    own ``default``/``type`` here.  ``group`` None = the root options group."""
    flag: str
    dest: str
    schema_path: str | None
    kind: str = "store"                 # "store" | "store_true" | "append"
    default: object = _FROM_SCHEMA
    type: str | None = None             # "int" | "float" | "str" | None
    nargs: str | None = None
    const: object = None
    choices: tuple | None = None
    metavar: str | None = None
    help: str | None = None
    group: str | None = None


CORE_FLAGS: tuple[FlagSpec, ...] = (
    FlagSpec(flag='--source', dest='source', schema_path=None, default='0', help='Video input source, defaults to webcam'),
    FlagSpec(flag='--save', dest='save', schema_path='output.save', nargs='?', const=True, metavar='PATH', help='Save annotated video. Omit a value to use Outputs/Video/[stem]_Video_Output.mp4, or supply a custom path.'),
    FlagSpec(flag='--log', dest='log', schema_path='output.log_path'),
    FlagSpec(flag='--summary', dest='summary', schema_path='output.summary_path', nargs='?', const=True, metavar='PATH', help='Save post-run summary CSV. Omit a value to use Outputs/CSV Files/[stem]_Summary_Output.csv, or supply a custom path.'),
    FlagSpec(flag='--heatmap', dest='heatmap', schema_path='output.heatmap_path', nargs='?', const=True, metavar='PATH', help='Save per-participant scene gaze heatmaps. Omit a value to use Outputs/heatmaps/[stem]_Heatmap_Output (one PNG per participant), or supply a custom directory/prefix path.'),
    FlagSpec(flag='--charts', dest='charts', schema_path='output.charts_path', nargs='?', const=True, metavar='PATH', help='Generate post-run time-series charts for each phenomena tracker. Omit a value to use Outputs/Charts/[stem]_Charts.png, or supply a custom path.'),
    FlagSpec(flag='--pipeline', dest='pipeline', schema_path=None, default=None, metavar='YAML', help='Load pipeline configuration from a YAML file. CLI flags override YAML values.'),
    FlagSpec(flag='--project', dest='project', schema_path=None, default=None, metavar='DIR', help='Run in project mode: process all staged videos (DIR/Inputs/Runs/ run folders, or the legacy flat DIR/Inputs/Videos/) using DIR/Pipeline/pipeline.yaml as config.'),
    FlagSpec(flag='--no-resume', dest='no_resume', schema_path=None, default=False, kind='store_true', help='Project mode: reprocess every video, ignoring the resume ledger. Does not archive prior outputs. Default resumes -- skipping videos whose ledger entry is done with an unchanged config.'),
    FlagSpec(flag='--preflight', dest='preflight', schema_path=None, default=False, kind='store_true', help='Print the project readiness report (requires --project) and exit. Checks pipeline config, weights, VP file, runs, participants/conditions, device, disk, and plugin load errors. Exit 0 if no failures, else 1.'),
    FlagSpec(flag='--participant-ids', dest='participant_ids', schema_path=None, default=None, metavar='IDS', help='Comma-separated participant labels for single-video mode. Positional: first label maps to track 0, second to track 1, etc. E.g. --participant-ids S70,S71,S72'),
    FlagSpec(flag='--participant-csv', dest='participant_csv', schema_path=None, default=None, metavar='CSV', help='Path to a participant_ids.csv mapping video filenames to custom participant labels (see docs for format).'),
    FlagSpec(flag='--aux-stream', dest='aux_streams_raw', schema_path=None, default=None, kind='append', metavar='SOURCE:VIDEO_TYPE:LABEL:PIDS', help='Auxiliary video stream. Format: SOURCE:VIDEO_TYPE:LABEL:PID1,PID2 where SOURCE is the file path, VIDEO_TYPE is one of eye_only/face_closeup/wide_closeup/custom, LABEL is a user-defined stream label, and PIDS is a comma-separated list of participant labels. Repeatable for multiple streams.'),
    FlagSpec(flag='--aux-auto-detect', dest='aux_auto_detect', schema_path=None, default=True, kind='store_true', help='Enable automatic face detection on wide/face auxiliary streams (default: enabled).'),
    FlagSpec(flag='--device', dest='device', schema_path=None, default='auto', help="Compute device for all backends: auto, cpu, cuda, or mps.  'auto' selects CUDA > MPS > CPU  (default: auto)."),
    FlagSpec(flag='--anonymize', dest='anonymize', schema_path='output.anonymize', choices=('blur', 'black'), help="Anonymize faces in the output video: 'blur' applies heavy Gaussian blur, 'black' fills with a solid rectangle."),
    FlagSpec(flag='--anonymize-padding', dest='anonymize_padding', schema_path='output.anonymize_padding', metavar='FRAC', help='Fraction of face bbox size added as padding for anonymization (default: 0.3).'),
    FlagSpec(flag='--fast', dest='fast', schema_path=None, default=False, kind='store_true', help='Enable bundled performance optimizations: skip phenomena on non-detection frames, throttle dashboard bridge, reduce GUI poll rate.', group='Performance'),
    FlagSpec(flag='--skip-phenomena', dest='skip_phenomena', schema_path=None, default=0, type='int', metavar='N', help='Run phenomena trackers only every N frames (0 = every frame). Independent of --skip-frames.  (default: 0)', group='Performance'),
    FlagSpec(flag='--lite-overlay', dest='lite_overlay', schema_path=None, default=False, kind='store_true', help='Minimal overlay: disable cone rendering, convergence markers, dwell arcs, and debug text.  Keeps gaze arrows, boxes, badges.', group='Performance'),
    FlagSpec(flag='--no-dashboard', dest='no_dashboard', schema_path=None, default=False, kind='store_true', help='Skip dashboard composition for maximum throughput. Displays the raw annotated frame only.', group='Performance'),
    FlagSpec(flag='--profile', dest='profile', schema_path=None, default=False, kind='store_true', help='Print per-stage timing breakdown every 100 frames.', group='Performance'),
    FlagSpec(flag='--depth', dest='depth', schema_path='depth.enabled', kind='store_true', help='Enable monocular depth estimation.', group='Depth Estimation'),
    FlagSpec(flag='--no-depth', dest='no_depth', schema_path=None, default=False, kind='store_true', help='Explicitly disable depth estimation.', group='Depth Estimation'),
    FlagSpec(flag='--depth-backend', dest='depth_backend', schema_path='depth.backend', help='Depth model backend (default: midas_small).', group='Depth Estimation'),
    FlagSpec(flag='--depth-input-size', dest='depth_input_size', schema_path='depth.input_size', metavar='PX', help='Depth model input resolution (default: 384).', group='Depth Estimation'),
    FlagSpec(flag='--depth-skip-frames', dest='depth_skip_frames', schema_path='depth.skip_frames', metavar='N', help='Run depth every N detection cycles (default: 1).', group='Depth Estimation'),
    FlagSpec(flag='--depth-aware-scoring', dest='depth_aware_scoring', schema_path='depth.depth_aware_scoring', kind='store_true', help='Enable depth-weighted snap scoring.', group='Depth Estimation'),
    FlagSpec(flag='--depth-w-depth', dest='depth_w_depth', schema_path='depth.snap_w_depth', metavar='W', help='Depth match weight in snap scoring (default: 0.4).', group='Depth Estimation'),
    FlagSpec(flag='--depth-sample-radius', dest='depth_sample_radius', schema_path='depth.gaze_sample_radius', metavar='PX', help='Half-size of patch for depth sampling (default: 2).', group='Depth Estimation'),
    FlagSpec(flag='--model', dest='model', schema_path=None, default='yolov8n.pt', help='Object Detection Model, defaults to yolov8n.pt'),
    FlagSpec(flag='--conf', dest='conf', schema_path='detection.conf', help='Object detection confidence threshold, defaults to 0.35'),
    FlagSpec(flag='--classes', dest='classes', schema_path=None, default=[], nargs='+', help='Specified YOLO Object Detection Classes Prompt'),
    FlagSpec(flag='--blacklist', dest='blacklist', schema_path=None, default=[], nargs='+', help='Specified YOLO Object Detection Classes Blacklist'),
    FlagSpec(flag='--skip-frames', dest='skip_frames', schema_path='tracker.skip_frames', help='Frames between object detection. Higher = faster but less accurate. (Defaults to 1, i.e. process every frame)'),
    FlagSpec(flag='--detect-scale', dest='detect_scale', schema_path='detection.detect_scale', help='Detection scale for Object Recognition'),
    FlagSpec(flag='--vp-file', dest='vp_file', schema_path=None, default=None, help='Path to visual prompt file for use with YOLOE object detection models'),
    FlagSpec(flag='--vp-model', dest='vp_model', schema_path=None, default='yoloe-26l-seg.pt', help='YOLOE model to use alongside visual prompting for object detection'),
    FlagSpec(flag='--no-detector', dest='no_detector', schema_path=None, default=False, kind='store_true', help='Run without any object-detection model: faces, gaze rays, and gaze-tip phenomena only (no object hits or lock-on). Not compatible with --vp-file.'),
    FlagSpec(flag='--obj-persistence', dest='obj_persistence', schema_path='tracker.obj_persistence', metavar='N', help='Dead-reckon object bboxes for N frames after they disappear (default 0).'),
    FlagSpec(flag='--merge-overlaps', dest='merge_overlaps', schema_path='detection.merge_overlaps', kind='store_true', help='Merge or filter overlapping same-class detections.'),
    FlagSpec(flag='--merge-overlap-strategy', dest='merge_overlap_strategy', schema_path='detection.merge_overlap_strategy', choices=('filter', 'merge', 'dynamic'), help="Overlap strategy: 'filter' keeps highest-conf box, 'merge' creates encompassing box, 'dynamic' chooses per-cluster based on confidence and size (default: dynamic)."),
    FlagSpec(flag='--merge-overlap-threshold', dest='merge_overlap_threshold', schema_path='detection.merge_overlap_threshold', metavar='THR', help='Overlap threshold to trigger merge (default: 0.7).'),
    FlagSpec(flag='--ray-length', dest='ray_length', schema_path='gaze.ray_length', help='Gaze ray-length multiplier, default 1.0'),
    FlagSpec(flag='--conf-ray', dest='conf_ray', schema_path='gaze.conf_ray', kind='store_true', help='Dynamically adjust gaze ray-length based on gaze confidence value'),
    FlagSpec(flag='--gaze-tips', dest='gaze_tips', schema_path='gaze.gaze_tips', kind='store_true', help='Adds circular bounding-box to tip of gaze-rays, used to determine intersection between gaze-rays. Set radius with --tip-radius (default 80).'),
    FlagSpec(flag='--tip-radius', dest='tip_radius', schema_path='gaze.tip_radius', help='Pixel radius for --gaze-tips (default 80)'),
    FlagSpec(flag='--adaptive-ray', dest='adaptive_ray', schema_path='gaze.adaptive_ray', choices=('off', 'extend', 'snap'), help="Adaptive ray mode: 'off' disables, 'extend' freely extends the ray toward the nearest object, 'snap' locks the endpoint to the object centre (default: off)."),
    FlagSpec(flag='--snap-dist', dest='snap_dist', schema_path='gaze.snap_dist'),
    FlagSpec(flag='--snap-bbox-scale', dest='snap_bbox_scale', schema_path='gaze.snap_bbox_scale', help='Fraction of bbox half-diagonal added to snap radius (default 0.0)'),
    FlagSpec(flag='--snap-w-dist', dest='snap_w_dist', schema_path='gaze.snap_w_dist', help='Snap scoring weight for normalized distance penalty (default 1.0)'),
    FlagSpec(flag='--snap-w-angle', dest='snap_w_angle', schema_path='gaze.snap_w_angle', help='Snap scoring weight for angular deviation penalty (default 0.8)'),
    FlagSpec(flag='--snap-w-size', dest='snap_w_size', schema_path='gaze.snap_w_size', help='Snap scoring weight for object size reward (default 0.0)'),
    FlagSpec(flag='--snap-w-intersect', dest='snap_w_intersect', schema_path='gaze.snap_w_intersect', help='Snap scoring bonus for ray-bbox intersection (default 0.5)'),
    FlagSpec(flag='--snap-w-temporal', dest='snap_w_temporal', schema_path='gaze.snap_w_temporal', help='Snap scoring bonus for previous-frame target stickiness (default 0.3)'),
    FlagSpec(flag='--snap-gate-angle', dest='snap_gate_angle', schema_path='gaze.snap_gate_angle', metavar='DEG', help='Hard angular cutoff in degrees: objects beyond this angle from the blended gaze+head direction are never snap candidates (default 60.0).'),
    FlagSpec(flag='--snap-head-blend', dest='snap_head_blend', schema_path='gaze.snap_head_blend', metavar='F', help='Blend factor for angular scoring: 0=pure gaze direction, 1=pure head orientation (default 0.3).'),
    FlagSpec(flag='--snap-quality-thresh', dest='snap_quality_thresh', schema_path='gaze.snap_quality_thresh', metavar='F', help='Maximum score to accept a snap match. Higher values are more permissive. Set lower to reject poor matches (default 0.8).'),
    FlagSpec(flag='--snap-tip-dist', dest='snap_tip_dist', schema_path='gaze.snap_tip_dist', metavar='PX', help='Tip-snap distance threshold. -1 = use --snap-dist (default -1).'),
    FlagSpec(flag='--snap-tip-quality', dest='snap_tip_quality', schema_path='gaze.snap_tip_quality', metavar='F', help='Tip-snap quality threshold. -1 = use --snap-quality-thresh (default -1).'),
    FlagSpec(flag='--hit-conf-gate', dest='hit_conf_gate', schema_path='gaze.hit_conf_gate', metavar='F', help='Minimum per-face gaze confidence for ray-object hit detection. 0.0 = disabled (default).'),
    FlagSpec(flag='--detect-extend', dest='detect_extend', schema_path='gaze.detect_extend', metavar='PX', help='Extend gaze-object detection N pixels past the visual ray/cone endpoint. 0 = detection matches visual exactly (default: %(default)s).'),
    FlagSpec(flag='--detect-extend-scope', dest='detect_extend_scope', schema_path='gaze.detect_extend_scope', choices=('objects', 'phenomena', 'both'), help="Scope for --detect-extend: 'objects' extends only ray-object hit detection, 'phenomena' extends only phenomena tracking (mutual gaze, social ref), 'both' extends both (default: objects)."),
    FlagSpec(flag='--gaze-cone', dest='gaze_cone', schema_path='gaze.gaze_cone_angle', metavar='DEGREES', help='Replaces standard gaze vectors with vision cones of a specified angle in degrees (disabled by default).'),
    FlagSpec(flag='--gaze-lock', dest='gaze_lock', schema_path='tracker.gaze_lock', kind='store_true', help='Enable fixation lock-on (default: off).'),
    FlagSpec(flag='--dwell-frames', dest='dwell_frames', schema_path='tracker.dwell_frames'),
    FlagSpec(flag='--lock-dist', dest='lock_dist', schema_path='tracker.lock_dist'),
    FlagSpec(flag='--gaze-debug', dest='gaze_debug', schema_path='gaze.gaze_debug', kind='store_true'),
    FlagSpec(flag='--snap-release-frames', dest='snap_release_frames', schema_path='tracker.snap_release_frames', metavar='N', help='Frames of no-match before releasing the held snap target (default 5).'),
    FlagSpec(flag='--snap-engage-frames', dest='snap_engage_frames', schema_path='tracker.snap_engage_frames', metavar='N', help='Frames of consistent match required before engaging snap for the first time. 0 = instant engage (default 0).'),
    FlagSpec(flag='--reid-grace-seconds', dest='reid_grace_seconds', schema_path='tracker.reid_grace_seconds', metavar='S', help='Seconds a lost face track stays in the re-ID buffer (default 1.0).'),
    FlagSpec(flag='--forward-gaze-threshold', dest='forward_gaze_threshold', schema_path='gaze.forward_gaze_threshold', metavar='DEG', help='Pitch/yaw angles below this (degrees) are treated as looking forward at the camera. Set to 0 to disable (default 5.0).'),
    FlagSpec(flag='--smooth-snap', dest='smooth_snap', schema_path='gaze.smooth_snap', choices=('off', 'objects', 'gaze_tips', 'all'), help="Smooth snap mode: smoothly interpolate the ray toward snap targets instead of jumping instantly. 'objects' = smooth object snaps only, 'gaze_tips' = smooth gaze-tip snaps only, 'all' = both (default: off)."),
    FlagSpec(flag='--smooth-snap-alpha', dest='smooth_snap_alpha', schema_path='gaze.smooth_snap_alpha', metavar='F', help='EMA rate for smooth snap: lower = smoother/slower, higher = faster/more responsive (default: 0.20).'),
    FlagSpec(flag='--rf-gazelle-model', dest='rf_gazelle_model', schema_path=None, default=None, metavar='PATH', help='Path to a Gaze-LLE checkpoint (.pt) for Gazelle blend ray forming. Used alongside a pitch/yaw backend (MGaze, etc.) to periodically correct rays with Gaze-LLE heatmaps.', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--rf-gazelle-name', dest='rf_gazelle_name', schema_path=None, default='gazelle_dinov2_vitb14', choices=('gazelle_dinov2_vitb14', 'gazelle_dinov2_vitb14_inout', 'gazelle_dinov2_vitl14', 'gazelle_dinov2_vitl14_inout'), metavar='NAME', help='Gaze-LLE model variant for ray forming (default: gazelle_dinov2_vitb14).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--rf-gazelle-interval', dest='rf_gazelle_interval', schema_path='rayforming.min_call_gap', default=None, metavar='N', help='(Legacy) alias for --min-call-gap. If both are given, --min-call-gap wins. Kept so old scripts/YAMLs keep working (default: unset -> min_call_gap default).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--min-call-gap', dest='min_call_gap', schema_path='rayforming.min_call_gap', default=None, metavar='N', help='Minimum frames between Gaze-LLE inference calls. The scheduler also requires at least one participant to be fixating before firing (default: 30).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--fixation-v-threshold', dest='fixation_v_threshold', schema_path='rayforming.fixation_v_threshold', metavar='F', help='Smoothed pitch/yaw velocity (rad/frame) at which a face is treated as 50%% fixating. Lower = safer anchoring (default: 0.04).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--fixation-d-threshold', dest='fixation_d_threshold', schema_path='rayforming.fixation_d_threshold', metavar='F', help='Windowed pitch/yaw dispersion (rad) at which a face is treated as 50%% fixating (default: 0.15).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--dir-min-cutoff', dest='dir_min_cutoff', schema_path='rayforming.dir_min_cutoff', metavar='F', help='Direction One Euro smoother floor cutoff (Hz). Lower = smoother at rest (default: 1.0).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--dir-beta', dest='dir_beta', schema_path='rayforming.dir_beta', metavar='F', help='Direction One Euro responsiveness. Higher = snaps to fast motion (default: 0.5).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--len-min-cutoff', dest='len_min_cutoff', schema_path='rayforming.len_min_cutoff', metavar='F', help='Length One Euro smoother floor cutoff (Hz) (default: 1.0).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--len-beta', dest='len_beta', schema_path='rayforming.len_beta', metavar='F', help='Length One Euro responsiveness. Lower than direction by default so length holds steadier (default: 0.3).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--len-hold-tau', dest='len_hold_tau', schema_path='rayforming.len_hold_tau', metavar='F', help='Seconds the Gaze-LLE-derived ray length persists after an accepted inference before decaying back to the pitch/yaw baseline. Direction reverts quickly on its own; raise this to hold ray reach longer between inferences (default: 5.0).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--depth-ray-length', dest='depth_ray_length', schema_path='rayforming.depth_ray_length', kind='store_true', help='Use depth map to scale ray length based on scene geometry (default: off).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--depth-length-min', dest='depth_length_min', schema_path='rayforming.depth_length_min', metavar='F', help='Ray length multiplier at depth=0 (nearest) (default: 0.5).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--depth-length-max', dest='depth_length_max', schema_path='rayforming.depth_length_max', metavar='F', help='Ray length multiplier at depth=1 (farthest) (default: 3.0).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--depth-belief-boost', dest='depth_belief_boost', schema_path='rayforming.depth_belief_boost', metavar='F', help='How much depth agreement boosts Gaze-LLE heatmap confidence in the belief update (default: 0.0).', group='Ray Forming (Gazelle blend)'),
    FlagSpec(flag='--joint-attention', dest='joint_attention', schema_path='phenomena.joint_attention', kind='store_true', help='Enable joint-attention tracking.'),
    FlagSpec(flag='--ja-window', dest='ja_window', schema_path='phenomena.ja_window', metavar='N', help='Temporal consistency window (frames). 0 = disabled (default).'),
    FlagSpec(flag='--ja-window-thresh', dest='ja_window_thresh', schema_path='phenomena.ja_window_thresh', metavar='F'),
    FlagSpec(flag='--ja-quorum', dest='ja_quorum', schema_path='phenomena.ja_quorum', metavar='F', help='Fraction of faces required for joint attention (default 1.0).'),
    FlagSpec(flag='--mutual-gaze', dest='mutual_gaze', schema_path='phenomena.mutual_gaze', kind='store_true'),
    FlagSpec(flag='--social-ref', dest='social_ref', schema_path='phenomena.social_ref', kind='store_true'),
    FlagSpec(flag='--social-ref-window', dest='social_ref_window', schema_path='phenomena.social_ref_window', metavar='N'),
    FlagSpec(flag='--gaze-follow', dest='gaze_follow', schema_path='phenomena.gaze_follow', kind='store_true'),
    FlagSpec(flag='--gaze-follow-lag', dest='gaze_follow_lag', schema_path='phenomena.gaze_follow_lag', metavar='N'),
    FlagSpec(flag='--gaze-aversion', dest='gaze_aversion', schema_path='phenomena.gaze_aversion', kind='store_true'),
    FlagSpec(flag='--aversion-window', dest='aversion_window', schema_path='phenomena.aversion_window', metavar='N'),
    FlagSpec(flag='--aversion-conf', dest='aversion_conf', schema_path='phenomena.aversion_conf', metavar='F'),
    FlagSpec(flag='--scanpath', dest='scanpath', schema_path='phenomena.scanpath', kind='store_true'),
    FlagSpec(flag='--scanpath-dwell', dest='scanpath_dwell', schema_path='phenomena.scanpath_dwell', metavar='N'),
    FlagSpec(flag='--gaze-leader', dest='gaze_leader', schema_path='phenomena.gaze_leader', kind='store_true'),
    FlagSpec(flag='--gaze-leader-tips', dest='gaze_leader_tips', schema_path='phenomena.gaze_leader_tips', kind='store_true', help='Also detect leadership via gaze-tip convergence (requires --gaze-tips).'),
    FlagSpec(flag='--gaze-leader-tip-lag', dest='gaze_leader_tip_lag', schema_path='phenomena.gaze_leader_tip_lag', metavar='N', help='Lookback frames for tip-arrival priority (default: 15).'),
    FlagSpec(flag='--attn-span', dest='attn_span', schema_path='phenomena.attn_span', kind='store_true', help='Track per-participant per-object average attention span (mean completed-glance duration). Most salient object shown in HUD.'),
    FlagSpec(flag='--all-phenomena', dest='all_phenomena', schema_path=None, default=False, kind='store_true', help='Enable all gaze-phenomena trackers at once.'),
)


_ARGPARSE_TYPES = {"int": int, "float": float, "str": str}


def _schema_field(path):
    """Return the pydantic FieldInfo for a "section.field" schema path."""
    from mindsight.config import PipelineConfig
    section, fname = path.split(".")
    sub = PipelineConfig.model_fields[section].annotation
    return sub.model_fields[fname]


def _schema_argparse_type(annotation):
    """Map a scalar pydantic annotation to an argparse ``type`` callable."""
    if annotation is int:
        return int
    if annotation is float:
        return float
    return None            # bool -> store_true (no type); str/unions -> None


def _add_flag(target, spec: FlagSpec):
    if spec.schema_path is not None:
        field = _schema_field(spec.schema_path)
        default = field.default if spec.default is _FROM_SCHEMA else spec.default
        atype = _schema_argparse_type(field.annotation)
    else:
        default = spec.default
        atype = _ARGPARSE_TYPES.get(spec.type)

    if spec.kind == "store_true":
        target.add_argument(spec.flag, dest=spec.dest, action="store_true",
                            default=default, help=spec.help)
        return
    if spec.kind == "append":
        append_kwargs: dict = {"dest": spec.dest, "action": "append",
                               "default": default, "help": spec.help}
        if spec.metavar is not None:
            append_kwargs["metavar"] = spec.metavar
        target.add_argument(spec.flag, **append_kwargs)
        return
    kwargs: dict = {"dest": spec.dest, "default": default, "help": spec.help}
    if atype is not None:
        kwargs["type"] = atype
    if spec.nargs is not None:
        kwargs["nargs"] = spec.nargs
    if spec.const is not None:
        kwargs["const"] = spec.const
    if spec.choices is not None:
        kwargs["choices"] = list(spec.choices)
    if spec.metavar is not None:
        kwargs["metavar"] = spec.metavar
    target.add_argument(spec.flag, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Construct the MindSight CLI parser from CORE_FLAGS + plugin registries."""
    parser = argparse.ArgumentParser("MindSight -- Eye-Gaze Intersection Tracker")

    groups: dict[str, argparse._ArgumentGroup] = {}
    for spec in CORE_FLAGS:
        if spec.group is None:
            target = parser
        else:
            if spec.group not in groups:
                groups[spec.group] = parser.add_argument_group(spec.group)
            target = groups[spec.group]
        _add_flag(target, spec)

    # Plugin-contributed arguments (paper contract) -- same order as legacy _args.
    for _pname in _gaze_registry.names():
        _gaze_registry.get(_pname).add_arguments(parser)
    # Core gaze backend flags (MGaze is core since SP1.6, not a registry plugin;
    # added here, after registry gaze plugins, to preserve the legacy flag order).
    from mindsight.GazeTracking.Backends.MGaze.MGaze_Tracking import MGazePlugin
    MGazePlugin.add_arguments(parser)
    for _pname in _od_registry.names():
        _od_registry.get(_pname).add_arguments(parser)
    for _pname in _phenomena_registry.names():
        _phenomena_registry.get(_pname).add_arguments(parser)

    return parser


def parse_cli(argv=None):
    """Parse argv and attach ``_explicit_cli`` (the dests the user actually
    typed), via the SUPPRESS double-parse moved verbatim from the legacy
    ``mindsight.cli._args``.  YAML precedence in the loader depends on this set."""
    parser = build_parser()

    # argparse cannot tell a user-typed flag from a default; suppress every
    # action's default and re-parse so the resulting namespace contains ONLY
    # the dests the user actually supplied.
    _saved_defaults = [(a, a.default) for a in parser._actions]
    for _action, _ in _saved_defaults:
        _action.default = argparse.SUPPRESS
    _suppressed_ns, _ = parser.parse_known_args(argv)
    for _action, _default in _saved_defaults:
        _action.default = _default

    ns = parser.parse_args(argv)
    ns._explicit_cli = frozenset(vars(_suppressed_ns))
    return ns
