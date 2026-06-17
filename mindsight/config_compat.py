"""
config_compat.py -- Compatibility layer between today's CLI/YAML spellings
and the unified ``mindsight.config.PipelineConfig`` schema (SP1.1).

Three frozen tables map every existing knob onto schema paths:

* ``CLI_ALIASES``       -- CLI flags whose argparse dest differs from the
                           schema field name.
* ``EXCLUDED_CLI_FLAGS``-- core (non-plugin) flags deliberately OUTSIDE the
                           schema, each with the reason.
* ``YAML_ALIASES``      -- old pipeline-YAML keys -> canonical schema paths
                           (plus ``YAML_UNMAPPED`` for keys with no home).

``PATH_MIRRORS`` records the fan-out where several schema fields read the
same argparse dest today (e.g. ``ray_length`` feeds both gaze and
rayforming); ``load_yaml`` applies it so a YAML value lands everywhere the
live loader + ``from_namespace`` route would put it.

``load_yaml(path)`` applies a pipeline YAML over SCHEMA DEFAULTS -- i.e. the
value a YAML author wrote always wins over a default.  Note this is the
*intended* semantics; the legacy ``load_pipeline`` (bottom of this module) merges
into a live namespace and its ``_is_default`` heuristic silently skips any
key whose namespace value is truthy (see tests/test_config_equivalence.py
for the pinned side-by-side).  Unknown keys are silently ignored, exactly
like the legacy loader.

``to_dataclasses(cfg)`` reconstructs the 8 existing config dataclasses.
Live model objects (YOLO, GazelleProvider, depth backend, plugins) are NOT
config -- ``factory.build_from_namespace`` keeps building those.

The legacy loader itself is untouched; it is retired in SP1.3.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

from mindsight.config import AuxStream, PipelineConfig
from mindsight.Phenomena.phenomena_config import PhenomenaConfig
from mindsight.pipeline_config import (
    AuxStreamConfig,
    DepthConfig,
    DetectionConfig,
    GazeConfig,
    OutputConfig,
    ProjectConfig,
    ProjectOutputConfig,
    TrackerConfig,
    VideoType,
)
from mindsight.PostProcessing.RayForming.ray_config import RayFormingConfig

# ══════════════════════════════════════════════════════════════════════════════
# CLI flag tables
# ══════════════════════════════════════════════════════════════════════════════

# Flags whose argparse dest differs from the schema field name.
# flag -> canonical "section.field" path.
CLI_ALIASES: dict[str, str] = {
    "--gaze-cone":           "gaze.gaze_cone_angle",       # dest gaze_cone
    "--log":                 "output.log_path",            # dest log
    "--summary":             "output.summary_path",        # dest summary
    "--heatmap":             "output.heatmap_path",        # dest heatmap
    "--charts":              "output.charts_path",         # dest charts
    "--depth":               "depth.enabled",              # dest depth
    "--depth-backend":       "depth.backend",              # dest depth_backend
    "--depth-input-size":    "depth.input_size",           # dest depth_input_size
    "--depth-skip-frames":   "depth.skip_frames",          # dest depth_skip_frames
    "--depth-w-depth":       "depth.snap_w_depth",         # dest depth_w_depth
    "--depth-sample-radius": "depth.gaze_sample_radius",   # dest depth_sample_radius
    # Legacy alias: --min-call-gap wins when both are given
    # (ray_config.resolve_min_call_gap encodes the precedence).
    "--rf-gazelle-interval": "rayforming.min_call_gap",    # dest rf_gazelle_interval
}

# Core flags deliberately outside the schema.  flag -> reason.
# (Plugin-registered flags -- MGaze/Gazelle backends, iris refine,
# gaze boost, eye movement, novel salience, pupillometry -- are excluded
# wholesale; the completeness test discovers them via the plugin registries
# so new plugins never need edits here.)
EXCLUDED_CLI_FLAGS: dict[str, str] = {
    # Model/backend wiring -- constructed by factories in _build_from_args;
    # providers are model wiring, not config (SP1.1 plan section 4).
    "--model":             "model wiring: YOLO weights path, create_yolo_detector",
    "--vp-file":           "model wiring: YOLOE visual-prompt file",
    "--vp-model":          "model wiring: YOLOE VP model path",
    "--device":            "model wiring: compute device for all backends",
    "--rf-gazelle-model":  "model wiring: GazelleProvider checkpoint",
    "--rf-gazelle-name":   "model wiring: GazelleProvider variant",
    # Raw name lists resolved against the loaded model into
    # detection.class_ids / detection.blacklist by create_yolo_detector.
    "--classes":           "raw class names; resolved to detection.class_ids at build",
    "--blacklist":         "raw class names; resolved to detection.blacklist at build",
    # Run-loop orchestration -- passed straight to cli.run(), not config
    # dataclasses.  Candidates for a future schema section (SP1.2+).
    "--source":            "run-loop input; passed directly to run()",
    "--fast":              "run-loop performance toggle; passed directly to run()",
    "--skip-phenomena":    "run-loop performance knob; passed directly to run()",
    "--lite-overlay":      "run-loop performance toggle; passed directly to run()",
    "--no-dashboard":      "run-loop performance toggle; passed directly to run()",
    "--profile":           "run-loop diagnostics toggle; passed directly to run()",
    # Config-loading meta flags.
    "--pipeline":          "meta: which pipeline YAML to load",
    "--project":           "meta: project-mode directory",
    "--no-resume":         "run-loop orchestration: project resume toggle",
    # Transient modifiers consumed before/inside from_namespace.
    "--no-depth":          "transient: flips ns.depth in _build_from_args",
    "--all-phenomena":     "transient: expanded inside from_namespace toggles",
    # Raw participant/aux inputs resolved into output.pid_map /
    # output.aux_streams by _build_from_args.
    "--participant-ids":   "raw input; resolved into output.pid_map",
    "--participant-csv":   "raw input; resolved into output.pid_map",
    "--aux-stream":        "raw SOURCE:TYPE:LABEL:PIDS; parsed into output.aux_streams",
    "--aux-auto-detect":   "raw input; folded into output.aux_streams entries",
}

# ══════════════════════════════════════════════════════════════════════════════
# Path fan-out (shared argparse dests)
# ══════════════════════════════════════════════════════════════════════════════

# canonical path -> additional schema paths populated from the same dest.
PATH_MIRRORS: dict[str, tuple[str, ...]] = {
    "gaze.ray_length":            ("rayforming.ray_length",),
    "gaze.conf_ray":              ("rayforming.conf_ray",),
    "gaze.forward_gaze_threshold": ("rayforming.forward_gaze_threshold",),
    "gaze.adaptive_ray":          ("rayforming.snap_mode",),
    "gaze.snap_dist":             ("rayforming.snap_dist",),
    "gaze.snap_bbox_scale":       ("rayforming.snap_bbox_scale",),
    "gaze.snap_w_dist":           ("rayforming.snap_w_dist",),
    "gaze.snap_w_angle":          ("rayforming.snap_w_angle",),
    "gaze.snap_w_size":           ("rayforming.snap_w_size",),
    "gaze.snap_w_intersect":      ("rayforming.snap_w_intersect",),
    "gaze.snap_w_temporal":       ("rayforming.snap_w_temporal",),
    "gaze.snap_gate_angle":       ("rayforming.snap_gate_angle",),
    "gaze.snap_head_blend":       ("rayforming.snap_head_blend",),
    "gaze.snap_quality_thresh":   ("rayforming.snap_quality_thresh",),
    "gaze.snap_tip_dist":         ("rayforming.snap_tip_dist",),
    "gaze.snap_tip_quality":      ("rayforming.snap_tip_quality",),
    "gaze.smooth_snap":           ("rayforming.smooth_snap",),
    "gaze.smooth_snap_alpha":     ("rayforming.smooth_snap_alpha",),
    "gaze.gaze_tips":             ("rayforming.gaze_tips",),
    "gaze.tip_radius":            ("rayforming.tip_radius",),
    "gaze.gaze_cone_angle":       ("rayforming.gaze_cone_angle",),
    "gaze.hit_conf_gate":         ("rayforming.hit_conf_gate",),
    "gaze.detect_extend":         ("rayforming.detect_extend",),
    "gaze.detect_extend_scope":   ("rayforming.detect_extend_scope",),
    "tracker.snap_release_frames": ("rayforming.snap_release_frames",),
    "tracker.snap_engage_frames": ("rayforming.snap_engage_frames",),
    "depth.depth_aware_scoring":  ("rayforming.depth_aware_scoring",),
    "phenomena.ja_quorum":        ("gaze.ja_quorum",),
    # NOTE deliberately absent: depth.snap_w_depth / depth.gaze_sample_radius
    # do NOT mirror into rayforming.snap_w_depth / rayforming.gaze_sample_radius
    # -- those rayforming fields read dests no current flag or YAML key sets.
}

# ══════════════════════════════════════════════════════════════════════════════
# YAML key tables (spellings inherited from the old loader's _YAML_MAP)
# ══════════════════════════════════════════════════════════════════════════════

# old YAML key -> canonical schema path.
YAML_ALIASES: dict[str, str] = {
    # Detection
    "detection.conf":            "detection.conf",
    "detection.detect_scale":    "detection.detect_scale",
    "detection.skip_frames":     "tracker.skip_frames",
    "detection.obj_persistence": "tracker.obj_persistence",
    # Gaze
    "gaze.ray_length":           "gaze.ray_length",
    "gaze.adaptive_ray":         "gaze.adaptive_ray",   # legacy bool handled below
    "gaze.snap_dist":            "gaze.snap_dist",
    "gaze.snap_bbox_scale":      "gaze.snap_bbox_scale",
    "gaze.snap_w_dist":          "gaze.snap_w_dist",
    "gaze.snap_w_angle":         "gaze.snap_w_angle",
    "gaze.snap_w_size":          "gaze.snap_w_size",
    "gaze.snap_w_intersect":     "gaze.snap_w_intersect",
    "gaze.snap_w_temporal":      "gaze.snap_w_temporal",
    "gaze.snap_gate_angle":      "gaze.snap_gate_angle",
    "gaze.snap_head_blend":      "gaze.snap_head_blend",
    "gaze.snap_quality_thresh":  "gaze.snap_quality_thresh",
    "gaze.snap_tip_dist":        "gaze.snap_tip_dist",
    "gaze.snap_tip_quality":     "gaze.snap_tip_quality",
    "gaze.conf_ray":             "gaze.conf_ray",
    "gaze.gaze_tips":            "gaze.gaze_tips",
    "gaze.tip_radius":           "gaze.tip_radius",
    "gaze.gaze_cone":            "gaze.gaze_cone_angle",
    "gaze.gaze_lock":            "tracker.gaze_lock",
    "gaze.dwell_frames":         "tracker.dwell_frames",
    "gaze.lock_dist":            "tracker.lock_dist",
    "gaze.gaze_debug":           "gaze.gaze_debug",
    "gaze.snap_release_frames":  "tracker.snap_release_frames",
    "gaze.snap_engage_frames":   "tracker.snap_engage_frames",
    "gaze.reid_grace_seconds":   "tracker.reid_grace_seconds",
    "gaze.hit_conf_gate":        "gaze.hit_conf_gate",
    "gaze.detect_extend":        "gaze.detect_extend",
    "gaze.detect_extend_scope":  "gaze.detect_extend_scope",
    # Output
    "output.save_video":         "output.save",
    "output.log_csv":            "output.log_path",
    "output.summary_csv":        "output.summary_path",
    "output.heatmaps":           "output.heatmap_path",
    "output.anonymize":          "output.anonymize",
    "output.anonymize_padding":  "output.anonymize_padding",
    # Joint attention (dict-style phenomena keys)
    "phenomena.ja_window":       "phenomena.ja_window",
    "phenomena.ja_window_thresh": "phenomena.ja_window_thresh",
    "phenomena.ja_quorum":       "phenomena.ja_quorum",
    # Depth estimation
    "depth.enabled":             "depth.enabled",
    "depth.backend":             "depth.backend",
    "depth.input_size":          "depth.input_size",
    "depth.skip_frames":         "depth.skip_frames",
    "depth.depth_aware_scoring": "depth.depth_aware_scoring",
    "depth.snap_w_depth":        "depth.snap_w_depth",
    "depth.gaze_sample_radius":  "depth.gaze_sample_radius",
}

# Legacy YAML keys recognized by the old loader's _YAML_MAP that have no
# schema home.  key -> reason.  (They keep working through the legacy
# loader until SP1.3; a future `migrate` command will warn on them.)
YAML_UNMAPPED: dict[str, str] = {
    "source":                    "run-loop input; passed directly to run()",
    "detection.model":           "model wiring: YOLO weights path",
    "detection.classes":         "raw class names; resolved at model build",
    "detection.blacklist":       "raw class names; resolved at model build",
    "detection.vp_file":         "model wiring: YOLOE visual-prompt file",
    "detection.vp_model":        "model wiring: YOLOE VP model path",
    "participants.csv":          "raw input; resolved into output.pid_map",
    "participants.ids":          "raw input; resolved into output.pid_map",
    "performance.fast":          "run-loop performance toggle",
    "performance.skip_phenomena": "run-loop performance knob",
    "performance.lite_overlay":  "run-loop performance toggle",
    "performance.no_dashboard":  "run-loop performance toggle",
    "performance.profile":       "run-loop diagnostics toggle",
    # Shim companion key, consumed by the adaptive_ray bool->enum shim only.
    "gaze.adaptive_snap":        "legacy shim input for boolean gaze.adaptive_ray",
}

# Phenomena list-format tables (mirror the old loader's _PHENOMENA_TOGGLES /
# _PHENOMENA_PARAMS, but targeting schema paths).
PHENOMENA_TOGGLE_PATHS: dict[str, str] = {
    "mutual_gaze":        "phenomena.mutual_gaze",
    "social_referencing": "phenomena.social_ref",
    "gaze_following":     "phenomena.gaze_follow",
    "gaze_aversion":      "phenomena.gaze_aversion",
    "scanpath":           "phenomena.scanpath",
    "gaze_leadership":    "phenomena.gaze_leader",
    "attention_span":     "phenomena.attn_span",
    "joint_attention":    "phenomena.joint_attention",
}

PHENOMENA_PARAM_PATHS: dict[str, str] = {
    "ja_window":          "phenomena.ja_window",
    "ja_quorum":          "phenomena.ja_quorum",
    "ja_window_thresh":   "phenomena.ja_window_thresh",
    "window":             "phenomena.social_ref_window",
    "lag":                "phenomena.gaze_follow_lag",
    "aversion_window":    "phenomena.aversion_window",
    "aversion_conf":      "phenomena.aversion_conf",
    "dwell":              "phenomena.scanpath_dwell",
}


# ══════════════════════════════════════════════════════════════════════════════
# YAML loading
# ══════════════════════════════════════════════════════════════════════════════

def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dicts into dot-separated keys (lists stay values)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _set_path(tree: dict, path: str, value) -> None:
    """Set a dotted ``section.field`` path inside a nested dict."""
    section, field = path.split(".", 1)
    tree.setdefault(section, {})[field] = value


def _apply(tree: dict, path: str, value) -> None:
    """Set a canonical path plus all of its PATH_MIRRORS."""
    _set_path(tree, path, value)
    for mirror in PATH_MIRRORS.get(path, ()):
        _set_path(tree, mirror, value)


def _parse_aux_streams(aux_list: list) -> list[AuxStream]:
    """Parse an aux_streams YAML section (same rules as the legacy loader)."""
    parsed: list[AuxStream] = []
    for item in aux_list:
        if not isinstance(item, dict):
            continue
        source = item.get("source", "")
        vtype_str = item.get("video_type", "custom")
        stream_label = item.get("stream_label", "")
        participants = item.get("participants", [])
        auto_detect = item.get("auto_detect_faces", True)
        if not (source and stream_label and participants):
            continue
        try:
            vtype = VideoType(vtype_str)
        except ValueError:
            print(f"Warning: unknown video_type '{vtype_str}', using 'custom'")
            vtype = VideoType.CUSTOM
        if isinstance(participants, str):
            participants = [participants]
        parsed.append(AuxStream(
            source=source,
            video_type=vtype,
            stream_label=stream_label,
            participants=list(participants),
            auto_detect_faces=bool(auto_detect),
        ))
    return parsed


def load_yaml(path: str | Path) -> PipelineConfig:
    """Load a pipeline YAML into a PipelineConfig over schema defaults.

    Applies ``YAML_ALIASES`` (with ``PATH_MIRRORS`` fan-out), the phenomena
    list format, the legacy boolean adaptive_ray/adaptive_snap shim, the
    ``rf_gazelle_interval`` -> ``min_call_gap`` precedence from the plugins
    section, and the aux_streams section.  Unknown keys are silently
    ignored, matching the legacy loader.  Plugin flags in the ``plugins``
    section stay namespace-passed (out of schema scope) and are ignored
    here, except the two ray-forming interval keys above.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    flat = _flatten(cfg)
    tree: dict = {}

    # 1. Aliased scalar keys.
    for yaml_key, schema_path in YAML_ALIASES.items():
        if yaml_key not in flat:
            continue
        value = flat[yaml_key]
        # Legacy shim: old YAMLs used adaptive_ray: true/false plus an
        # adaptive_snap boolean; new spelling is the "off"/"extend"/"snap"
        # enum (handled the same way by the legacy load_pipeline below).
        if yaml_key == "gaze.adaptive_ray" and isinstance(value, bool):
            if value:
                value = "snap" if flat.get("gaze.adaptive_snap", False) else "extend"
            else:
                value = "off"
        _apply(tree, schema_path, value)

    # 2. Phenomena list format: "- mutual_gaze" or "- joint_attention: {...}".
    phenomena_list = cfg.get("phenomena", [])
    if isinstance(phenomena_list, list):
        for item in phenomena_list:
            if isinstance(item, str):
                items = {item: {}}
            elif isinstance(item, dict):
                items = item
            else:
                continue
            for name, params in items.items():
                toggle_path = PHENOMENA_TOGGLE_PATHS.get(name)
                if toggle_path is None:
                    continue
                _apply(tree, toggle_path, True)
                for param_key, param_path in PHENOMENA_PARAM_PATHS.items():
                    if param_key in (params or {}):
                        _apply(tree, param_path, params[param_key])

    # 3. Plugins section: plugin flags are out of schema scope, but the two
    #    ray-forming interval spellings route to rayforming.min_call_gap
    #    with the same precedence resolve_min_call_gap applies (explicit
    #    min_call_gap wins over the legacy rf_gazelle_interval alias).
    plugins_cfg = cfg.get("plugins", {})
    if isinstance(plugins_cfg, dict):
        norm = {str(k).replace("-", "_"): v for k, v in plugins_cfg.items()}
        if norm.get("min_call_gap") is not None:
            _apply(tree, "rayforming.min_call_gap", int(norm["min_call_gap"]))
        elif norm.get("rf_gazelle_interval") is not None:
            _apply(tree, "rayforming.min_call_gap", int(norm["rf_gazelle_interval"]))

    # 4. Aux streams.
    aux_list = cfg.get("aux_streams", [])
    if isinstance(aux_list, list) and aux_list:
        parsed = _parse_aux_streams(aux_list)
        if parsed:
            _set_path(tree, "output.aux_streams", parsed)

    return PipelineConfig(**tree)


# ══════════════════════════════════════════════════════════════════════════════
# Schema -> existing dataclasses
# ══════════════════════════════════════════════════════════════════════════════

def to_dataclasses(cfg: PipelineConfig) -> tuple[
    GazeConfig, DetectionConfig, TrackerConfig, RayFormingConfig,
    DepthConfig, PhenomenaConfig, OutputConfig, ProjectConfig,
]:
    """Reconstruct the 8 existing config dataclasses from the schema.

    Returns ``(gaze, detection, tracker, rayforming, depth, phenomena,
    output, project)``.  Live model objects (YOLO, face detector, gaze
    engine, GazelleProvider, depth backend, plugin instances) are NOT built
    here -- they are model wiring and stay in ``cli._build_from_args``.
    """
    d, g, t, r = cfg.detection, cfg.gaze, cfg.tracker, cfg.rayforming
    dp, ph, o, pj = cfg.depth, cfg.phenomena, cfg.output, cfg.project

    gaze = GazeConfig(
        ray_length=g.ray_length,
        adaptive_ray=g.adaptive_ray,
        snap_dist=g.snap_dist,
        snap_bbox_scale=g.snap_bbox_scale,
        snap_w_dist=g.snap_w_dist,
        snap_w_angle=g.snap_w_angle,
        snap_w_size=g.snap_w_size,
        snap_w_intersect=g.snap_w_intersect,
        snap_w_temporal=g.snap_w_temporal,
        snap_gate_angle=g.snap_gate_angle,
        snap_head_blend=g.snap_head_blend,
        snap_quality_thresh=g.snap_quality_thresh,
        snap_tip_dist=g.snap_tip_dist,
        snap_tip_quality=g.snap_tip_quality,
        conf_ray=g.conf_ray,
        gaze_tips=g.gaze_tips,
        tip_radius=g.tip_radius,
        gaze_cone_angle=g.gaze_cone_angle,
        hit_conf_gate=g.hit_conf_gate,
        detect_extend=g.detect_extend,
        detect_extend_scope=g.detect_extend_scope,
        ja_quorum=g.ja_quorum,
        gaze_debug=g.gaze_debug,
        forward_gaze_threshold=g.forward_gaze_threshold,
        smooth_snap=g.smooth_snap,
        smooth_snap_alpha=g.smooth_snap_alpha,
    )
    detection = DetectionConfig(
        conf=d.conf,
        class_ids=list(d.class_ids) if d.class_ids is not None else None,
        blacklist=set(d.blacklist),
        detect_scale=d.detect_scale,
        merge_overlaps=d.merge_overlaps,
        merge_overlap_strategy=d.merge_overlap_strategy,
        merge_overlap_threshold=d.merge_overlap_threshold,
    )
    tracker = TrackerConfig(
        gaze_lock=t.gaze_lock,
        dwell_frames=t.dwell_frames,
        lock_dist=t.lock_dist,
        skip_frames=t.skip_frames,
        obj_persistence=t.obj_persistence,
        snap_release_frames=t.snap_release_frames,
        snap_engage_frames=t.snap_engage_frames,
        reid_grace_seconds=t.reid_grace_seconds,
        reid_max_dist=t.reid_max_dist,
    )
    rayforming = RayFormingConfig(
        ray_length=r.ray_length,
        conf_ray=r.conf_ray,
        forward_gaze_threshold=r.forward_gaze_threshold,
        fixation_v_threshold=r.fixation_v_threshold,
        fixation_d_threshold=r.fixation_d_threshold,
        min_call_gap=r.min_call_gap,
        dir_min_cutoff=r.dir_min_cutoff,
        dir_beta=r.dir_beta,
        len_min_cutoff=r.len_min_cutoff,
        len_beta=r.len_beta,
        len_hold_tau=r.len_hold_tau,
        snap_mode=r.snap_mode,
        snap_dist=r.snap_dist,
        snap_bbox_scale=r.snap_bbox_scale,
        snap_w_dist=r.snap_w_dist,
        snap_w_angle=r.snap_w_angle,
        snap_w_size=r.snap_w_size,
        snap_w_intersect=r.snap_w_intersect,
        snap_w_temporal=r.snap_w_temporal,
        snap_gate_angle=r.snap_gate_angle,
        snap_head_blend=r.snap_head_blend,
        snap_quality_thresh=r.snap_quality_thresh,
        snap_release_frames=r.snap_release_frames,
        snap_engage_frames=r.snap_engage_frames,
        snap_tip_dist=r.snap_tip_dist,
        snap_tip_quality=r.snap_tip_quality,
        smooth_snap=r.smooth_snap,
        smooth_snap_alpha=r.smooth_snap_alpha,
        obj_snap_targets=r.obj_snap_targets,
        depth_ray_length=r.depth_ray_length,
        depth_length_min=r.depth_length_min,
        depth_length_max=r.depth_length_max,
        depth_belief_boost=r.depth_belief_boost,
        depth_aware_scoring=r.depth_aware_scoring,
        snap_w_depth=r.snap_w_depth,
        gaze_sample_radius=r.gaze_sample_radius,
        gaze_tips=r.gaze_tips,
        tip_radius=r.tip_radius,
        gaze_cone_angle=r.gaze_cone_angle,
        hit_conf_gate=r.hit_conf_gate,
        detect_extend=r.detect_extend,
        detect_extend_scope=r.detect_extend_scope,
    )
    depth = DepthConfig(
        enabled=dp.enabled,
        backend=dp.backend,
        input_size=dp.input_size,
        skip_frames=dp.skip_frames,
        depth_aware_scoring=dp.depth_aware_scoring,
        snap_w_depth=dp.snap_w_depth,
        gaze_sample_radius=dp.gaze_sample_radius,
    )
    phenomena = PhenomenaConfig(
        joint_attention=ph.joint_attention,
        ja_window=ph.ja_window,
        ja_window_thresh=ph.ja_window_thresh,
        ja_quorum=ph.ja_quorum,
        mutual_gaze=ph.mutual_gaze,
        social_ref=ph.social_ref,
        social_ref_window=ph.social_ref_window,
        gaze_follow=ph.gaze_follow,
        gaze_follow_lag=ph.gaze_follow_lag,
        gaze_aversion=ph.gaze_aversion,
        aversion_window=ph.aversion_window,
        aversion_conf=ph.aversion_conf,
        scanpath=ph.scanpath,
        scanpath_dwell=ph.scanpath_dwell,
        gaze_leader=ph.gaze_leader,
        gaze_leader_tips=ph.gaze_leader_tips,
        gaze_leader_tip_lag=ph.gaze_leader_tip_lag,
        attn_span=ph.attn_span,
    )
    aux_streams = None
    if o.aux_streams:
        aux_streams = [
            AuxStreamConfig(
                source=a.source,
                video_type=a.video_type,
                stream_label=a.stream_label,
                participants=list(a.participants),
                auto_detect_faces=a.auto_detect_faces,
            )
            for a in o.aux_streams
        ]
    output = OutputConfig(
        save=o.save,
        log_path=o.log_path,
        summary_path=o.summary_path,
        heatmap_path=o.heatmap_path,
        charts_path=o.charts_path,
        pid_map=dict(o.pid_map) if o.pid_map is not None else None,
        aux_streams=aux_streams,
        anonymize=o.anonymize,
        anonymize_padding=o.anonymize_padding,
        video_name=o.video_name,
        conditions=o.conditions,
    )
    project = ProjectConfig(
        pipeline_path=pj.pipeline_path,
        conditions={k: list(v) for k, v in pj.conditions.items()},
        participants={k: dict(v) for k, v in pj.participants.items()},
        output=ProjectOutputConfig(directory=pj.output.directory),
    )
    return (gaze, detection, tracker, rayforming, depth, phenomena,
            output, project)


# ══════════════════════════════════════════════════════════════════════════════
# Legacy namespace loader  (moved verbatim from the deleted pipeline_loader module, SP1.3)
# ══════════════════════════════════════════════════════════════════════════════
# The GUI (empty-namespace route) and the CLI (default-namespace + _explicit_cli
# route) both merge a pipeline YAML into a live argparse namespace here.  This is
# distinct from load_yaml() above, which builds a PipelineConfig over schema
# defaults; the namespace-merge semantics (explicit-flag precedence, plugins
# passthrough, phenomena list format, adaptive_ray bool shim) are preserved as-is
# and reuse this module's _flatten.

_YAML_MAP: dict[str, str] = {
    # Top-level
    'source':               'source',

    # Detection
    'detection.model':      'model',
    'detection.conf':       'conf',
    'detection.classes':    'classes',
    'detection.blacklist':  'blacklist',
    'detection.detect_scale': 'detect_scale',
    'detection.vp_file':    'vp_file',
    'detection.vp_model':   'vp_model',
    'detection.skip_frames': 'skip_frames',
    'detection.obj_persistence': 'obj_persistence',

    # Gaze
    'gaze.ray_length':     'ray_length',
    'gaze.adaptive_ray':   'adaptive_ray',
    'gaze.snap_dist':      'snap_dist',
    'gaze.snap_bbox_scale': 'snap_bbox_scale',
    'gaze.snap_w_dist':    'snap_w_dist',
    'gaze.snap_w_angle':   'snap_w_angle',
    'gaze.snap_w_size':    'snap_w_size',
    'gaze.snap_w_intersect': 'snap_w_intersect',
    'gaze.snap_w_temporal': 'snap_w_temporal',
    'gaze.snap_gate_angle': 'snap_gate_angle',
    'gaze.snap_head_blend': 'snap_head_blend',
    'gaze.snap_quality_thresh': 'snap_quality_thresh',
    'gaze.snap_tip_dist':  'snap_tip_dist',
    'gaze.snap_tip_quality': 'snap_tip_quality',
    'gaze.conf_ray':       'conf_ray',
    'gaze.gaze_tips':      'gaze_tips',
    'gaze.tip_radius':     'tip_radius',
    'gaze.gaze_cone':      'gaze_cone',
    'gaze.gaze_lock':      'gaze_lock',
    'gaze.dwell_frames':   'dwell_frames',
    'gaze.lock_dist':      'lock_dist',
    'gaze.gaze_debug':     'gaze_debug',
    'gaze.snap_release_frames': 'snap_release_frames',
    'gaze.snap_engage_frames': 'snap_engage_frames',
    'gaze.reid_grace_seconds': 'reid_grace_seconds',
    'gaze.hit_conf_gate':      'hit_conf_gate',
    'gaze.detect_extend':      'detect_extend',
    'gaze.detect_extend_scope': 'detect_extend_scope',

    # Output
    'output.save_video':       'save',
    'output.log_csv':          'log',
    'output.summary_csv':      'summary',
    'output.heatmaps':         'heatmap',
    'output.anonymize':        'anonymize',
    'output.anonymize_padding': 'anonymize_padding',

    # Participants
    'participants.csv':    'participant_csv',
    'participants.ids':    'participant_ids',

    # Performance
    'performance.fast':             'fast',
    'performance.skip_phenomena':   'skip_phenomena',
    'performance.lite_overlay':     'lite_overlay',
    'performance.no_dashboard':     'no_dashboard',
    'performance.profile':          'profile',

    # Joint attention (top-level phenomena keys)
    'phenomena.ja_window':       'ja_window',
    'phenomena.ja_window_thresh': 'ja_window_thresh',
    'phenomena.ja_quorum':       'ja_quorum',

    # Depth estimation
    'depth.enabled':             'depth',
    'depth.backend':             'depth_backend',
    'depth.input_size':          'depth_input_size',
    'depth.skip_frames':         'depth_skip_frames',
    'depth.depth_aware_scoring': 'depth_aware_scoring',
    'depth.snap_w_depth':        'depth_w_depth',
    'depth.gaze_sample_radius':  'depth_sample_radius',
}

# Phenomena tracker toggles — boolean flags
_PHENOMENA_TOGGLES: dict[str, str] = {
    'mutual_gaze':       'mutual_gaze',
    'social_referencing': 'social_ref',
    'gaze_following':    'gaze_follow',
    'gaze_aversion':     'gaze_aversion',
    'scanpath':          'scanpath',
    'gaze_leadership':   'gaze_leader',
    'attention_span':    'attn_span',
    'joint_attention':   'joint_attention',
}

# Phenomena per-tracker params
_PHENOMENA_PARAMS: dict[str, str] = {
    'ja_window':         'ja_window',
    'ja_quorum':         'ja_quorum',
    'ja_window_thresh':  'ja_window_thresh',
    'window':            'social_ref_window',
    'lag':               'gaze_follow_lag',
    'aversion_window':   'aversion_window',
    'aversion_conf':     'aversion_conf',
    'dwell':             'scanpath_dwell',
}

def load_pipeline(path: str | Path, ns: Namespace | None = None) -> Namespace:
    """
    Load a pipeline YAML config and populate an argparse Namespace.

    Parameters
    ----------
    path : str or Path
        Path to the YAML pipeline file.
    ns : Namespace, optional
        Existing namespace to update.  CLI-set values (non-default) are
        preserved; only defaults are overwritten by YAML values.
        If None, a new Namespace is created.

    Returns
    -------
    Namespace with YAML values merged in.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if ns is None:
        ns = Namespace()

    # Precedence source of truth.  When the CLI parser attached ``_explicit_cli``
    # (the exact set of user-typed dests, see mindsight.cli._args), YAML applies to
    # every dest the user did NOT type and the _is_default heuristic is bypassed.
    # GUI-constructed / synthetic namespaces lack the attr and keep the legacy
    # _is_default path unchanged.
    explicit = getattr(ns, '_explicit_cli', None)

    # 1. Process flat and nested sections (detection, gaze, output)
    flat = _flatten(cfg)
    for yaml_key, attr in _YAML_MAP.items():
        if yaml_key in flat:
            # Only set if the CLI did not explicitly provide this dest
            if _should_set(ns, attr, explicit):
                setattr(ns, attr, flat[yaml_key])

    # 2. Process phenomena list (special format)
    phenomena_list = cfg.get('phenomena', [])
    if isinstance(phenomena_list, list):
        for item in phenomena_list:
            if isinstance(item, str):
                # Simple toggle: "- mutual_gaze"
                _set_phenomenon(ns, item, {}, explicit)
            elif isinstance(item, dict):
                # Toggle with params: "- joint_attention: {ja_window: 30}"
                for name, params in item.items():
                    _set_phenomenon(ns, name, params or {}, explicit)

    # 3. Process plugins section (pass-through to argparse attributes)
    #    Keys are mapped directly: hyphens → underscores.
    #    This supports ANY plugin flag without needing a hardcoded mapping.
    plugins_cfg = cfg.get('plugins', {})
    if isinstance(plugins_cfg, dict):
        for key, val in plugins_cfg.items():
            attr = key.replace('-', '_')
            if _should_set(ns, attr, explicit):
                setattr(ns, attr, val)

    # 4. Process aux_streams section (optional per-participant video feeds)
    aux_list = cfg.get('aux_streams', [])
    if isinstance(aux_list, list) and aux_list:
        if _should_set(ns, 'aux_streams', explicit):
            from mindsight.pipeline_config import AuxStreamConfig, VideoType
            parsed = []
            for item in aux_list:
                if isinstance(item, dict):
                    source = item.get('source', '')
                    vtype_str = item.get('video_type', 'custom')
                    stream_label = item.get('stream_label', '')
                    participants = item.get('participants', [])
                    auto_detect = item.get('auto_detect_faces', True)
                    if source and stream_label and participants:
                        try:
                            vtype = VideoType(vtype_str)
                        except ValueError:
                            print(f"Warning: unknown video_type '{vtype_str}', "
                                  f"using 'custom'")
                            vtype = VideoType.CUSTOM
                        if isinstance(participants, str):
                            participants = [participants]
                        parsed.append(AuxStreamConfig(
                            source=source,
                            video_type=vtype,
                            stream_label=stream_label,
                            participants=list(participants),
                            auto_detect_faces=bool(auto_detect),
                        ))
            if parsed:
                setattr(ns, 'aux_streams', parsed)

    # ── Backward compat: old adaptive_ray (bool) + adaptive_snap (bool) ──
    ar = getattr(ns, 'adaptive_ray', 'off')
    if isinstance(ar, bool):
        # Old YAML had adaptive_ray: true/false
        adaptive_snap = flat.get('gaze.adaptive_snap', False)
        if ar:
            setattr(ns, 'adaptive_ray', 'snap' if adaptive_snap else 'extend')
        else:
            setattr(ns, 'adaptive_ray', 'off')

    return ns


def _set_phenomenon(ns: Namespace, name: str, params: dict,
                    explicit: frozenset | None = None) -> None:
    """Enable a phenomenon tracker and set its parameters."""
    attr = _PHENOMENA_TOGGLES.get(name)
    if attr is None:
        return
    if _should_set(ns, attr, explicit):
        setattr(ns, attr, True)

    for param_key, param_attr in _PHENOMENA_PARAMS.items():
        if param_key in params:
            if _should_set(ns, param_attr, explicit):
                setattr(ns, param_attr, params[param_key])


def _should_set(ns: Namespace, attr: str, explicit: frozenset | None) -> bool:
    """Decide whether a YAML value may populate ``attr`` on the namespace.

    When ``explicit`` is provided (the CLI route -- the exact set of dests the
    user typed, from mindsight.cli._args), YAML wins for every dest NOT in that set;
    the ``_is_default`` heuristic is bypassed entirely.  When ``explicit`` is
    None (GUI / synthetic namespaces), fall back to the legacy heuristic:
    overwrite only attrs that are missing or look default-like.
    """
    if explicit is not None:
        return attr not in explicit
    return not hasattr(ns, attr) or _is_default(ns, attr)


def _is_default(ns: Namespace, attr: str) -> bool:
    """Heuristic: a value is 'default' if it matches common defaults.

    This is a best-effort check since argparse doesn't track which values
    were explicitly set by the user.  CLI flags that differ from defaults
    are considered user-set and will not be overwritten.
    """
    val = getattr(ns, attr, None)
    # None, False, 0, 0.0, empty list/string are considered default-like
    if val is None or val is False or val == 0 or val == 0.0:
        return True
    if isinstance(val, (list, str)) and len(val) == 0:
        return True
    return False
