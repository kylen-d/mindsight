"""
inference_settings/spec.py -- the Inference Settings dialog layout contract
(UP2 Batch B).

``SETTINGS_SPEC`` is a literal transcription of the user-triaged spec doc
(``up2-inference-settings-spec-DRAFT.md`` r2): seven tabs -> groups -> fields.
Every user-facing label, description, tier and group placement here comes
VERBATIM from that doc -- it is the contract.  This module holds NO widget
knowledge of its own beyond the layout: the render-time metadata for each
``dest`` (widget kind, numeric range/step, choices, default, tooltip, toggle
off-value) is resolved from the live sources -- the schema ``ui`` metadata
(via ``ui_spec``) where a dest has it, else the FlagSpec census built from the
parser -- so the field definitions are never duplicated (they cannot drift).

This module imports NO Qt; the dialog (``dialog.py``) renders it and the
census test (``tests/test_inference_settings_spec.py``) welds every dest to the
live parser.  Advanced umbrella clusters (e.g. "Lock-on scoring") expand to
their individual advanced dests, each rendered with its schema/FlagSpec label
and help -- the spec-doc umbrella name/description becomes the group caption.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# Weight/model dests rendered as a path field (line-edit + Browse).
_PATH_DESTS = frozenset({
    "model", "vp_model", "vp_file", "mgaze_model", "rf_gazelle_model",
    "gazelle_model",
})


# ══════════════════════════════════════════════════════════════════════════════
# Layout records (pure data -- transcribed from the spec doc)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SpecField:
    """One rendered control.

    ``dest`` is the argparse name; ``label``/``description`` are the spec-doc
    user-facing prose (empty description -> the resolved FlagSpec help is used).
    ``tier`` is 'B' (basic) or 'A' (advanced -- collapsed in-tab).  ``end_labels``
    supplies slider endpoint captions (Q14); ``choice_labels`` maps raw combo
    values to friendly display text (e.g. adaptive_ray extend->"reach toward
    object"); ``inverted`` marks a checkbox whose dest value = not-checked.
    """
    dest: str
    label: str
    description: str = ""
    tier: str = "B"
    end_labels: tuple[str, str] | None = None
    choice_labels: dict[str, str] | None = None
    inverted: bool = False


@dataclass(frozen=True)
class SpecGroup:
    """A settings group.  ``toggle`` (owner dest) makes it a checkable group
    whose checked state drives that dest to its schema off-value when unchecked
    (same semantics SchemaPanel implements).  ``toggle_label``/``toggle_desc``
    are the checkbox's spec-doc prose.  ``caption`` is a group note."""
    key: str
    title: str
    fields: tuple[SpecField, ...] = ()
    caption: str = ""
    toggle: str | None = None
    toggle_label: str = ""
    toggle_desc: str = ""
    toggle_choice_labels: dict[str, str] | None = None


@dataclass(frozen=True)
class SpecTab:
    key: str
    title: str
    groups: tuple[SpecGroup, ...] = ()
    caption: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS_SPEC -- seven tabs, verbatim from the spec doc r2
# ══════════════════════════════════════════════════════════════════════════════

# -- Tab 1: Models & Device ----------------------------------------------------
_TAB_MODELS = SpecTab(
    "models", "Models & Device",
    caption=("Weight paths display as family names / bare filenames resolved "
             "against the shared Weights folder.  Use the Models tab to "
             "download / verify models."),
    groups=(
        SpecGroup("models", "Models & Device", fields=(
            SpecField("device", "Compute device",
                      "Where models run: Auto picks the best available "
                      "(NVIDIA GPU > Apple GPU > CPU)."),
            SpecField("no_detector", "No object detection",
                      "Run without an object-detection model: faces, gaze "
                      "rays, and gaze-tip phenomena only. Object hits and "
                      "object lock-on are disabled -- for lightweight "
                      "attention studies. Not compatible with a visual "
                      "prompt file."),
            SpecField("model", "Object detection model",
                      "The YOLO model that finds objects and people in each "
                      "frame. Smaller = faster, larger = more accurate."),
            SpecField("vp_model", "Visual prompt model",
                      "The YOLOE model used when a visual prompt file teaches "
                      "the detector your study's custom objects."),
            SpecField("vp_file", "Visual prompt file",
                      "The .vp.json from the VP Builder describing your study's "
                      "objects. Empty = standard classes only."),
            SpecField("vp_ignore_conditions", "Use full visual prompt",
                      "Prompt every class in the visual prompt file, ignoring "
                      "its condition tags and each video's conditions. Off = "
                      "condition-tagged classes apply only to matching "
                      "videos."),
            SpecField("mgaze_model", "Gaze model (MobileGaze)",
                      "The per-face gaze direction model. Family name (e.g. "
                      "\"resnet50\") auto-selects the right build for your "
                      "device."),
            SpecField("rf_gazelle_model", "Gaze-LLE model",
                      "The scene-aware model used by Gaze-LLE Correction (Gaze "
                      "Estimation tab) to periodically correct gaze rays."),
            SpecField("mgaze_arch", "MobileGaze architecture", tier="A"),
            SpecField("mgaze_dataset", "MobileGaze dataset key", tier="A"),
            SpecField("rf_gazelle_name", "Gaze-LLE variant", tier="A"),
        ), caption="Model variants / dataset keys: MobileGaze arch + dataset "
                   "key, Gaze-LLE variant name."),
    ),
)

# -- Tab 2: Gaze Estimation ----------------------------------------------------
_TAB_GAZE = SpecTab(
    "gaze", "Gaze Estimation",
    groups=(
        SpecGroup("gaze_rays", "Gaze rays", fields=(
            SpecField("ray_length", "Ray length",
                      "How far the drawn gaze ray reaches, as a multiplier of "
                      "face size.", end_labels=("shorter", "longer")),
            SpecField("forward_gaze_threshold", "Looking-at-camera threshold (deg)",
                      "Pitch/yaw below this counts as \"looking at the "
                      "camera\", not at the scene. 0 disables."),
            SpecField("gaze_cone", "Gaze cone (deg)",
                      "Replace the thin ray with a vision cone of this angle. "
                      "0 = ray. Cones catch more objects, less precisely."),
            SpecField("conf_ray", "Confidence-scaled ray length",
                      "Shorten the ray when the gaze model is unsure.", tier="A"),
            SpecField("face_eye_origin", "Eye-midpoint ray origin",
                      "Anchor rays at the detected eye midpoint instead of "
                      "the face-box centre. Changes ray origins vs. blessed "
                      "baselines.", tier="A"),
        )),
        SpecGroup("gazelle_blend", "Gaze-LLE Blend",
                  toggle="rf_gazelle_model",
                  toggle_label="Enable Gaze-LLE Blend",
                  toggle_desc="Periodically run the scene-aware Gaze-LLE model "
                              "to correct each person's gaze ray toward what "
                              "they're actually fixating. The primary validated "
                              "mode.",
                  fields=(
            SpecField("min_call_gap", "Correction interval (frames)",
                      "Minimum frames between corrections. Lower = more "
                      "corrections, slower processing."),
            SpecField("dir_beta", "Direction responsiveness",
                      "How quickly the corrected ray direction follows fast "
                      "motion.", end_labels=("steadier", "snappier")),
            SpecField("len_beta", "Length responsiveness",
                      "Same, for corrected ray length.",
                      end_labels=("steadier", "snappier")),
            SpecField("len_hold_tau", "Length hold (s)",
                      "How long a corrected ray length persists before decaying "
                      "to baseline."),
            SpecField("fixation_v_threshold", "Fixation velocity threshold",
                      tier="A"),
            SpecField("fixation_d_threshold", "Fixation dispersion threshold",
                      tier="A"),
            SpecField("dir_min_cutoff", "Direction smoother floor (Hz)",
                      tier="A"),
            SpecField("len_min_cutoff", "Length smoother floor (Hz)", tier="A"),
            SpecField("rf_inout_gate", "In/out-of-frame gate",
                      "Use the Gaze-LLE in/out-of-frame estimate (when the "
                      "checkpoint has the head): corrections with an "
                      "in-frame score below this are rejected and blend "
                      "trust scales with the score. 0 = off.",
                      tier="A"),
            SpecField("rf_reuse_eps", "Reuse unchanged scenes",
                      "Skip a scheduled Gaze-LLE correction when the frame "
                      "is visually unchanged since the last one (mean pixel "
                      "difference below this threshold) and re-apply the "
                      "cached result instead. 0 = off.",
                      tier="A"),
            SpecField("rf_onset_samples", "Onset warmup samples",
                      "Let a newly appeared face qualify for its first "
                      "correction after this many gaze samples instead of "
                      "the default 5. 0 = default warmup.",
                      tier="A"),
            SpecField("rf_onset_gap", "Onset call gap (frames)",
                      "Let a face that never had a correction fire after "
                      "this many frames since the last call, instead of the "
                      "full minimum call gap. 0 = off.",
                      tier="A"),
            SpecField("rf_len_refresh_gap", "Length refresh gap (frames)",
                      "Every N frames, run a cheap half-precision Gaze-LLE "
                      "pass that refreshes ray length only; direction stays "
                      "with the full-precision corrections. 0 = off.",
                      tier="A"),
            SpecField("rf_len_slew", "Length slew (frames)",
                      "When a refresh re-latches ray length, transition to "
                      "the new value over this many frames instead of "
                      "snapping instantly. Suggested: half the length "
                      "refresh gap. 0 = instant snap.",
                      tier="A"),
            SpecField("rf_len_gain", "Length gain",
                      "Scale the blend ray-length target. Rays measured "
                      "systematically short on the v1.3 eval; the 1.10 "
                      "default recovers reach. 1.0 = off.",
                      tier="A"),
            SpecField("rf_endpoint_extract", "Endpoint extraction",
                      "How the correction heatmap becomes a ray endpoint: "
                      "centroid (full-map, historical) or topp (top-mass "
                      "cells only -- diffuse maps stop dragging the "
                      "endpoint toward the origin).",
                      tier="A"),
            SpecField("rf_gazelle_fp16", "Half precision (fp16)",
                      "Run Gaze-LLE in half precision on CUDA/MPS. Faster per "
                      "correction; results differ slightly from full "
                      "precision, so keep off for regression baselines.",
                      tier="A"),
            SpecField("rf_gazelle_compile", "torch.compile (experimental)",
                      "Compile the Gaze-LLE model (PyTorch 2+). First "
                      "correction pays a warmup cost; MPS support immature.",
                      tier="A"),
        ), caption="Fixation sensitivity + smoother floor cutoffs (Hz) tune "
                   "when corrections fire and how the One-Euro smoother floors "
                   "direction/length."),
        SpecGroup("object_lockon", "Object lock-on",
                  toggle="adaptive_ray",
                  toggle_label="Lock rays onto objects",
                  toggle_desc="When a ray passes near a detected object: "
                              "\"reach toward object\" extends it; \"lock onto "
                              "object\" pins the endpoint to the object.",
                  toggle_choice_labels={"extend": "reach toward object",
                                        "snap": "lock onto object"},
                  fields=(
            SpecField("snap_dist", "Lock-on distance (px)",
                      "How close a ray must pass to an object to lock on."),
            SpecField("smooth_snap", "Smooth lock-on movement",
                      "Glide the ray toward its target instead of jumping: "
                      "objects / gaze tips / all.",
                      choice_labels={"off": "off", "objects": "objects",
                                     "gaze_tips": "gaze tips", "all": "all"}),
            SpecField("smooth_snap_alpha", "Smoothing rate", "",
                      end_labels=("smoother", "faster")),
            SpecField("snap_bbox_scale", "Bbox scale", tier="A"),
            SpecField("snap_w_dist", "Distance weight", tier="A"),
            SpecField("snap_w_angle", "Angle weight", tier="A"),
            SpecField("snap_w_size", "Size weight", tier="A"),
            SpecField("snap_w_intersect", "Intersection weight", tier="A"),
            SpecField("snap_w_temporal", "Stickiness weight", tier="A"),
            SpecField("snap_gate_angle", "Gate angle (deg)", tier="A"),
            SpecField("snap_head_blend", "Head-direction blend", tier="A"),
            SpecField("snap_quality_thresh", "Quality gate", tier="A"),
            SpecField("snap_tip_dist", "Tip distance (px)", tier="A"),
            SpecField("snap_tip_quality", "Tip quality gate", tier="A"),
            SpecField("snap_release_frames", "Release frames", tier="A"),
            SpecField("snap_engage_frames", "Engage frames", tier="A"),
        ), caption="Lock-on scoring: how candidate objects are ranked and "
                   "accepted -- distance, angle, size, intersection, "
                   "stickiness, head-direction blend, quality gates, "
                   "engage/release timing."),
        SpecGroup("gaze_tips", "Gaze tips",
                  toggle="gaze_tips",
                  toggle_label="Gaze tips (virtual targets)",
                  toggle_desc="Mark each ray's endpoint with a circular target "
                              "so two people's gaze can meet in empty space -- "
                              "tip convergence counts as joint attention.",
                  fields=(
            SpecField("tip_radius", "Tip radius (px)",
                      "Size of the endpoint target."),
        )),
        SpecGroup("gaze_object_hits", "Gaze-object hits", fields=(
            SpecField("hit_conf_gate", "Hit confidence gate",
                      "Ignore gaze-object hits from faces with weaker gaze "
                      "estimates than this. 0 = off."),
            SpecField("detect_extend", "Extend hit reach (px)",
                      "Count objects up to N px past the visible ray end as "
                      "hits.", tier="A"),
            SpecField("detect_extend_scope", "Extended reach applies to",
                      "Object hits, phenomena, or both.", tier="A"),
        )),
    ),
)

# -- Tab 3: Object Detection ---------------------------------------------------
_TAB_DETECTION = SpecTab(
    "detection", "Object Detection",
    groups=(
        SpecGroup("detection", "Object Detection", fields=(
            SpecField("conf", "Detection confidence",
                      "Minimum confidence to accept a detection. Lower = more "
                      "objects, more false positives."),
            SpecField("merge_overlaps", "Merge overlapping detections",
                      "Combine duplicate boxes on the same object."),
            SpecField("merge_overlap_strategy", "Merge strategy",
                      "Keep best box / merge boxes / decide per case "
                      "(\"dynamic\")."),
            SpecField("merge_overlap_threshold", "Merge threshold",
                      "How much boxes must overlap before merging."),
            SpecField("classes", "Object classes", "Restrict detected classes.",
                      tier="A"),
            SpecField("blacklist", "Class blacklist", "Exclude detected classes.",
                      tier="A"),
            SpecField("obj_persistence", "Keep lost objects (frames)",
                      "Keep an object alive N frames after the detector loses "
                      "it.", tier="A"),
            SpecField("reid_grace_seconds", "Track re-ID grace (s)",
                      "How long a lost person track can reappear with the same "
                      "identity.", tier="A"),
            SpecField("face_reid_sim", "Face re-ID similarity",
                      "Verify track revivals with face embeddings: a lost "
                      "track is re-identified anywhere in frame when "
                      "similarity is at least this value. 0 = off.",
                      tier="A"),
        )),
        SpecGroup("gaze_boost", "Gaze-guided detection boost",
                  toggle="gaze_boost",
                  toggle_label="Gaze-guided detection boost",
                  toggle_desc="Boost detector confidence for objects near "
                              "where people look.",
                  caption="Enable + factor / radius / conf bounds / classes.",
                  fields=(
            SpecField("gaze_boost_factor", "Boost factor", tier="A"),
            SpecField("gaze_boost_radius", "Boost radius (px)", tier="A"),
            SpecField("gaze_boost_min_conf", "Min confidence", tier="A"),
            SpecField("gaze_boost_max_conf", "Max confidence", tier="A"),
            SpecField("gaze_boost_classes", "Boost classes", tier="A"),
        )),
    ),
)

# -- Tab 4: Phenomena ----------------------------------------------------------
# One checkable group per phenomenon; the "Enable all phenomena" bulk action is
# a dialog button (not a persisted flag), handled in the dialog.
_TAB_PHENOMENA = SpecTab(
    "phenomena", "Phenomena",
    caption="Enable all phenomena is a bulk toggle; each phenomenon's checkbox "
            "enables its tracker.",
    groups=(
        SpecGroup("joint_attention", "Joint Attention",
                  toggle="joint_attention", toggle_label="Joint Attention",
                  toggle_desc="Enable joint-attention tracking.",
                  caption="Tip convergence IS joint attention (per-frame union, "
                          "never double-counted) -- Gaze tips extend JA to "
                          "empty-space convergence.",
                  fields=(
            SpecField("ja_window", "Consistency window (frames, 0 = off)",
                      "Frames a joint-attention episode must hold to count. "
                      "0 = off."),
            SpecField("ja_window_thresh", "Window threshold",
                      "Fraction of the consistency window that must agree "
                      "before joint attention is counted."),
            SpecField("ja_quorum", "Participant quorum",
                      "Fraction of participants that must share a target."),
        )),
        SpecGroup("mutual_gaze", "Mutual Gaze", toggle="mutual_gaze",
                  toggle_label="Mutual Gaze",
                  toggle_desc="Enable mutual-gaze tracking."),
        SpecGroup("social_ref", "Social Referencing", toggle="social_ref",
                  toggle_label="Social Referencing",
                  toggle_desc="Enable social-referencing tracking.",
                  fields=(
            SpecField("social_ref_window", "Window (frames)",
                      "Frames over which a social-referencing look-back is "
                      "counted."),
        )),
        SpecGroup("gaze_follow", "Gaze Following", toggle="gaze_follow",
                  toggle_label="Gaze Following",
                  toggle_desc="Enable gaze-following tracking.",
                  fields=(
            SpecField("gaze_follow_lag", "Max follow lag (frames)",
                      "Longest delay still counted as one person following "
                      "another's gaze."),
        )),
        SpecGroup("gaze_leader", "Gaze Leadership", toggle="gaze_leader",
                  toggle_label="Gaze Leadership",
                  toggle_desc="Enable gaze-leadership tracking.",
                  fields=(
            SpecField("gaze_leader_tips", "Count tip convergence", ""),
            SpecField("gaze_leader_tip_lag", "Tip lookback (frames)", ""),
        )),
        SpecGroup("gaze_aversion", "Gaze Aversion", toggle="gaze_aversion",
                  toggle_label="Gaze Aversion",
                  toggle_desc="Enable gaze-aversion tracking.",
                  fields=(
            SpecField("aversion_window", "Window",
                      "Frames over which gaze aversion is measured."),
            SpecField("aversion_conf", "Confidence",
                      "Minimum gaze confidence for an aversion to count."),
        )),
        SpecGroup("scanpath", "Scanpath", toggle="scanpath",
                  toggle_label="Scanpath",
                  toggle_desc="Enable scanpath tracking.",
                  fields=(
            SpecField("scanpath_dwell", "Min dwell (frames)",
                      "Minimum frames on a target before it enters the "
                      "scanpath."),
        )),
        SpecGroup("attn_span", "Attention Span", toggle="attn_span",
                  toggle_label="Attention Span",
                  toggle_desc="Enable attention-span tracking."),
        SpecGroup("eye_movement", "Eye Movement Classification",
                  toggle="eye_movement",
                  toggle_label="Eye Movement Classification",
                  toggle_desc="Enable eye-movement classification.",
                  caption="Source / thresholds / min-fixation / window.",
                  fields=(
            SpecField("em_source", "Velocity source", tier="A"),
            SpecField("em_saccade_thresh", "Saccade threshold", tier="A"),
            SpecField("em_fixation_thresh", "Fixation threshold", tier="A"),
            SpecField("em_min_fixation", "Min fixation frames", tier="A"),
            SpecField("em_velocity_window", "Velocity window", tier="A"),
        )),
        SpecGroup("novel_salience", "Novel Salience", toggle="novel_salience",
                  toggle_label="Novel Salience",
                  toggle_desc="Enable novel-salience tracking.",
                  caption="Speed / cooldown / history / flash.",
                  fields=(
            SpecField("ns_speed_thresh", "Speed threshold", tier="A"),
            SpecField("ns_cooldown", "Cooldown (frames)", tier="A"),
            SpecField("ns_history", "History", tier="A"),
            SpecField("ns_flash", "Flash (frames)", tier="A"),
        )),
        SpecGroup("pupillometry", "Pupillometry", toggle="pupillometry",
                  toggle_label="Pupillometry",
                  toggle_desc="Enable pupillometry.",
                  caption="Mode / baseline / upscale / filters / blink / "
                          "outlier / per-eye.",
                  fields=(
            SpecField("pupil_mode", "Measurement mode", tier="A"),
            SpecField("pupil_baseline", "Baseline frames", tier="A"),
            SpecField("pupil_upscale", "Upscale", tier="A"),
            SpecField("pupil_filter", "Filter", tier="A"),
            SpecField("pupil_ema_alpha", "EMA alpha", tier="A"),
            SpecField("pupil_kalman_meas_noise", "Kalman measurement noise",
                      tier="A"),
            SpecField("pupil_kalman_process_noise", "Kalman process noise",
                      tier="A"),
            SpecField("pupil_blink_frames", "Blink frames", tier="A"),
            SpecField("pupil_ear_thresh", "Eye-aspect-ratio threshold", tier="A"),
            SpecField("pupil_ir_thresh", "IR threshold", tier="A"),
            SpecField("pupil_outlier_window", "Outlier window", tier="A"),
            SpecField("pupil_per_eye", "Per-eye", tier="A"),
        )),
    ),
)

# -- Tab 5: Output -------------------------------------------------------------
_TAB_OUTPUT = SpecTab(
    "output", "Output",
    caption="Events CSV + summary CSV are ALWAYS written -- not toggles. Paths "
            "are supplied per-run by the project / quick-run layer.",
    groups=(
        SpecGroup("output", "Output", fields=(
            SpecField("save", "Save annotated video",
                      "Record the video with overlays drawn."),
            SpecField("heatmap", "Gaze heatmaps",
                      "Per-participant heatmap images after each run."),
            SpecField("charts", "Post-run charts",
                      "Time-series charts per phenomenon appear in the Charts "
                      "tab after a run (no separate chart files are written to "
                      "disk for GUI runs)."),
            SpecField("lite_overlay", "Overlay detail",
                      "Full overlays vs minimal (no cones, markers, debug "
                      "text). Checked = minimal."),
            SpecField("overlay_theme", "Overlay theme",
                      "Styling of the drawn overlays: classic "
                      "(high-saturation) or mindsight (brand palette: "
                      "indigo label tabs, logo magenta/jade accents). "
                      "Cosmetic only."),
            SpecField("no_dashboard", "Show dashboard panels",
                      "Compose side dashboard onto processed frames (off = "
                      "fastest; the GUI Live tab works regardless).",
                      tier="A", inverted=True),
        )),
        SpecGroup("anonymize", "Anonymize faces",
                  toggle="anonymize", toggle_label="Anonymize faces",
                  toggle_desc="Blur or black-box faces in the output video.",
                  fields=(
            SpecField("anonymize_padding", "Padding", tier="A"),
        )),
    ),
)

# -- Tab 6: Performance --------------------------------------------------------
_TAB_PERFORMANCE = SpecTab(
    "performance", "Performance",
    caption="Weak hardware? load the Low Power preset -- UNVALIDATED for "
            "research conclusions.",
    groups=(
        SpecGroup("performance", "Performance", fields=(
            SpecField("skip_frames", "Detect every Nth frame",
                      "Run object detection every N frames; tracking fills "
                      "gaps. Higher = faster, less accurate."),
            SpecField("detect_scale", "Detection scale",
                      "Downscale frames before detection. 1.0 = full "
                      "resolution.", end_labels=("faster", "full res")),
            SpecField("fast", "Fast mode",
                      "Bundled speed optimizations (skip phenomena on "
                      "non-detection frames, throttle previews)."),
            SpecField("skip_phenomena", "Phenomena every Nth frame",
                      "Run phenomena trackers every N frames. 0 = every frame.",
                      tier="A"),
            SpecField("mgaze_reuse_eps", "Reuse unchanged faces",
                      "Skip the per-face gaze model when a face crop is "
                      "visually unchanged from the previous frame (mean "
                      "pixel difference below this threshold). 0 = off.",
                      tier="A"),
        )),
    ),
)

# -- Tab 7: Advanced & Experimental --------------------------------------------
# Whole unvalidated/half-wired features, each a checkable group with a warning.
_TAB_EXPERIMENTAL = SpecTab(
    "experimental", "Advanced & Experimental",
    caption="Whole unvalidated or half-wired features -- not supported "
            "controls. Handle with care.",
    groups=(
        SpecGroup("depth", "Depth estimation (EXPERIMENTAL)",
                  toggle="depth", toggle_label="Depth estimation",
                  toggle_desc="EXPERIMENTAL: estimate scene depth for "
                              "depth-aware lock-on scoring.",
                  fields=(
            SpecField("depth_backend", "Backend"),
            SpecField("depth_input_size", "Input size (px)"),
            SpecField("depth_skip_frames", "Skip frames"),
            SpecField("depth_aware_scoring", "Depth-aware lock-on scoring"),
            SpecField("depth_w_depth", "Depth weight"),
            SpecField("depth_sample_radius", "Sample radius"),
        )),
        SpecGroup("depth_ray_length", "Depth-scaled ray length (EXPERIMENTAL)",
                  toggle="depth_ray_length",
                  toggle_label="Depth-scaled ray length",
                  toggle_desc="EXPERIMENTAL: scale ray length from scene depth.",
                  fields=(
            SpecField("depth_length_min", "Min multiplier"),
            SpecField("depth_length_max", "Max multiplier"),
            SpecField("depth_belief_boost", "Belief boost"),
        )),
        SpecGroup("iris_refine", "Iris refinement (EXPERIMENTAL)",
                  toggle="iris_refine", toggle_label="Iris refinement",
                  toggle_desc="EXPERIMENTAL: wrap the gaze backend with "
                              "iris-based correction.",
                  fields=(
            SpecField("iris_refine_weight", "Weight"),
            SpecField("iris_refine_upscale", "Upscale"),
        )),
        SpecGroup("gazelle_backend",
                  "Alternative gaze backend: Gaze-LLE only (EXPERIMENTAL)",
                  toggle="gazelle_model",
                  toggle_label="Gaze-LLE-only backend",
                  toggle_desc="EXPERIMENTAL: a heatmap-only gaze backend, "
                              "distinct from the validated MobileGaze+Blend "
                              "path.",
                  fields=(
            SpecField("gazelle_name", "Variant"),
            SpecField("gazelle_inout_threshold", "In/out threshold"),
            SpecField("gazelle_device", "Device"),
            SpecField("gazelle_skip_frames", "Skip frames"),
            SpecField("gazelle_fp16", "FP16"),
            SpecField("gazelle_compile", "Compile"),
        )),
        SpecGroup("gaze_debug", "Debug overlay", fields=(
            SpecField("gaze_debug", "Debug overlay",
                      "Pitch/yaw debug text drawn on video."),
        )),
    ),
)

SETTINGS_SPEC: tuple[SpecTab, ...] = (
    _TAB_MODELS, _TAB_GAZE, _TAB_DETECTION, _TAB_PHENOMENA,
    _TAB_OUTPUT, _TAB_PERFORMANCE, _TAB_EXPERIMENTAL,
)


# ══════════════════════════════════════════════════════════════════════════════
# Render-time metadata resolution (live sources -- never duplicated here)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FieldMeta:
    """Resolved widget metadata for a dest, from the live schema/FlagSpec."""
    dest: str
    widget: str                    # 'check'|'spin'|'double'|'combo'|'line'|'path'
    default: object
    tooltip: str = ""
    choices: tuple | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int | None = None
    off_value: object = None       # for toggle owners


@lru_cache(maxsize=1)
def _parser_census() -> dict[str, dict]:
    """dest -> {kind, type, default, choices, help} from the full built parser
    (core + all plugins).  Cached; builds the parser only (no models)."""
    import argparse

    from mindsight.cli_flags import build_parser
    parser = build_parser()
    out: dict[str, dict] = {}
    for a in parser._actions:
        if not a.dest or a.dest in ("help", "_explicit_cli"):
            continue
        if isinstance(a, argparse._StoreTrueAction):
            kind = "store_true"
        elif isinstance(a, argparse._StoreFalseAction):
            kind = "store_false"
        else:
            kind = "store"
        out[a.dest] = {
            "kind": kind,
            "type": a.type,
            "default": a.default,
            "choices": tuple(a.choices) if a.choices else None,
            "help": a.help or "",
        }
    return out


@lru_cache(maxsize=1)
def _schema_meta() -> dict[str, dict]:
    """dest -> schema ``ui`` metadata (ranges/steps/decimals/off_value)."""
    from mindsight.GUI.ui_spec import _schema_index
    return {d: rec["ui"] for d, rec in _schema_index().items()}


def parser_census_dests() -> set[str]:
    """Every dest the live parser exposes (for the census test)."""
    return set(_parser_census())


def _widget_for(dest: str, rec: dict) -> str:
    if dest in _PATH_DESTS:
        return "path"
    if rec["kind"] in ("store_true", "store_false"):
        return "check"
    if rec["choices"]:
        return "combo"
    t = rec["type"]
    if t is int:
        return "spin"
    if t is float:
        return "double"
    return "line"


def field_meta(dest: str) -> FieldMeta:
    """Resolve the render-time metadata for *dest* from the live sources.

    Widget kind + default + choices + help come from the parser census; numeric
    range/step/decimals + toggle off-value come from the schema ``ui`` metadata
    when the dest has it (else left None -- the widget falls back to a broad
    range, and typed values are never clamped anyway)."""
    rec = _parser_census().get(dest)
    if rec is None:
        raise KeyError(f"inference_settings.spec: dest {dest!r} not in the "
                       f"live parser census")
    ui = _schema_meta().get(dest, {})
    return FieldMeta(
        dest=dest,
        widget=_widget_for(dest, rec),
        default=ui.get("default", rec["default"]) if "default" in ui
        else rec["default"],
        tooltip=rec["help"],
        choices=rec["choices"],
        minimum=ui.get("min"),
        maximum=ui.get("max"),
        step=ui.get("step"),
        decimals=ui.get("decimals"),
        off_value=ui.get("off_value"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Traversal helpers
# ══════════════════════════════════════════════════════════════════════════════

def iter_fields(spec: tuple[SpecTab, ...] = SETTINGS_SPEC):
    """Yield every SpecField across every tab/group."""
    for tab in spec:
        for group in tab.groups:
            yield from group.fields


def all_dests(spec: tuple[SpecTab, ...] = SETTINGS_SPEC) -> set[str]:
    """Every dest the dialog writes -- field dests plus toggle owners."""
    dests: set[str] = set()
    for tab in spec:
        for group in tab.groups:
            if group.toggle is not None:
                dests.add(group.toggle)
            for f in group.fields:
                dests.add(f.dest)
    return dests


def tab_field_dests(tab: SpecTab) -> list[str]:
    """Ordered field + toggle-owner dests for one tab (for count checks)."""
    out: list[str] = []
    for group in tab.groups:
        if group.toggle is not None:
            out.append(group.toggle)
        out.extend(f.dest for f in group.fields)
    return out


# Dests the spec doc explicitly DROPS from the dialog (must never appear here).
DROPPED_DESTS: frozenset[str] = frozenset({
    "gaze_lock", "dwell_frames", "lock_dist",          # Fixation lock-on (Q4)
    "profile",                                          # per-stage timing (Q11)
    "aux_streams_raw", "aux_auto_detect",               # aux streams (Q12)
    "source", "log", "summary", "participant_ids",      # run/CLI mechanics
    "participant_csv", "pipeline", "project",
    "no_resume", "preflight", "all_phenomena",          # bulk action, not a flag
})
