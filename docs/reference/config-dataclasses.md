# Configuration Dataclasses

MindSight groups its many parameters into typed dataclasses defined in `mindsight/pipeline_config.py`, `mindsight/Phenomena/phenomena_config.py`, and `mindsight/PostProcessing/RayForming/ray_config.py`. Each dataclass has a `from_namespace(ns)` classmethod that constructs it from an `argparse.Namespace`. A parallel pydantic schema (`mindsight/config.py`) mirrors these dataclasses field-for-field and is the single source of truth for defaults; `config_compat.to_dataclasses()` reconstructs the dataclasses below from it.

---

## GazeConfig

Defined in `mindsight/pipeline_config.py`. All gaze-estimation and ray-intersection tuning parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ray_length` | float | `1.0` | Gaze ray length multiplier |
| `adaptive_ray` | str | `"off"` | Adaptive ray mode: `"off"`, `"extend"`, or `"snap"` |
| `snap_dist` | float | `150.0` | Maximum snap distance in pixels |
| `snap_bbox_scale` | float | `0.0` | Fraction of bbox half-diagonal added to snap radius |
| `snap_w_dist` | float | `1.0` | Snap scoring weight: normalized distance penalty |
| `snap_w_angle` | float | `0.8` | Snap scoring weight: angular deviation penalty |
| `snap_w_size` | float | `0.0` | Snap scoring weight: object size reward (off by default) |
| `snap_w_intersect` | float | `0.5` | Snap scoring bonus for ray-bbox intersection |
| `snap_w_temporal` | float | `0.3` | Snap scoring bonus for previous-frame target stickiness |
| `snap_gate_angle` | float | `60.0` | Hard angular cutoff (degrees) beyond which objects are never snap candidates |
| `snap_head_blend` | float | `0.3` | Angular scoring blend: 0 = pure gaze direction, 1 = pure head orientation |
| `snap_quality_thresh` | float | `0.8` | Maximum score to accept a snap match (higher = more permissive) |
| `snap_tip_dist` | float | `-1.0` | Tip-snap distance threshold; -1 = use `snap_dist` |
| `snap_tip_quality` | float | `-1.0` | Tip-snap quality threshold; -1 = use `snap_quality_thresh` |
| `conf_ray` | bool | `False` | Scale ray length by face-detection confidence |
| `gaze_tips` | bool | `False` | Enable gaze-tip convergence detection |
| `tip_radius` | int | `80` | Pixel radius for convergence check |
| `gaze_cone_angle` | float | `0.0` | Half-angle (degrees) of gaze cone; 0 = ray only (set by `--gaze-cone`, dest `gaze_cone`) |
| `hit_conf_gate` | float | `0.0` | Minimum face confidence required for a hit to register |
| `detect_extend` | float | `0.0` | Extra pixels past visual ray for detection (0 = visual parity) |
| `detect_extend_scope` | str | `"objects"` | What detect-extend applies to: `"objects"`, `"phenomena"`, or `"both"` |
| `ja_quorum` | float | `1.0` | Fraction of detected persons required for joint attention |
| `gaze_debug` | bool | `False` | Draw debug annotations for gaze processing |
| `forward_gaze_threshold` | float | `5.0` | Yaw/pitch threshold (degrees) below which gaze is forward-facing |
| `smooth_snap` | str | `"off"` | Smooth-snap mode: `"off"`, `"objects"`, `"gaze_tips"`, or `"all"` |
| `smooth_snap_alpha` | float | `0.20` | EMA rate for smooth snap (lower = smoother/slower) |

---

## DetectionConfig

Defined in `mindsight/pipeline_config.py`. Object-detection parameters passed through to YOLO.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conf` | float | `0.35` | Minimum detection confidence threshold |
| `class_ids` | list or None | `None` | Resolved YOLO class IDs to detect (None = all) |
| `blacklist` | set | `set()` | Set of class names to exclude from detections |
| `detect_scale` | float | `1.0` | Scale factor applied to input before detection |
| `merge_overlaps` | bool | `False` | Merge or filter overlapping same-class detections |
| `merge_overlap_strategy` | str | `"dynamic"` | Overlap strategy: `"filter"` (keep highest-conf), `"merge"` (encompassing box), or `"dynamic"` (per-cluster) |
| `merge_overlap_threshold` | float | `0.7` | Overlap fraction that triggers a merge |

Note: `from_namespace(ns, class_ids, blacklist)` takes pre-resolved class IDs and blacklist set as additional arguments.

---

## TrackerConfig

Defined in `mindsight/pipeline_config.py`. Parameters used by `run()` to construct per-run tracker instances.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gaze_lock` | bool | `False` | Enable gaze lock-on behaviour |
| `dwell_frames` | int | `15` | Frames of sustained gaze required to trigger lock-on |
| `lock_dist` | int | `100` | Maximum pixel distance for lock-on to persist |
| `skip_frames` | int | `1` | Process detection every N-th frame |
| `obj_persistence` | int | `0` | Keep detections alive for N frames after a miss |
| `snap_release_frames` | int | `5` | Frames of no-match before releasing the held snap target |
| `snap_engage_frames` | int | `0` | Frames of consistent match required before first engaging snap (0 = instant) |
| `reid_grace_seconds` | float | `1.0` | Grace period (seconds) for face re-ID after a miss |
| `reid_max_dist` | int | `200` | Maximum pixel distance for face re-identification (no CLI flag) |

---

## DepthConfig

Defined in `mindsight/pipeline_config.py`. Monocular depth-estimation parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `False` | Enable monocular depth estimation (set by `--depth`, dest `depth`) |
| `backend` | str | `"midas_small"` | Depth model backend |
| `input_size` | int | `384` | Depth model input resolution (smaller = faster) |
| `skip_frames` | int | `1` | Run depth every N detection cycles |
| `depth_aware_scoring` | bool | `False` | Opt-in depth-weighted snap scoring |
| `snap_w_depth` | float | `0.4` | Scoring weight for depth agreement (only used when `depth_aware_scoring`) |
| `gaze_sample_radius` | int | `2` | Half-size of the patch sampled for depth at the gaze point |

`from_namespace(ns)` reads the prefixed CLI dests (`depth`, `depth_backend`, `depth_input_size`, `depth_skip_frames`, `depth_aware_scoring`, `depth_w_depth`, `depth_sample_radius`).

---

## RayFormingConfig

Defined in `mindsight/PostProcessing/RayForming/ray_config.py`. The largest config object: all ray-forming, Gaze-LLE-blend, snap, fixation, and hit-detection parameters for the primary ray-forming pipeline. Built from a namespace with `from_namespace(ns)`, or from a legacy `GazeConfig` (+ optional `DepthConfig`) with `from_gaze_config(...)`. Many fields mirror `GazeConfig` / `TrackerConfig` / `DepthConfig` values (populated from the same argparse dests).

### Ray geometry

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ray_length` | float | `1.0` | Gaze ray length multiplier (mirror of `GazeConfig.ray_length`) |
| `conf_ray` | bool | `False` | Scale ray length by confidence (mirror) |
| `forward_gaze_threshold` | float | `5.0` | Forward-facing yaw/pitch threshold in degrees (mirror) |

### Gaze-LLE blend (scheduler + One Euro smoother)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fixation_v_threshold` | float | `0.04` | Smoothed pitch/yaw velocity (rad/frame) at 50% fixation likelihood |
| `fixation_d_threshold` | float | `0.15` | Windowed pitch/yaw dispersion (rad) at 50% fixation likelihood |
| `min_call_gap` | int | `30` | Minimum frames between Gaze-LLE inference calls |
| `dir_min_cutoff` | float | `1.0` | Direction One Euro floor cutoff (Hz) |
| `dir_beta` | float | `0.5` | Direction One Euro speed responsiveness |
| `len_min_cutoff` | float | `1.0` | Length One Euro floor cutoff (Hz) |
| `len_beta` | float | `0.3` | Length One Euro speed responsiveness |
| `len_hold_tau` | float | `5.0` | Seconds the Gaze-LLE-derived length persists before decaying to baseline |

### Object snap

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `snap_mode` | str | `"off"` | `"off"`, `"extend"`, or `"snap"` (mirror of `adaptive_ray`) |
| `snap_dist` | float | `150.0` | Maximum snap distance in pixels |
| `snap_bbox_scale` | float | `0.0` | Fraction of bbox half-diagonal added to snap radius |
| `snap_w_dist` | float | `1.0` | Distance penalty weight |
| `snap_w_angle` | float | `0.8` | Angular deviation penalty weight |
| `snap_w_size` | float | `0.0` | Object size reward weight |
| `snap_w_intersect` | float | `0.5` | Ray-bbox intersection bonus |
| `snap_w_temporal` | float | `0.3` | Previous-target stickiness bonus |
| `snap_gate_angle` | float | `60.0` | Hard angular cutoff (degrees) |
| `snap_head_blend` | float | `0.3` | Angular blend: 0 = gaze, 1 = head |
| `snap_quality_thresh` | float | `0.8` | Maximum score to accept a match |
| `snap_release_frames` | int | `5` | Frames of no-match before releasing a held target |
| `snap_engage_frames` | int | `0` | Frames of consistent match before first engaging |
| `snap_tip_dist` | float | `-1.0` | Tip-snap distance threshold; -1 = use `snap_dist` |
| `snap_tip_quality` | float | `-1.0` | Tip-snap quality threshold; -1 = use `snap_quality_thresh` |
| `smooth_snap` | str | `"off"` | `"off"`, `"objects"`, `"gaze_tips"`, or `"all"` |
| `smooth_snap_alpha` | float | `0.20` | EMA rate for smooth snap |
| `obj_snap_targets` | str | `"all"` | Snap-target scope: `"all"`, `"faces_only"`, or `"off"` (no CLI flag; GUI-set) |

### Depth integration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `depth_ray_length` | bool | `False` | Scale ray length from the depth map |
| `depth_length_min` | float | `0.5` | Ray-length multiplier at depth 0 (nearest) |
| `depth_length_max` | float | `3.0` | Ray-length multiplier at depth 1 (farthest) |
| `depth_belief_boost` | float | `0.0` | How much depth agreement boosts Gaze-LLE heatmap confidence |
| `depth_aware_scoring` | bool | `False` | Depth-weighted snap scoring (mirror of `DepthConfig`) |
| `snap_w_depth` | float | `0.0` | Depth match weight (reads a dest no current flag sets; stays at default on CLI runs) |
| `gaze_sample_radius` | int | `2` | Depth sampling patch half-size (reads a dest no current flag sets) |

### Hit detection

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gaze_tips` | bool | `False` | Enable gaze-tip convergence (mirror) |
| `tip_radius` | int | `80` | Pixel radius for convergence (mirror) |
| `gaze_cone_angle` | float | `0.0` | Gaze cone half-angle in degrees (mirror) |
| `hit_conf_gate` | float | `0.0` | Minimum face confidence for a hit (mirror) |
| `detect_extend` | float | `0.0` | Extra pixels past the visual ray for detection (mirror) |
| `detect_extend_scope` | str | `"objects"` | `"objects"`, `"phenomena"`, or `"both"` (mirror) |

---

## OutputConfig

Defined in `mindsight/pipeline_config.py`. Paths and flags controlling run-loop outputs.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save` | bool/str/None | `None` | Save annotated output video (path or auto-name if True) |
| `log_path` | str or None | `None` | Path for per-frame CSV log |
| `summary_path` | str or None | `None` | Path for post-run summary CSV |
| `heatmap_path` | str or None | `None` | Path for gaze heatmap image |
| `charts_path` | bool/str/None | `None` | Path for chart images |
| `pid_map` | dict[int, str] or None | `None` | Track ID to participant label mapping |
| `aux_streams` | list[AuxStreamConfig] or None | `None` | Auxiliary video stream configurations |
| `anonymize` | str or None | `None` | Face anonymization mode: `"blur"` or `"black"` |
| `anonymize_padding` | float | `0.3` | Fraction of bbox size added as anonymization margin |
| `video_name` | str or None | `None` | Source video stem, set automatically in project mode |
| `conditions` | str or None | `None` | Pipe-delimited condition tags, set automatically in project mode |

---

## AuxStreamConfig

Defined in `mindsight/pipeline_config.py`. Configuration for a single auxiliary video stream.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | str | (required) | File path or device index string |
| `video_type` | VideoType | (required) | How the video is framed (see enum below) |
| `stream_label` | str | (required) | User-defined stream label (e.g. `"left_eye_cam"`) |
| `participants` | list[str] | (required) | Participant labels visible in this stream |
| `auto_detect_faces` | bool | `True` | Run face detection on wide/face streams |

`video_type` is a `VideoType` enum (also in `mindsight/pipeline_config.py`):

| Value | Meaning |
|-------|---------|
| `wide_closeup` | Multiple participants in view |
| `face_closeup` | Single-person face |
| `eye_only` | Single-person eye region |
| `custom` | Arbitrary user-defined type |

Auxiliary streams are optional video feeds (a dedicated eye-tracking camera, a wide room camera, a first-person view, etc.) that are frame-synchronised with the main source. They are exposed in `FrameContext['aux_frames']` (keyed by `(pid, stream_label, video_type)`) for consumption by plugins, but are not processed by any built-in pipeline stage. Plugins declare `preferred_video_types` and the system auto-routes matching streams. The helper `find_aux_frame(aux_frames, pid, video_type=..., stream_label=...)` looks up the best match for a participant.

---

## ProjectConfig

Defined in `mindsight/pipeline_config.py`. Project-level metadata loaded from `project.yaml`. Stores study-level information (condition tags, participant labels, output settings) that is separate from pipeline processing parameters in `pipeline.yaml`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline_path` | str or None | `None` | Relative or absolute path to the pipeline YAML file |
| `conditions` | dict[str, list[str]] | `{}` | Per-video condition tags: `{video_filename: [tag, ...]}` |
| `participants` | dict[str, dict[int, str]] | `{}` | Per-video participant labels: `{video_filename: {track_id: label}}` |
| `output` | ProjectOutputConfig | (defaults) | Output directory configuration |

When `project.yaml` exists, its `participants` section takes precedence over `participant_ids.csv`. If neither exists, MindSight uses default labels (`P0`, `P1`, etc.).

---

## ProjectOutputConfig

Defined in `mindsight/pipeline_config.py`. Controls where project-level outputs are written.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `directory` | str or None | `None` | Output root directory (absolute or relative to project root). `None` defaults to `project/Outputs/`. |

The `resolve_root(project)` method returns the resolved output root as a `Path`.

---

## PhenomenaConfig

Defined in `mindsight/Phenomena/phenomena_config.py`. All phenomena-related configuration in one object.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `joint_attention` | bool | `False` | Enable joint attention tracking |
| `ja_window` | int | `0` | Sliding-window size (frames) for temporal JA smoothing |
| `ja_window_thresh` | float | `0.70` | Fraction of window frames required for JA confirmation |
| `ja_quorum` | float | `1.0` | Fraction of persons required for joint attention |
| `mutual_gaze` | bool | `False` | Enable mutual gaze detection |
| `social_ref` | bool | `False` | Enable social referencing detection |
| `social_ref_window` | int | `60` | Window size (frames) for social referencing |
| `gaze_follow` | bool | `False` | Enable gaze-following detection |
| `gaze_follow_lag` | int | `30` | Maximum lag (frames) for gaze-following alignment |
| `gaze_aversion` | bool | `False` | Enable gaze aversion detection |
| `aversion_window` | int | `60` | Window size (frames) for gaze aversion |
| `aversion_conf` | float | `0.5` | Confidence threshold for gaze aversion |
| `scanpath` | bool | `False` | Enable scanpath recording |
| `scanpath_dwell` | int | `8` | Minimum fixation dwell (frames) for scanpath points |
| `gaze_leader` | bool | `False` | Enable gaze leadership detection |
| `gaze_leader_tips` | bool | `False` | Enable tip-based gaze leadership |
| `gaze_leader_tip_lag` | int | `15` | Lag (frames) for tip-based gaze leadership |
| `attn_span` | bool | `False` | Enable attention span tracking |

The `from_namespace(ns)` classmethod honours the `--all-phenomena` flag: when set, all boolean toggles default to `True`.
