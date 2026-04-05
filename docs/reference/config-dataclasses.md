# Configuration Dataclasses

MindSight groups its many parameters into typed dataclasses defined in `ms/pipeline_config.py` and `ms/Phenomena/phenomena_config.py`. Each dataclass has a `from_namespace(ns)` classmethod that constructs it from an `argparse.Namespace`.

---

## GazeConfig

Defined in `ms/pipeline_config.py`. All gaze-estimation and ray-intersection tuning parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ray_length` | float | `1.0` | Gaze ray length multiplier |
| `adaptive_ray` | str | `"off"` | Adaptive ray mode: `"off"`, `"extend"`, or `"snap"` |
| `snap_dist` | float | `150.0` | Maximum snap distance in pixels |
| `snap_bbox_scale` | float | `0.0` | Fraction of bbox half-diagonal added to snap radius |
| `snap_w_dist` | float | `1.0` | Snap scoring weight: normalized distance penalty |
| `snap_w_size` | float | `0.0` | Snap scoring weight: angular size reward (off by default) |
| `snap_w_intersect` | float | `0.5` | Snap scoring bonus for ray-bbox intersection |
| `conf_ray` | bool | `False` | Scale ray length by face-detection confidence |
| `gaze_tips` | bool | `False` | Enable gaze-tip convergence detection |
| `tip_radius` | int | `80` | Pixel radius for convergence check |
| `gaze_cone_angle` | float | `0.0` | Half-angle (degrees) of gaze cone; 0 = ray only |
| `hit_conf_gate` | float | `0.0` | Minimum face confidence required for a hit to register |
| `detect_extend` | float | `0.0` | Extra pixels past visual ray for detection (0 = visual parity) |
| `detect_extend_scope` | str | `"objects"` | What detect-extend applies to: `"objects"`, `"phenomena"`, or `"both"` |
| `ja_quorum` | float | `1.0` | Fraction of detected persons required for joint attention |
| `gaze_debug` | bool | `False` | Draw debug annotations for gaze processing |
| `forward_gaze_threshold` | float | `5.0` | Yaw/pitch threshold (degrees) below which gaze is forward-facing |

---

## DetectionConfig

Defined in `ms/pipeline_config.py`. Object-detection parameters passed through to YOLO.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conf` | float | `0.35` | Minimum detection confidence threshold |
| `class_ids` | list or None | `None` | Resolved YOLO class IDs to detect (None = all) |
| `blacklist` | set | `set()` | Set of class names to exclude from detections |
| `detect_scale` | float | `1.0` | Scale factor applied to input before detection |

Note: `from_namespace(ns, class_ids, blacklist)` takes pre-resolved class IDs and blacklist set as additional arguments.

---

## TrackerConfig

Defined in `ms/pipeline_config.py`. Parameters used by `run()` to construct per-run tracker instances.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gaze_lock` | bool | `False` | Enable gaze lock-on behaviour |
| `dwell_frames` | int | `15` | Frames of sustained gaze required to trigger lock-on |
| `lock_dist` | int | `100` | Maximum pixel distance for lock-on to persist |
| `skip_frames` | int | `1` | Process detection every N-th frame |
| `obj_persistence` | int | `0` | Keep detections alive for N frames after a miss |
| `snap_switch_frames` | int | `8` | Hysteresis frames before switching snap target |
| `reid_grace_seconds` | float | `1.0` | Grace period (seconds) for face re-ID after a miss |
| `reid_max_dist` | int | `200` | Maximum pixel distance for face re-identification |

---

## OutputConfig

Defined in `ms/pipeline_config.py`. Paths and flags controlling run-loop outputs.

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

Defined in `ms/pipeline_config.py`. Configuration for a single auxiliary video stream mapped to a participant.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pid` | str | (required) | Participant label (e.g. `"S70"`) |
| `stream_type` | str | (required) | Purpose tag (e.g. `"eye_camera"`, `"first_person_view"`) |
| `source` | str | (required) | File path or device index string |

Auxiliary streams are optional per-participant video feeds that are frame-synchronised with the main source. They are exposed in `FrameContext['aux_frames']` for consumption by plugins but are not processed by any built-in pipeline stage.

---

## ProjectConfig

Defined in `ms/pipeline_config.py`. Project-level metadata loaded from `project.yaml`. Stores study-level information (condition tags, participant labels, output settings) that is separate from pipeline processing parameters in `pipeline.yaml`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline_path` | str or None | `None` | Relative or absolute path to the pipeline YAML file |
| `conditions` | dict[str, list[str]] | `{}` | Per-video condition tags: `{video_filename: [tag, ...]}` |
| `participants` | dict[str, dict[int, str]] | `{}` | Per-video participant labels: `{video_filename: {track_id: label}}` |
| `output` | ProjectOutputConfig | (defaults) | Output directory configuration |

When `project.yaml` exists, its `participants` section takes precedence over `participant_ids.csv`. If neither exists, MindSight uses default labels (`P0`, `P1`, etc.).

---

## ProjectOutputConfig

Defined in `ms/pipeline_config.py`. Controls where project-level outputs are written.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `directory` | str or None | `None` | Output root directory (absolute or relative to project root). `None` defaults to `project/Outputs/`. |

The `resolve_root(project)` method returns the resolved output root as a `Path`.

---

## PhenomenaConfig

Defined in `ms/Phenomena/phenomena_config.py`. All phenomena-related configuration in one object.

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
