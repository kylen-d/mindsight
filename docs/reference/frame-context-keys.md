# FrameContext Keys

`FrameContext` is the mutable per-frame data carrier that flows through all pipeline stages. Each stage reads what it needs and writes its results. The class lives in `pipeline_config.py` and supports dict-like access (`ctx['key']`, `ctx.get('key', default)`, `'key' in ctx`).

---

## Constructor Keys

Set when the `FrameContext` is instantiated at the start of each frame.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `frame` | ndarray | constructor | all stages | The current BGR frame (mutated in-place by overlay) |
| `frame_no` | int | constructor | data_pipeline, phenomena | Zero-based frame counter |

## Detection Pipeline Keys

Written by `detection_pipeline.run_detection_step()`.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `all_dets` | list[Detection] | detection_pipeline | gaze_pipeline, data_pipeline | All detections (persons + objects) for this frame |
| `persons` | list[Detection] | detection_pipeline | gaze_pipeline, process_frame | Person-class detections only |
| `objects` | list[Detection] | detection_pipeline | gaze_pipeline, phenomena | Non-person detections only |
| `detection_frame` | ndarray | detection_pipeline | gaze_pipeline | Scaled frame used for detection (may differ from display frame) |
| `inverse_scale` | float | detection_pipeline | gaze_pipeline | Scale factor to map detection coords back to original frame |

## Gaze Pipeline Keys

Written by `gaze_pipeline.run_gaze_step()`.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `persons_gaze` | list[dict] | gaze_pipeline | phenomena, overlay, data_pipeline | Per-person gaze data (origin, tip, yaw, pitch, track_id, etc.) |
| `face_confs` | list[float] | gaze_pipeline | overlay | Face detection confidence per detected face |
| `face_bboxes` | list[tuple] | gaze_pipeline | anonymization, overlay | Face bounding boxes (x1, y1, x2, y2) |
| `face_track_ids` | list[int] | gaze_pipeline | anonymization, data_pipeline | Stable track ID for each face |
| `all_targets` | list[Detection] | gaze_pipeline | intersection, phenomena | Combined object + person targets for ray intersection |
| `hits` | dict[int, set] | gaze_pipeline | JA, phenomena, overlay | Map from person track_id to set of hit object indices |
| `hit_events` | list[dict] | gaze_pipeline | data_pipeline | Structured hit event records for CSV logging |
| `lock_info` | dict | gaze_pipeline | overlay | Gaze-lock state per person (locked target, dwell progress) |
| `ray_snapped` | dict[int, bool] | gaze_pipeline | overlay | Whether each person's ray was snapped this frame |
| `ray_extended` | dict[int, bool] | gaze_pipeline | overlay | Whether each person's ray was extended this frame |
| `faces` | list | gaze_pipeline | detection_pipeline (cache) | Raw face detection results |

## Process Frame Keys

Written by `process_frame()` in `MindSight.py` after gaze and JA computation.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `joint_objs` | set[int] | process_frame | overlay, phenomena | Indices of objects currently under joint attention |
| `tip_convergences` | list[tuple] | process_frame | overlay | Gaze-tip convergence points (x, y, count) |
| `tip_radius` | int | process_frame | overlay | Pixel radius used for convergence detection |
| `detect_extend` | float | process_frame | overlay | Current detect-extend distance (for debug display) |
| `detect_extend_scope` | str | process_frame | overlay | Current detect-extend scope setting |

## Phenomena Pipeline Keys

Written by `phenomena_pipeline.update_phenomena_step()`.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `confirmed_objs` | set[int] | phenomena_pipeline | overlay | Object indices confirmed by temporal JA window |
| `extra_hud` | list[str] | phenomena_pipeline | overlay | Additional HUD text lines from phenomena trackers |
| `joint_pct` | float | phenomena_pipeline | overlay, data_pipeline | JA window confirmation percentage (0.0--1.0) |

## Run Context Base Keys

Set once per run in the `run()` function and carried across all frames.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `smoother` | GazeSmootherReID | run() | gaze_pipeline | Per-person gaze angle smoother with re-ID support |
| `locker` | GazeLockTracker | run() | gaze_pipeline | Gaze lock-on state machine |
| `snap_hysteresis` | SnapHysteresisTracker | run() | gaze_pipeline | Hysteresis tracker for snap target switching |
| `all_trackers` | list | run() | phenomena_pipeline | All active phenomena tracker instances |
| `look_counts` | dict | run() | data_pipeline | Cumulative per-object look frame counts |
| `heatmap_path` | str or None | run() | data_pipeline | Output path for heatmap image |
| `heatmap_gaze` | list | run() | data_pipeline | Accumulated gaze points for heatmap generation |
| `charts_path` | str or None | run() | data_pipeline | Output path for chart images |
| `summary_path` | str or None | run() | data_pipeline | Output path for summary CSV |
| `pid_map` | dict or None | run() | overlay, data_pipeline | Track ID to participant label mapping |
| `anonymize` | str or None | run() | process_frame | Face anonymization mode (`"blur"` or `"black"`) |
| `anonymize_padding` | float | run() | process_frame | Fraction of bbox added as anonymization margin |
| `anon_smoother` | AnonSmoother | run() | process_frame | Temporal smoother for anonymization bounding boxes |
| `source` | str | run() | data_pipeline | The input source path or device string |

## Run Loop Keys

Set or updated per-frame inside the main run loop.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `aux_frames` | dict[str, ndarray] | run loop | plugins | Auxiliary stream frames keyed by `"PID:TYPE"` |
| `do_cache` | bool | run loop | process_frame | Whether to cache detections this frame (skip-frames logic) |
| `cached_all_dets` | list[Detection] | run loop | detection_pipeline | Cached detections from the previous detection frame |
| `cached_faces` | list | run loop | gaze_pipeline | Cached face detections from the previous detection frame |
| `fps` | float | run loop | overlay | Current frames-per-second measurement |
| `n_dets` | int | run loop | overlay | Number of detections this frame |
| `is_joint` | bool | run loop | overlay | Whether joint attention is active this frame |
| `is_confirmed` | bool | run loop | overlay | Whether JA is temporally confirmed this frame |
