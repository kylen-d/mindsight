# FrameContext Keys

`FrameContext` is the mutable per-frame data carrier that flows through all pipeline
stages. Each stage reads what it needs and writes its results. The class lives in
`mindsight/pipeline_config.py` and supports dict-like access (`ctx['key']`,
`ctx.get('key', default)`, `'key' in ctx`).

Each frame's context is a fresh `FrameContext` seeded with the run-level base values (see
**Run Context Base Keys**) plus `frame` / `frame_no` / `aux_frames`; the per-frame stages
then populate the rest. The orchestrator is `process_frame()` and `run()` in
`mindsight/pipeline.py` (`MindSight.py` is only a thin launcher shim).

---

## Constructor Keys

Set when the `FrameContext` is instantiated at the start of each frame.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `frame` | ndarray | constructor | all stages | The current BGR frame (mutated in-place by overlay) |
| `frame_no` | int | constructor | detection, phenomena | Zero-based frame counter |
| `aux_frames` | dict | run loop | plugins | Auxiliary stream frames, keyed by `(pid, stream_label, video_type)` |

## Detection Pipeline Keys

Written by `detection_pipeline.run_detection_step()` (and the depth step).

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `all_dets` | list[Detection] | detection_pipeline | gaze_pipeline, run loop | All detections (persons + objects) for this frame |
| `persons` | list[Detection] | detection_pipeline | gaze_pipeline, process_frame | Person-class detections only |
| `objects` | list[Detection] | detection_pipeline | gaze_pipeline, phenomena | Non-person detections only |
| `detection_frame` | ndarray | detection_pipeline | gaze_pipeline | Scaled frame used for detection (may differ from display frame) |
| `inverse_scale` | float | detection_pipeline | gaze_pipeline | Scale factor mapping detection coords back to the original frame |
| `depth_map` | ndarray | depth step / run loop | gaze/snap scoring | Per-pixel depth map (present only when depth is enabled) |
| `depth_cfg` | DepthConfig | process_frame | gaze/snap scoring | Active depth config (set only when depth is enabled) |
| `cached_all_dets_out` | list[Detection] | process_frame | run loop | This frame's detections, exported for caching on detection frames |

## Gaze Pipeline Keys

Written by `gaze_pipeline.run_gaze_step()`.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `persons_gaze` | list[tuple] | gaze_pipeline | phenomena, overlay, run loop | Per-person gaze as `(origin, ray_end, angles)` tuples; `angles` may be `None` |
| `face_confs` | list[float] | gaze_pipeline | overlay | Face detection confidence per detected face |
| `face_bboxes` | list[tuple] | gaze_pipeline | anonymization, overlay | Face bounding boxes `(x1, y1, x2, y2)` |
| `face_track_ids` | list[int] | gaze_pipeline | anonymization, run loop | Stable track ID for each face |
| `all_targets` | list[Detection] | gaze_pipeline | intersection, phenomena | Combined object + person targets for ray intersection |
| `hits` | set[tuple] | gaze_pipeline | JA, phenomena, overlay | Set of `(face_idx, target_idx)` pairs that intersect this frame |
| `hit_events` | list[dict] | gaze_pipeline | run loop, event log | Structured hit event records for CSV logging |
| `lock_info` | list[tuple] | gaze_pipeline | overlay | Per-person `(locked_obj_idx_or_None, dwell_fraction)` lock state |
| `ray_snapped` | list[bool] | gaze_pipeline | overlay | Per-person flag: whether the ray was snapped this frame |
| `ray_extended` | list[bool] | gaze_pipeline | overlay | Per-person flag: whether the ray was extended this frame |
| `faces` | list | gaze_pipeline | run loop (cache) | Raw face detection results |

## Process Frame Keys

Written by `process_frame()` in `mindsight/pipeline.py` after gaze and JA computation.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `joint_objs` | set[int] | process_frame | overlay, phenomena, run loop | Indices of objects currently under joint attention |
| `tip_convergences` | list[tuple] | process_frame | overlay | Gaze-tip convergence points |
| `tip_radius` | int | process_frame | overlay | Pixel radius used for convergence detection |
| `detect_extend` | float | process_frame | overlay | Current detect-extend distance (for debug display) |
| `detect_extend_scope` | str | process_frame | overlay | Current detect-extend scope setting |

## Phenomena Pipeline Keys

Written by `phenomena_pipeline.update_phenomena_step()`.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `confirmed_objs` | set[int] | phenomena_pipeline | overlay | Object indices confirmed by the temporal JA window |
| `extra_hud` | str or None | phenomena_pipeline | overlay | JA window-fill / accuracy-mode status line for the HUD |
| `joint_pct` | float | phenomena_pipeline | overlay, run loop | Running JA confirmation percentage (0--100) |

## Run Context Base Keys

Set once per run in `run()` and carried across all frames (each frame's `FrameContext` is
seeded from this base).

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `source` | str | run() | run loop | Input source path or device string |
| `smoother` | GazeSmootherReID | run() | gaze_pipeline | Per-person gaze-angle smoother with re-ID support |
| `locker` | GazeLockTracker | run() | gaze_pipeline | Gaze lock-on state machine |
| `snap_temporal` | SnapTemporalState | run() | ray forming | Temporal state for snap engage/release hysteresis |
| `smooth_snap_tracker` | SmoothSnapTracker | run() | ray forming | EMA state for smooth-snap interpolation |
| `all_trackers` | list | run() | phenomena_pipeline | All active phenomena tracker instances |
| `look_counts` | dict | run() | run loop | Cumulative per-object look frame counts |
| `heatmap_path` | str or None | run() | finalize | Output path for heatmap image(s) |
| `heatmap_gaze` | dict | run() | finalize | Accumulated per-participant gaze points for heatmaps |
| `charts_path` | bool/str/None | run() | finalize | Output path for post-run charts |
| `summary_path` | str or None | run() | finalize | Output path for the summary CSV |
| `pid_map` | dict or None | run() | overlay, finalize | Track ID to participant-label mapping |
| `anonymize` | str or None | run() | process_frame | Face anonymization mode (`"blur"` or `"black"`) |
| `anonymize_padding` | float | run() | process_frame | Fraction of bbox added as anonymization margin |
| `anon_smoother` | AnonSmoother or None | run() | process_frame | Temporal smoother for anonymization bboxes |
| `face_det` | face detector | run() | aux enrichment | Shared face detector instance |
| `video_name` | str or None | run() | finalize, CSV | Source video stem (project mode) |
| `conditions` | str or None | run() | finalize, CSV | Pipe-delimited condition tags (project mode) |
| `video_fps` | float | run() | finalize | Source video frame rate |
| `gazelle_provider` | GazelleProvider or None | run() | gaze_pipeline | Gaze-LLE heatmap provider (blend mode) |
| `gazelle_blender` | GazeLLEBlender or None | run() | ray forming | Gaze-LLE belief blender (present when provider + ray_cfg set) |
| `ray_cfg` | RayFormingConfig or None | run() | ray forming | Ray-forming configuration |
| `ray_object_snap` | ObjectSnap or None | run() | ray forming | Object-snap engine (present when ray_cfg set) |
| `data_plugins` | list | run() | finalize | Data-collection plugins (empty for in-repo runs) |

## Run Loop Keys

Set or updated per-frame inside the main run loop.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `do_cache` | bool | run loop | process_frame | Whether this is a detection frame (drives detection caching) |
| `cached_all_dets` | list[Detection] | run loop | detection_pipeline | Cached detections reused on non-detection frames |
| `cached_faces` | list | run loop | gaze_pipeline | Cached face detections reused on non-detection frames |
| `prev_persons_gaze` | list[tuple] | run loop | detection plugins | Previous frame's gaze data |
| `prev_face_track_ids` | list[int] | run loop | detection plugins | Previous frame's face track IDs |
| `fps` | float | run loop | overlay | Current frames-per-second measurement |
| `n_dets` | int | run loop | overlay | Number of hit events this frame |
| `is_joint` | bool | run loop | overlay, CSV | Whether joint attention is active this frame (object JA OR tip convergence) |
| `is_confirmed` | bool | run loop | overlay, CSV | Whether JA is temporally confirmed this frame |

## Finalize Keys

Set on the post-run context passed to run finalization.

| Key | Type | Written By | Consumed By | Description |
|-----|------|-----------|-------------|-------------|
| `total_frames` | int | run() | finalize | Total frames processed in the run |
| `total_hits` | int | run() | finalize | Total ray-object hits across the run |
