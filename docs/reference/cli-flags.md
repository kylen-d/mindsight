# CLI Flags Reference

All flags accepted by `python MindSight.py`. Run `python MindSight.py --help` for the built-in help text.

---

## Orchestration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--source` | str | `"0"` | Input source: video file path, image path, or webcam index |
| `--save` | str (optional) | None | Save annotated output video to this path (omit value for auto-named file) |
| `--log` | str | None | Path for per-frame CSV log output |
| `--summary` | str (optional) | None | Path for post-run summary CSV (omit value for auto-named file) |
| `--heatmap` | str (optional) | None | Path to save gaze heatmap image (omit value for auto-named file) |
| `--charts` | str (optional) | None | Path to save chart images (omit value for auto-named file) |
| `--pipeline` | str | None | Path to a `pipeline.yaml` configuration file |
| `--project` | str | None | Path to a MindSight project directory |
| `--participant-ids` | str | None | Comma-separated participant ID assignments (e.g. `"1:Alice,2:Bob"`) |
| `--participant-csv` | str | None | CSV file mapping track IDs to participant labels |
| `--aux-stream` | str (repeatable) | [] | Auxiliary stream in `PID:TYPE:SOURCE` format (may be specified multiple times) |
| `--device` | str | `"auto"` | Compute device for model inference (`"auto"`, `"cpu"`, `"cuda"`, `"mps"`) |
| `--anonymize` | str | None | Face anonymization mode: `blur` or `black` |
| `--anonymize-padding` | float | `0.3` | Fraction of bounding-box size added as margin around anonymized faces |

## Performance

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--fast` | flag | off | Enable fast mode (reduces processing for higher FPS) |
| `--skip-phenomena` | int | `0` | Run phenomena trackers only every N frames (0 = every frame) |
| `--lite-overlay` | flag | off | Use lightweight overlay rendering (fewer draw calls) |
| `--no-dashboard` | flag | off | Disable the side-panel dashboard display |
| `--profile` | flag | off | Print per-stage timing information each frame |

## Detection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `"yolov8n.pt"` | YOLO model file for object detection |
| `--conf` | float | `0.35` | Minimum detection confidence threshold |
| `--classes` | str[] | `[]` | Whitelist of YOLO class names to detect (empty = all) |
| `--blacklist` | str[] | `[]` | Class names to exclude from detections |
| `--skip-frames` | int | `1` | Process detection every N-th frame (1 = every frame) |
| `--detect-scale` | float | `1.0` | Scale factor applied to input before detection (< 1.0 for speed) |
| `--vp-file` | str | None | Path to a Visual Prompt `.vp.json` file for YOLOE |
| `--vp-model` | str | `"yoloe-26l-seg.pt"` | YOLOE model file used with `--vp-file` |
| `--obj-persistence` | int | `0` | Keep detections alive for N frames after a miss (0 = disabled) |

## Gaze

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ray-length` | float | `1.0` | Gaze ray length multiplier |
| `--conf-ray` | flag | off | Scale ray length by face-detection confidence |
| `--gaze-tips` | flag | off | Enable gaze-tip convergence detection |
| `--tip-radius` | int | `80` | Pixel radius for gaze-tip convergence check |
| `--adaptive-ray` | str | `"off"` | Adaptive ray mode: `off`, `extend`, or `snap` |
| `--snap-dist` | float | `150.0` | Maximum snap distance in pixels |
| `--snap-bbox-scale` | float | `0.0` | Fraction of bbox half-diagonal added to snap radius |
| `--snap-w-dist` | float | `1.0` | Snap scoring weight for normalized distance penalty |
| `--snap-w-size` | float | `0.0` | Snap scoring weight for angular size reward |
| `--snap-w-intersect` | float | `0.5` | Snap scoring bonus for ray-bbox intersection |
| `--hit-conf-gate` | float | `0.0` | Minimum face confidence required for a hit to register |
| `--detect-extend` | float | `0.0` | Extra pixels added past visual ray for detection (0 = visual parity) |
| `--detect-extend-scope` | str | `"objects"` | What detect-extend applies to: `objects`, `phenomena`, or `both` |
| `--gaze-cone` | float | `0.0` | Half-angle (degrees) of gaze cone (0 = ray only) |
| `--gaze-lock` | flag | off | Enable gaze lock-on behaviour |
| `--dwell-frames` | int | `15` | Frames of sustained gaze required to trigger lock-on |
| `--lock-dist` | int | `100` | Maximum pixel distance for lock-on to persist |
| `--gaze-debug` | flag | off | Draw debug annotations for gaze processing |
| `--snap-switch-frames` | int | `8` | Hysteresis frames before switching snap target |
| `--reid-grace-seconds` | float | `1.0` | Grace period (seconds) for face re-identification after a miss |
| `--forward-gaze-threshold` | float | `5.0` | Yaw/pitch threshold (degrees) below which gaze is considered forward-facing |

## Gaze Backends

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mgaze-model` | str | (built-in ONNX path) | Path to MGaze model (`.onnx` or `.pt`). Inference mode auto-detected from extension. |
| `--mgaze-arch` | str | None | MGaze backbone architecture override |
| `--mgaze-dataset` | str | `"gaze360"` | Dataset the MGaze model was trained on |
| `--l2cs-model` | str | None | Path to L2CS model file (enables L2CS backend) |
| `--l2cs-arch` | str | `"ResNet50"` | L2CS backbone architecture |
| `--l2cs-dataset` | str | `"gaze360"` | Dataset the L2CS model was trained on |
| `--unigaze-model` | str | None | Path to UniGaze model file (enables UniGaze backend) |
| `--gazelle-model` | str | None | Path to Gazelle model file (enables Gazelle backend) |
| `--gazelle-name` | str | `"gazelle_dinov2_vitb14"` | Gazelle model variant name |
| `--gazelle-inout-threshold` | float | `0.5` | Gazelle in-frame / out-of-frame gaze threshold |

## Phenomena

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--joint-attention` | flag | off | Enable joint attention tracking |
| `--ja-window` | int | `0` | Sliding-window size (frames) for temporal JA smoothing (0 = instantaneous) |
| `--ja-window-thresh` | float | `0.70` | Fraction of window frames an object must be attended for JA confirmation |
| `--ja-quorum` | float | `1.0` | Fraction of detected persons required for joint attention (1.0 = all) |
| `--mutual-gaze` | flag | off | Enable mutual gaze detection |
| `--social-ref` | flag | off | Enable social referencing detection |
| `--social-ref-window` | int | `60` | Window size (frames) for social referencing |
| `--gaze-follow` | flag | off | Enable gaze-following detection |
| `--gaze-follow-lag` | int | `30` | Maximum lag (frames) for gaze-following alignment |
| `--gaze-aversion` | flag | off | Enable gaze aversion detection |
| `--aversion-window` | int | `60` | Window size (frames) for gaze aversion |
| `--aversion-conf` | float | `0.5` | Confidence threshold for gaze aversion |
| `--scanpath` | flag | off | Enable scanpath recording |
| `--scanpath-dwell` | int | `8` | Minimum fixation dwell (frames) for scanpath points |
| `--gaze-leader` | flag | off | Enable gaze leadership detection |
| `--gaze-leader-tips` | flag | off | Enable tip-based gaze leadership |
| `--gaze-leader-tip-lag` | int | `15` | Lag (frames) for tip-based gaze leadership |
| `--attn-span` | flag | off | Enable attention span tracking |
| `--all-phenomena` | flag | off | Enable all phenomena trackers at once |
