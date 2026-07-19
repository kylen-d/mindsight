# CLI Flags Reference

Every flag accepted by `python MindSight.py`. Run `python MindSight.py --help` for the
built-in help text. This page is generated from the live parser
(`mindsight/cli_flags.py` `CORE_FLAGS` plus the gaze / object-detection / phenomena
plugin registries): **151 flags** in total.

## How flags relate to `pipeline.yaml` and the Inference Settings dialog

The CLI flags, the `pipeline.yaml` keys, and the GUI's **Inference Settings** dialog are
three front-ends onto the same underlying pipeline parameters:

- **`pipeline.yaml`** (loaded with `--pipeline`) sets the same values declaratively. Most
  flags have a YAML home; the exact key spellings (which are grouped into `detection:`,
  `gaze:`, `output:`, `depth:`, `phenomena:`, `performance:` sections, plus a `plugins:`
  pass-through) are documented on the
  [pipeline.yaml schema](pipeline-yaml-schema.md) page. Plugin/backend flags that have no
  dedicated YAML key (for example `--mgaze-model`, `--rf-gazelle-model`, the
  `--gazelle-*` family) are set through the `plugins:` pass-through, which accepts any
  flag's argparse destination with hyphens rewritten as underscores.
- **Inference Settings** (Tools -> Inference Settings in the GUI) is the per-run authority
  for study runs; its fields map onto the same parameters. See
  [Inference Settings](inference-settings.md). The decoupled **Inference Tuning** tab is a
  non-authoritative playground for the same knobs.

**Precedence:** a CLI flag always overrides the value from `--pipeline`. YAML values are
applied only for parameters the user did not type on the command line. Within the GUI, the
Inference Settings dialog is authoritative for a study run.

Flags marked *flag* are boolean switches (`store_true`); *str (optional)* flags (`--save`,
`--summary`, `--heatmap`, `--charts`) may be given with no value to use an auto-named
output path, or with a path to override it. *str (repeatable)* flags (`--aux-stream`) may
be supplied multiple times.

---

### Input & Orchestration

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--source` | str | `"0"` | Video input source, defaults to webcam |
| `--pipeline` | str | None | Load pipeline configuration from a YAML file. CLI flags override YAML values. |
| `--project` | str | None | Run in project mode: process all staged videos (DIR/Inputs/Runs/ run folders, or the legacy flat DIR/Inputs/Videos/) using DIR/Pipeline/pipeline.yaml as config. |
| `--no-resume` | flag | off | Project mode: reprocess every video, ignoring the resume ledger. Does not archive prior outputs. Default resumes -- skipping videos whose ledger entry is done with an unchanged config. |
| `--preflight` | flag | off | Print the project readiness report (requires --project) and exit. Checks pipeline config, weights, VP file, runs, participants/conditions, device, disk, and plugin load errors. Exit 0 if no failures, else 1. |
| `--participant-ids` | str | None | Comma-separated participant labels for single-video mode. Positional: first label maps to track 0, second to track 1, etc. E.g. --participant-ids S70,S71,S72 |
| `--participant-csv` | str | None | Path to a participant_ids.csv mapping video filenames to custom participant labels (see docs for format). |
| `--aux-stream` | str (repeatable) | None | Auxiliary video stream. Format: SOURCE:VIDEO_TYPE:LABEL:PID1,PID2 where SOURCE is the file path, VIDEO_TYPE is one of eye_only/face_closeup/wide_closeup/custom, LABEL is a user-defined stream label, and PIDS is a comma-separated list of participant labels. Repeatable for multiple streams. |
| `--aux-auto-detect` | flag | `True` | Enable automatic face detection on wide/face auxiliary streams (default: enabled). |
| `--device` | `auto` / `cpu` / `cuda` / `mps` | `"auto"` | Compute device for all backends: auto, cpu, cuda, or mps.  'auto' selects CUDA > MPS > CPU  (default: auto). |

### Output

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--save` | str (optional) | None | Save annotated video. Omit a value to use Outputs/Video/[stem]_Video_Output.mp4, or supply a custom path. |
| `--log` | str | None | Path for the per-frame CSV log. |
| `--summary` | str (optional) | None | Save post-run summary CSV. Omit a value to use Outputs/CSV Files/[stem]_Summary_Output.csv, or supply a custom path. |
| `--heatmap` | str (optional) | None | Save per-participant scene gaze heatmaps. Omit a value to use Outputs/heatmaps/[stem]_Heatmap_Output (one PNG per participant), or supply a custom directory/prefix path. |
| `--charts` | str (optional) | None | Generate post-run time-series charts for each phenomena tracker. Omit a value to use Outputs/Charts/[stem]_Charts.png, or supply a custom path. |
| `--anonymize` | `blur` / `black` | None | Anonymize faces in the output video: 'blur' applies heavy Gaussian blur, 'black' fills with a solid rectangle. |
| `--anonymize-padding` | float | `0.3` | Fraction of face bbox size added as padding for anonymization (default: 0.3). |

### Performance

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--fast` | flag | off | Enable bundled performance optimizations: skip phenomena on non-detection frames, throttle dashboard bridge, reduce GUI poll rate. |
| `--skip-phenomena` | int | `0` | Run phenomena trackers only every N frames (0 = every frame). Independent of --skip-frames.  (default: 0) |
| `--lite-overlay` | flag | off | Minimal overlay: disable cone rendering, convergence markers, dwell arcs, and debug text.  Keeps gaze arrows, boxes, badges. |
| `--no-dashboard` | flag | off | Skip dashboard composition for maximum throughput. Displays the raw annotated frame only. |
| `--profile` | flag | off | Print per-stage timing breakdown every 100 frames. |

### Detection

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--model` | str | `"yolo11n.pt"` | Object Detection Model (default: yolo11n.pt since v1.1; yolo11n.onnx in the Models tab is a faster option for CPU-bound installs) |
| `--face-model` | str | `"r34"` | RetinaFace backbone variant (default: r34 -- the v1.1 eval pick: best accuracy AND ~40% faster than mnet_v2 on the CoreML path; weights auto-download on first use) |
| `--no-face-eye-origin` | flag | off | Anchor gaze rays at the face-bbox centre (the pre-v1.1 behavior), overriding the eye-midpoint default |
| `--conf` | float | `0.35` | Object detection confidence threshold, defaults to 0.35 |
| `--classes` | str[] | `[]` | Specified YOLO Object Detection Classes Prompt |
| `--blacklist` | str[] | `[]` | Specified YOLO Object Detection Classes Blacklist |
| `--skip-frames` | int | `1` | Frames between object detection. Higher = faster but less accurate. (Defaults to 1, i.e. process every frame) |
| `--detect-scale` | float | `1.0` | Detection scale for Object Recognition |
| `--vp-file` | str | None | Path to visual prompt file for use with YOLOE object detection models |
| `--vp-model` | str | `"yoloe-26l-seg.pt"` | YOLOE model to use alongside visual prompting for object detection |
| `--no-detector` | flag | off | Run without any object-detection model: faces, gaze rays, and gaze-tip phenomena only (no object hits or lock-on). Not compatible with --vp-file. |
| `--obj-persistence` | int | `0` | Dead-reckon object bboxes for N frames after they disappear (default 0). |
| `--merge-overlaps` | flag | off | Merge or filter overlapping same-class detections. |
| `--merge-overlap-strategy` | `filter` / `merge` / `dynamic` | `"dynamic"` | Overlap strategy: 'filter' keeps highest-conf box, 'merge' creates encompassing box, 'dynamic' chooses per-cluster based on confidence and size (default: dynamic). |
| `--merge-overlap-threshold` | float | `0.7` | Overlap threshold to trigger merge (default: 0.7). |

### Gaze & Ray Intersection

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--ray-length` | float | `1.0` | Gaze ray-length multiplier, default 1.0 |
| `--conf-ray` | flag | off | Dynamically adjust gaze ray-length based on gaze confidence value |
| `--gaze-tips` | flag | off | Adds circular bounding-box to tip of gaze-rays, used to determine intersection between gaze-rays. Set radius with --tip-radius (default 80). |
| `--tip-radius` | int | `80` | Pixel radius for --gaze-tips (default 80) |
| `--adaptive-ray` | `off` / `extend` / `snap` | `"off"` | Adaptive ray mode: 'off' disables, 'extend' freely extends the ray toward the nearest object, 'snap' locks the endpoint to the object centre (default: off). |
| `--snap-dist` | float | `150.0` | Maximum snap distance in pixels. |
| `--snap-bbox-scale` | float | `0.0` | Fraction of bbox half-diagonal added to snap radius (default 0.0) |
| `--snap-w-dist` | float | `1.0` | Snap scoring weight for normalized distance penalty (default 1.0) |
| `--snap-w-angle` | float | `0.8` | Snap scoring weight for angular deviation penalty (default 0.8) |
| `--snap-w-size` | float | `0.0` | Snap scoring weight for object size reward (default 0.0) |
| `--snap-w-intersect` | float | `0.5` | Snap scoring bonus for ray-bbox intersection (default 0.5) |
| `--snap-w-temporal` | float | `0.3` | Snap scoring bonus for previous-frame target stickiness (default 0.3) |
| `--snap-gate-angle` | float | `60.0` | Hard angular cutoff in degrees: objects beyond this angle from the blended gaze+head direction are never snap candidates (default 60.0). |
| `--snap-head-blend` | float | `0.3` | Blend factor for angular scoring: 0=pure gaze direction, 1=pure head orientation (default 0.3). |
| `--snap-quality-thresh` | float | `0.8` | Maximum score to accept a snap match. Higher values are more permissive. Set lower to reject poor matches (default 0.8). |
| `--snap-tip-dist` | float | `-1.0` | Tip-snap distance threshold. -1 = use --snap-dist (default -1). |
| `--snap-tip-quality` | float | `-1.0` | Tip-snap quality threshold. -1 = use --snap-quality-thresh (default -1). |
| `--hit-conf-gate` | float | `0.0` | Minimum per-face gaze confidence for ray-object hit detection. 0.0 = disabled (default). |
| `--detect-extend` | float | `0.0` | Extend gaze-object detection N pixels past the visual ray/cone endpoint. 0 = detection matches visual exactly (default: the default). |
| `--detect-extend-scope` | `objects` / `phenomena` / `both` | `"objects"` | Scope for --detect-extend: 'objects' extends only ray-object hit detection, 'phenomena' extends only phenomena tracking (mutual gaze, social ref), 'both' extends both (default: objects). |
| `--gaze-cone` | float | `0.0` | Replaces standard gaze vectors with vision cones of a specified angle in degrees (disabled by default). |
| `--gaze-lock` | flag | off | Enable fixation lock-on (default: off). |
| `--dwell-frames` | int | `15` | Frames of sustained gaze required to trigger lock-on. |
| `--lock-dist` | int | `100` | Maximum pixel distance for lock-on to persist. |
| `--gaze-debug` | flag | off | Draw debug annotations for gaze processing. |
| `--snap-release-frames` | int | `5` | Frames of no-match before releasing the held snap target (default 5). |
| `--snap-engage-frames` | int | `0` | Frames of consistent match required before engaging snap for the first time. 0 = instant engage (default 0). |
| `--reid-grace-seconds` | float | `1.0` | Seconds a lost face track stays in the re-ID buffer (default 1.0). |
| `--forward-gaze-threshold` | float | `5.0` | Pitch/yaw angles below this (degrees) are treated as looking forward at the camera. Set to 0 to disable (default 5.0). |
| `--smooth-snap` | `off` / `objects` / `gaze_tips` / `all` | `"off"` | Smooth snap mode: smoothly interpolate the ray toward snap targets instead of jumping instantly. 'objects' = smooth object snaps only, 'gaze_tips' = smooth gaze-tip snaps only, 'all' = both (default: off). |
| `--smooth-snap-alpha` | float | `0.2` | EMA rate for smooth snap: lower = smoother/slower, higher = faster/more responsive (default: 0.20). |

### Depth Estimation

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--depth` | flag | off | Enable monocular depth estimation. |
| `--no-depth` | flag | off | Explicitly disable depth estimation. |
| `--depth-backend` | str | `"midas_small"` | Depth model backend (default: midas_small). |
| `--depth-input-size` | int | `384` | Depth model input resolution (default: 384). |
| `--depth-skip-frames` | int | `1` | Run depth every N detection cycles (default: 1). |
| `--depth-aware-scoring` | flag | off | Enable depth-weighted snap scoring. |
| `--depth-w-depth` | float | `0.4` | Depth match weight in snap scoring (default: 0.4). |
| `--depth-sample-radius` | int | `2` | Half-size of patch for depth sampling (default: 2). |

### Ray Forming (Gazelle blend)

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--rf-gazelle-model` | str | None | Path to a Gaze-LLE checkpoint (.pt) for Gazelle blend ray forming. Used alongside a pitch/yaw backend (MGaze, etc.) to periodically correct rays with Gaze-LLE heatmaps. |
| `--rf-gazelle-name` | `gazelle_dinov2_vitb14` / `gazelle_dinov2_vitb14_inout` / `gazelle_dinov2_vitl14` / `gazelle_dinov2_vitl14_inout` | `"gazelle_dinov2_vitb14"` | Gaze-LLE model variant for ray forming (default: gazelle_dinov2_vitb14). |
| `--rf-gazelle-interval` | int | None | (Legacy) alias for --min-call-gap. If both are given, --min-call-gap wins. Kept so old scripts/YAMLs keep working (default: unset -> min_call_gap default). |
| `--min-call-gap` | int | None | Minimum frames between Gaze-LLE inference calls. The scheduler also requires at least one participant to be fixating before firing (default: 30). |
| `--fixation-v-threshold` | float | `0.04` | Smoothed pitch/yaw velocity (rad/frame) at which a face is treated as 50% fixating. Lower = safer anchoring (default: 0.04). |
| `--fixation-d-threshold` | float | `0.15` | Windowed pitch/yaw dispersion (rad) at which a face is treated as 50% fixating (default: 0.15). |
| `--dir-min-cutoff` | float | `1.0` | Direction One Euro smoother floor cutoff (Hz). Lower = smoother at rest (default: 1.0). |
| `--dir-beta` | float | `0.5` | Direction One Euro responsiveness. Higher = snaps to fast motion (default: 0.5). |
| `--len-min-cutoff` | float | `1.0` | Length One Euro smoother floor cutoff (Hz) (default: 1.0). |
| `--len-beta` | float | `0.3` | Length One Euro responsiveness. Lower than direction by default so length holds steadier (default: 0.3). |
| `--len-hold-tau` | float | `5.0` | Seconds the Gaze-LLE-derived ray length persists after an accepted inference before decaying back to the pitch/yaw baseline. Direction reverts quickly on its own; raise this to hold ray reach longer between inferences (default: 5.0). |
| `--depth-ray-length` | flag | off | Use depth map to scale ray length based on scene geometry (default: off). |
| `--depth-length-min` | float | `0.5` | Ray length multiplier at depth=0 (nearest) (default: 0.5). |
| `--depth-length-max` | float | `3.0` | Ray length multiplier at depth=1 (farthest) (default: 3.0). |
| `--depth-belief-boost` | float | `0.0` | How much depth agreement boosts Gaze-LLE heatmap confidence in the belief update (default: 0.0). |

### Adas Gaze backend

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--adas-gaze-model` | str | None | Path to the Intel gaze-estimation-adas-0002 ONNX model. Activates the head-pose-normalized adas gaze backend (Apache-2.0, provenance-clean; requires the MediaPipe `face_landmarker.task` asset). |

### Gazelle backend

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--gazelle-model` | str | None | Path to a Gazelle checkpoint (.pt).  Activates the Gazelle backend. |
| `--gazelle-name` | `gazelle_dinov2_vitb14` / `gazelle_dinov2_vitb14_inout` / `gazelle_dinov2_vitl14` / `gazelle_dinov2_vitl14_inout` | `"gazelle_dinov2_vitb14"` | Gazelle model variant  (choices: gazelle_dinov2_vitb14, gazelle_dinov2_vitb14_inout, gazelle_dinov2_vitl14, gazelle_dinov2_vitl14_inout;  default: gazelle_dinov2_vitb14). |
| `--gazelle-inout-threshold` | float | `0.5` | In/out-of-view confidence threshold for *_inout model variants.  No effect on non-inout models  (default: 0.5). |
| `--gazelle-device` | `auto` / `cpu` / `cuda` / `mps` | `"auto"` | Compute device: auto, cpu, cuda, or mps.  'auto' selects CUDA > MPS > CPU  (default: auto). |
| `--gazelle-skip-frames` | int | `0` | Reuse the previous gaze result for N frames between inference runs.  0 = no skipping  (default: 0). |
| `--gazelle-fp16` | flag | off | Use half-precision (float16) inference on CUDA/MPS (ignored on CPU). |
| `--gazelle-compile` | flag | off | Use torch.compile() for the Gazelle model (PyTorch 2.0+ only). |

### MPIIFaceGaze backend

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--mpiifacegaze-model` | str | None | Path to the hysts MPIIFaceGaze resnet_simple checkpoint (.pth). Activates the head-pose-normalized MPIIFaceGaze backend (requires the MediaPipe `face_landmarker.task` asset). Weights are research-provenance -- trained on MPIIFaceGaze (CC BY-NC-SA); see THIRD_PARTY_LICENSES. |

### MGaze backend

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--mgaze-model` | str | (bundled ONNX weights) | Path to MGaze model weights (.onnx or .pt) |
| `--mgaze-arch` | `resnet18` / `resnet34` / `resnet50` / `mobilenetv2` / `mobileone_s0` / `mobileone_s1` / `mobileone_s2` / `mobileone_s3` / `mobileone_s4` | None | Architecture name (required for .pt models) |
| `--mgaze-dataset` | str | `"gaze360"` | Dataset config key (default: gaze360) |

### Iris-Refined Gaze

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--iris-refine` | flag | off | Enable iris-based gaze refinement (wraps active backend). |
| `--iris-refine-weight` | float | `0.3` | Blending weight for iris correction (default: 0.3). |
| `--iris-refine-upscale` | float | `2.0` | Upscale face crops before iris extraction (default: 2.0). |

### Phenomena

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--joint-attention` | flag | off | Enable joint-attention tracking. |
| `--ja-window` | int | `0` | Temporal consistency window (frames). 0 = disabled (default). |
| `--ja-window-thresh` | float | `0.7` | Fraction of window frames an object must be attended for JA confirmation. |
| `--ja-quorum` | float | `1.0` | Fraction of faces required for joint attention (default 1.0). |
| `--mutual-gaze` | flag | off | Enable mutual-gaze detection. |
| `--social-ref` | flag | off | Enable social-referencing detection. |
| `--social-ref-window` | int | `60` | Window size (frames) for social referencing. |
| `--gaze-follow` | flag | off | Enable gaze-following detection. |
| `--gaze-follow-lag` | int | `30` | Maximum lag (frames) for gaze-following alignment. |
| `--gaze-aversion` | flag | off | Enable gaze-aversion detection. |
| `--aversion-window` | int | `60` | Window size (frames) for gaze aversion. |
| `--aversion-conf` | float | `0.5` | Confidence threshold for gaze aversion. |
| `--scanpath` | flag | off | Enable scanpath recording. |
| `--scanpath-dwell` | int | `8` | Minimum fixation dwell (frames) for scanpath points. |
| `--gaze-leader` | flag | off | Enable gaze-leadership detection. |
| `--gaze-leader-tips` | flag | off | Also detect leadership via gaze-tip convergence (requires --gaze-tips). |
| `--gaze-leader-tip-lag` | int | `15` | Lookback frames for tip-arrival priority (default: 15). |
| `--attn-span` | flag | off | Track per-participant per-object average attention span (mean completed-glance duration). Most salient object shown in HUD. |
| `--all-phenomena` | flag | off | Enable all gaze-phenomena trackers at once. |

### Gaze Boost plugin

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--gaze-boost` | flag | off | Boost confidence of detections near gaze endpoints. |
| `--gaze-boost-factor` | float | `1.5` | Multiplicative confidence boost (default: 1.5). |
| `--gaze-boost-radius` | float | `100.0` | Pixel radius around gaze endpoints (default: 100). |
| `--gaze-boost-min-conf` | float | `0.1` | Minimum YOLO conf for sub-threshold candidates (default: 0.10). |
| `--gaze-boost-max-conf` | float | `0.95` | Cap on boosted confidence (default: 0.95). |
| `--gaze-boost-classes` | str[] | None | Only boost these class names (default: all non-person classes). |

### Eye Movement plugin

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--eye-movement` | flag | off | Enable eye movement classification. |
| `--em-source` | `gaze` / `iris` | `"gaze"` | Velocity source (default: gaze). |
| `--em-saccade-thresh` | float | `30.0` | Saccade threshold px/frame (default: 30.0). |
| `--em-fixation-thresh` | float | `10.0` | Fixation threshold px/frame (default: 10.0). |
| `--em-min-fixation` | int | `4` | Min fixation duration frames (default: 4). |
| `--em-velocity-window` | int | `3` | Median filter window (default: 3). |

### Novel Salience plugin

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--novel-salience` | flag | off | Enable novel-salience detection (rapid gaze-shift tracking). |
| `--ns-speed-thresh` | float | `40.0` | Gaze-endpoint speed threshold in pixels/frame to flag a novel-salience event (default: 40).  Lower = more sensitive. |
| `--ns-cooldown` | int | `20` | Minimum frames between consecutive novel-salience events for the same face track (default: 20). |
| `--ns-history` | int | `2` | Sliding-window depth for velocity smoothing (default: 2).  1 = instantaneous; 3 = heavier smoothing. |
| `--ns-flash` | int | `12` | Frames the visual saccade indicator persists after an event (default: 12). |

### Pupillometry plugin

| Flag | Type / choices | Default | Description |
|------|----------------|---------|-------------|
| `--pupillometry` | flag | off | Enable pupillometry tracking. |
| `--pupil-mode` | `rgb` / `ir` | `"rgb"` | Measurement mode (default: rgb). |
| `--pupil-baseline` | int | `90` | Baseline frames for calibration (default: 90). |
| `--pupil-upscale` | float | `2.0` | Upscale factor for RGB mode (default: 2.0). |
| `--pupil-ir-thresh` | int | `40` | IR threshold for dark-pupil segmentation (default: 40). |
| `--pupil-filter` | `kalman` / `ema` | `"kalman"` | Ratio smoothing filter (default: kalman). |
| `--pupil-ema-alpha` | float | `0.3` | EMA smoothing alpha -- only used when --pupil-filter is ema (default: 0.3). |
| `--pupil-kalman-process-noise` | float | `0.0001` | Kalman process noise -- controls how fast the filter adapts to ratio changes. Only used when --pupil-filter is kalman (default: 0.0001). |
| `--pupil-kalman-meas-noise` | float | `0.01` | Kalman measurement noise -- higher values smooth more aggressively. Only used when --pupil-filter is kalman (default: 0.01). |
| `--pupil-ear-thresh` | float | `0.21` | EAR threshold for blink detection (default: 0.21). |
| `--pupil-blink-frames` | int | `2` | Min consecutive low-EAR frames for blink (default: 2). |
| `--pupil-outlier-window` | int | `15` | Hampel outlier filter window size (default: 15). |
| `--pupil-per-eye` | flag | off | Enable per-eye (left/right) measurements. |
