# pipeline.yaml Schema

MindSight loads a declarative configuration file with `python MindSight.py --pipeline
path/to/pipeline.yaml`. The loader (`mindsight.config_compat.load_pipeline`) reads the
YAML, maps each recognised key onto an argparse destination, and merges it into the run
namespace. The GUI's **Import Pipeline YAML** and **Load Preset** actions use the same
loader. CLI flags always take precedence over YAML values.

Every key on this page corresponds to a CLI flag documented in the
[CLI Flags Reference](cli-flags.md); the YAML form is the declarative equivalent.

!!! warning "Unknown keys are silently ignored"
    The loader recognises a fixed set of key spellings (the tables below). Any key it does
    not recognise is dropped with no error and no warning. In particular, a few natural
    guesses do **not** work: `output.charts`, `output.conditions`, and `output.pid_map`
    have no YAML home; `gaze.smooth_snap`, `gaze.smooth_snap_alpha`, and
    `gaze.forward_gaze_threshold` are **not** gaze-section keys (set them through the
    `plugins:` pass-through instead); and mapping-style `phenomena: {joint_attention:
    true}` is ignored because the loader only reads the **list** form of `phenomena:`
    (see below). When a value seems not to take effect, check its spelling against these
    tables first.

!!! note "Two loaders"
    `--pipeline` and the GUI use `load_pipeline`, which merges into an argparse namespace
    and passes the **entire** `plugins:` section through to flag destinations. A second,
    schema-validated loader, `load_yaml` (returning a `PipelineConfig`), applies the same
    scalar/section keys, the phenomena list, and `aux_streams`, but only routes the
    ray-forming interval keys from `plugins:`. This page documents the runtime
    (`load_pipeline`) behaviour; the annotated example at the end is verified against both.

---

## Top-level

| YAML Key | Argparse dest | Type | Default |
|---|---|---|---|
| `source` | `source` | str | `"0"` |

`source` is normally omitted from a preset: it is a runtime input, and pinning it would
override the video the user selected.

## Detection Section

| YAML Key | Argparse dest | Type | Default |
|---|---|---|---|
| `detection.model` | `model` | str | `"yolov8n.pt"` |
| `detection.conf` | `conf` | float | `0.35` |
| `detection.classes` | `classes` | list[str] | `[]` |
| `detection.blacklist` | `blacklist` | list[str] | `[]` |
| `detection.detect_scale` | `detect_scale` | float | `1.0` |
| `detection.vp_file` | `vp_file` | str | None |
| `detection.vp_model` | `vp_model` | str | `"yoloe-26l-seg.pt"` |
| `detection.skip_frames` | `skip_frames` | int | `1` |
| `detection.obj_persistence` | `obj_persistence` | int | `0` |

`detection.classes` / `detection.blacklist` are class **names** (e.g. `[person, chair]`),
not COCO ids; they are resolved against the loaded model at build time. The
`merge_overlaps` family has no dedicated detection key -- set it through `plugins:`.

## Gaze Section

| YAML Key | Argparse dest | Type | Default |
|---|---|---|---|
| `gaze.ray_length` | `ray_length` | float | `1.0` |
| `gaze.adaptive_ray` | `adaptive_ray` | str | `"off"` |
| `gaze.snap_dist` | `snap_dist` | float | `150.0` |
| `gaze.snap_bbox_scale` | `snap_bbox_scale` | float | `0.0` |
| `gaze.snap_w_dist` | `snap_w_dist` | float | `1.0` |
| `gaze.snap_w_angle` | `snap_w_angle` | float | `0.8` |
| `gaze.snap_w_size` | `snap_w_size` | float | `0.0` |
| `gaze.snap_w_intersect` | `snap_w_intersect` | float | `0.5` |
| `gaze.snap_w_temporal` | `snap_w_temporal` | float | `0.3` |
| `gaze.snap_gate_angle` | `snap_gate_angle` | float | `60.0` |
| `gaze.snap_head_blend` | `snap_head_blend` | float | `0.3` |
| `gaze.snap_quality_thresh` | `snap_quality_thresh` | float | `0.8` |
| `gaze.snap_tip_dist` | `snap_tip_dist` | float | `-1.0` |
| `gaze.snap_tip_quality` | `snap_tip_quality` | float | `-1.0` |
| `gaze.conf_ray` | `conf_ray` | bool | `false` |
| `gaze.gaze_tips` | `gaze_tips` | bool | `false` |
| `gaze.tip_radius` | `tip_radius` | int | `80` |
| `gaze.gaze_cone` | `gaze_cone` | float | `0.0` |
| `gaze.gaze_lock` | `gaze_lock` | bool | `false` |
| `gaze.dwell_frames` | `dwell_frames` | int | `15` |
| `gaze.lock_dist` | `lock_dist` | int | `100` |
| `gaze.gaze_debug` | `gaze_debug` | bool | `false` |
| `gaze.snap_release_frames` | `snap_release_frames` | int | `5` |
| `gaze.snap_engage_frames` | `snap_engage_frames` | int | `0` |
| `gaze.reid_grace_seconds` | `reid_grace_seconds` | float | `1.0` |
| `gaze.hit_conf_gate` | `hit_conf_gate` | float | `0.0` |
| `gaze.detect_extend` | `detect_extend` | float | `0.0` |
| `gaze.detect_extend_scope` | `detect_extend_scope` | str | `"objects"` |

`gaze.adaptive_ray` is the `off` / `extend` / `snap` enum. (Legacy YAMLs that used a
boolean `adaptive_ray:` plus an `adaptive_snap:` companion are still accepted and mapped
onto the enum.) The old `gaze.snap_switch_frames` key no longer exists; the snap
hysteresis is now split into `snap_release_frames` (default 5) and `snap_engage_frames`
(default 0). `smooth_snap`, `smooth_snap_alpha`, and `forward_gaze_threshold` are set
through `plugins:`, not here.

## Output Section

| YAML Key | Argparse dest | Type | Default |
|---|---|---|---|
| `output.save_video` | `save` | str/bool | None |
| `output.log_csv` | `log` | str | None |
| `output.summary_csv` | `summary` | str | None |
| `output.heatmaps` | `heatmap` | str | None |
| `output.anonymize` | `anonymize` | str (`blur`/`black`) | None |
| `output.anonymize_padding` | `anonymize_padding` | float | `0.3` |

These six are the **only** recognised `output` keys. There is no YAML key for charts,
conditions, or the participant map: `--charts` is CLI-only, and conditions / participant
labels come from project metadata (`project.yaml`) or `participants:` (below). A boolean
`true` for `save_video` / `log_csv` / `summary_csv` / `heatmaps` selects the auto-named
output path.

## Participants Section

| YAML Key | Argparse dest | Type | Default |
|---|---|---|---|
| `participants.csv` | `participant_csv` | str | None |
| `participants.ids` | `participant_ids` | str | None |

`participants.ids` is a **positional** comma-separated label list (`"S70,S71,S72"`): the
first label maps to track 0, the second to track 1, and so on -- it is not a `id:name`
mapping. `participants.csv` points at a `participant_ids.csv` mapping video filenames to
labels. (These are runtime keys honoured by `load_pipeline`; the schema loader `load_yaml`
leaves participant resolution to build time.)

## Performance Section

| YAML Key | Argparse dest | Type | Default |
|---|---|---|---|
| `performance.fast` | `fast` | bool | `false` |
| `performance.skip_phenomena` | `skip_phenomena` | int | `0` |
| `performance.lite_overlay` | `lite_overlay` | bool | `false` |
| `performance.no_dashboard` | `no_dashboard` | bool | `false` |
| `performance.profile` | `profile` | bool | `false` |

## Depth Section

| YAML Key | Argparse dest | Type | Default |
|---|---|---|---|
| `depth.enabled` | `depth` | bool | `false` |
| `depth.backend` | `depth_backend` | str | `"midas_small"` |
| `depth.input_size` | `depth_input_size` | int | `384` |
| `depth.skip_frames` | `depth_skip_frames` | int | `1` |
| `depth.depth_aware_scoring` | `depth_aware_scoring` | bool | `false` |
| `depth.snap_w_depth` | `depth_w_depth` | float | `0.4` |
| `depth.gaze_sample_radius` | `depth_sample_radius` | int | `2` |

Monocular depth estimation. The ray-length-from-depth knobs (`--depth-ray-length`,
`--depth-length-min/max`, `--depth-belief-boost`) belong to ray forming and are set
through `plugins:`.

## Phenomena Section

`phenomena` is read as a **YAML list** of tracker toggles. Each entry is either a plain
string (enable with defaults) or a single-key mapping (enable with parameters):

```yaml
phenomena:
  - mutual_gaze                 # enable with defaults
  - scanpath
  - joint_attention:           # enable with parameters
      ja_window: 30
      ja_quorum: 0.8
  - social_referencing:
      window: 90
  - gaze_following:
      lag: 20
```

A `phenomena:` mapping (rather than a list) is ignored. Three joint-attention parameters
may **also** appear as top-level `phenomena.*` keys, independent of the list:
`phenomena.ja_window`, `phenomena.ja_window_thresh`, `phenomena.ja_quorum`.

### Toggle strings

| YAML string | Enables (dest) |
|---|---|
| `joint_attention` | `joint_attention` |
| `mutual_gaze` | `mutual_gaze` |
| `social_referencing` | `social_ref` |
| `gaze_following` | `gaze_follow` |
| `gaze_aversion` | `gaze_aversion` |
| `scanpath` | `scanpath` |
| `gaze_leadership` | `gaze_leader` |
| `attention_span` | `attn_span` |

### Per-tracker parameter keys

| Param key | Argparse dest |
|---|---|
| `ja_window` | `ja_window` |
| `ja_quorum` | `ja_quorum` |
| `ja_window_thresh` | `ja_window_thresh` |
| `window` | `social_ref_window` |
| `lag` | `gaze_follow_lag` |
| `aversion_window` | `aversion_window` |
| `aversion_conf` | `aversion_conf` |
| `dwell` | `scanpath_dwell` |

## Plugins Section (pass-through)

The `plugins:` section is a direct pass-through: every key is mapped to an argparse
destination with hyphens rewritten as underscores, and set on the namespace as-is. It
therefore accepts **any** flag's destination -- there is no per-key allow-list. This is how
backend and plugin flags that have no dedicated section are configured, and it is how the
shipped `configs/pipeline_known_good.yaml` preset sets roughly twenty core parameters,
for example:

```yaml
plugins:
  all_phenomena: true               # -> --all-phenomena
  mgaze_model: resnet50             # -> --mgaze-model (bare name resolves per install)
  mgaze_dataset: gaze360
  rf_gazelle_model: gazelle_hgnetv2_pico_inout_distill_1x3x640x640_1xNx4.onnx   # -> --rf-gazelle-model (Gaze-LLE blend; .onnx = ONNX engine, .pt = torch)
  rf_gazelle_name: gazelle_dinov2_vitb14      # torch variant (used when the model is a .pt)
  min_call_gap: 25
  forward_gaze_threshold: 13.0      # gaze knob with no gaze-section key
  smooth_snap: all
  smooth_snap_alpha: 0.9
  merge_overlaps: true              # detection knob with no detection-section key
  merge_overlap_strategy: dynamic
  merge_overlap_threshold: 0.55
```

Unknown `plugins:` keys become namespace attributes and are silently ignored if no backend
consumes them.

## Auxiliary Streams Section

`aux_streams` is a list of optional per-participant video feeds, frame-synchronised with the
main source and exposed to plugins in `FrameContext['aux_frames']`. Each entry requires
`source`, `stream_label`, and a non-empty `participants` list; an entry missing any of the
three is skipped.

```yaml
aux_streams:
  - source: /data/s70_eye.mp4
    video_type: eye_only          # wide_closeup | face_closeup | eye_only | custom
    stream_label: left_eye_cam
    participants: [S70]
    auto_detect_faces: false      # default true; runs face detection on wide/face streams
```

| Key | Type | Required | Default |
|---|---|---|---|
| `source` | str | yes | -- |
| `video_type` | enum | no | `custom` |
| `stream_label` | str | yes | -- |
| `participants` | list[str] | yes | -- |
| `auto_detect_faces` | bool | no | `true` |

`video_type` must be one of `wide_closeup`, `face_closeup`, `eye_only`, or `custom`; an
unrecognised value falls back to `custom` with a warning. Entries parse into
`AuxStreamConfig` instances.

---

## Fully Annotated Example

This example is verified by loading it through `mindsight.config_compat.load_yaml` and
`load_pipeline` (see the proof note below).

```yaml
# pipeline.yaml -- fully annotated MindSight pipeline configuration

source: /data/experiment_01.mp4      # runtime input; a preset normally omits this

detection:
  model: yolov8n.pt                  # YOLO weights (bare name resolves per install)
  conf: 0.30                         # detection confidence threshold
  classes: [person, chair]           # whitelist of class NAMES (not COCO ids)
  detect_scale: 1.0
  skip_frames: 2                     # detect every 2nd frame (-> tracker.skip_frames)
  obj_persistence: 3                 # dead-reckon boxes for 3 frames after a miss

gaze:
  ray_length: 1.3
  adaptive_ray: snap                 # off | extend | snap
  snap_dist: 180.0
  snap_w_intersect: 0.6
  gaze_tips: true
  tip_radius: 70
  gaze_cone: 5.0                     # YAML key is gaze_cone (-> gaze.gaze_cone_angle)
  gaze_lock: true                    # -> tracker.gaze_lock
  dwell_frames: 20                   # -> tracker.dwell_frames
  lock_dist: 120                     # -> tracker.lock_dist
  snap_release_frames: 5             # -> tracker.snap_release_frames
  snap_engage_frames: 0              # -> tracker.snap_engage_frames
  reid_grace_seconds: 4.5            # -> tracker.reid_grace_seconds
  detect_extend: 0.0
  detect_extend_scope: both

output:
  save_video: /results/annotated.mp4   # -> output.save
  log_csv: /results/frames.csv          # -> output.log_path
  summary_csv: /results/summary.csv     # -> output.summary_path
  heatmaps: /results/heatmaps           # -> output.heatmap_path
  anonymize: blur                       # blur | black
  anonymize_padding: 0.4

participants:
  ids: "S70,S71"                     # positional labels (first -> track 0)

performance:
  fast: false
  skip_phenomena: 0
  no_dashboard: true

depth:
  enabled: true
  backend: midas_small
  input_size: 384
  skip_frames: 1
  depth_aware_scoring: true
  snap_w_depth: 0.4
  gaze_sample_radius: 2

phenomena:
  - joint_attention:
      ja_window: 30
      ja_quorum: 0.8
      ja_window_thresh: 0.75
  - mutual_gaze
  - social_referencing:
      window: 90
  - gaze_following:
      lag: 20
  - gaze_aversion:
      aversion_window: 45
      aversion_conf: 0.6
  - scanpath:
      dwell: 10
  - gaze_leadership
  - attention_span

plugins:
  mgaze_model: resnet50
  mgaze_dataset: gaze360
  rf_gazelle_model: gazelle_hgnetv2_pico_inout_distill_1x3x640x640_1xNx4.onnx
  rf_gazelle_name: gazelle_dinov2_vitb14
  min_call_gap: 25
  forward_gaze_threshold: 13.0
  smooth_snap: all
  smooth_snap_alpha: 0.9
  merge_overlaps: true
  merge_overlap_strategy: dynamic
  merge_overlap_threshold: 0.55

aux_streams:
  - source: /data/s70_eye.mp4
    video_type: eye_only
    stream_label: left_eye_cam
    participants: [S70]
    auto_detect_faces: false
```

Loaded through the schema loader, the section keys, phenomena list, and `aux_streams` all
land as expected (e.g. `detection.conf` -> `0.30`, `detection.skip_frames` ->
`tracker.skip_frames` = `2`, `gaze.adaptive_ray` -> `"snap"`, `gaze.gaze_cone` ->
`gaze.gaze_cone_angle` = `5.0`, `output.save_video` -> `output.save`, `depth.enabled` ->
`True`, `joint_attention` with `ja_window=30`, `social_referencing`'s `window` ->
`social_ref_window=90`, `scanpath`'s `dwell` -> `scanpath_dwell=10`, and the `aux_streams`
entry as `video_type=eye_only`, `auto_detect_faces=false`). The full `plugins:` block
(`merge_overlaps`, `mgaze_model`, `forward_gaze_threshold`, `smooth_snap`, ...) is applied
by the runtime `load_pipeline` path that `--pipeline` and the GUI use.

## Precedence Rules

1. A CLI flag always overrides the corresponding YAML value.
2. On the CLI, YAML is applied to every parameter the user did **not** type on the command
   line (the parser records the exact flags typed). GUI / synthetic namespaces fall back to
   a "default-like" heuristic: a YAML value overwrites only an attribute that is missing or
   still at a default-like value (None, False, 0, empty string/list).
3. The `phenomena` list, `plugins` pass-through, and `aux_streams` follow the same
   precedence rule.
