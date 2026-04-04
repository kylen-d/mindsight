# pipeline.yaml Schema

MindSight can load a declarative YAML configuration file via `--pipeline path/to/pipeline.yaml`. The loader (`pipeline_loader.py`) reads the YAML, maps keys to argparse attributes, and populates the namespace. CLI flags always take precedence over YAML values.

---

## YAML Key to Argparse Mapping

### Top-level

| YAML Key Path | Argparse Attribute | Type | Default |
|---|---|---|---|
| `source` | `source` | str | `"0"` |

### Detection Section

| YAML Key Path | Argparse Attribute | Type | Default |
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

### Gaze Section

| YAML Key Path | Argparse Attribute | Type | Default |
|---|---|---|---|
| `gaze.ray_length` | `ray_length` | float | `1.0` |
| `gaze.adaptive_ray` | `adaptive_ray` | str | `"off"` |
| `gaze.snap_dist` | `snap_dist` | float | `150.0` |
| `gaze.snap_bbox_scale` | `snap_bbox_scale` | float | `0.0` |
| `gaze.snap_w_dist` | `snap_w_dist` | float | `1.0` |
| `gaze.snap_w_size` | `snap_w_size` | float | `0.0` |
| `gaze.snap_w_intersect` | `snap_w_intersect` | float | `0.5` |
| `gaze.conf_ray` | `conf_ray` | bool | `false` |
| `gaze.gaze_tips` | `gaze_tips` | bool | `false` |
| `gaze.tip_radius` | `tip_radius` | int | `80` |
| `gaze.gaze_cone` | `gaze_cone` | float | `0.0` |
| `gaze.gaze_lock` | `gaze_lock` | bool | `false` |
| `gaze.dwell_frames` | `dwell_frames` | int | `15` |
| `gaze.lock_dist` | `lock_dist` | int | `100` |
| `gaze.gaze_debug` | `gaze_debug` | bool | `false` |
| `gaze.snap_switch_frames` | `snap_switch_frames` | int | `8` |
| `gaze.reid_grace_seconds` | `reid_grace_seconds` | float | `1.0` |
| `gaze.hit_conf_gate` | `hit_conf_gate` | float | `0.0` |
| `gaze.detect_extend` | `detect_extend` | float | `0.0` |
| `gaze.detect_extend_scope` | `detect_extend_scope` | str | `"objects"` |

### Output Section

| YAML Key Path | Argparse Attribute | Type | Default |
|---|---|---|---|
| `output.save_video` | `save` | str/bool | None |
| `output.log_csv` | `log` | str | None |
| `output.summary_csv` | `summary` | str | None |
| `output.heatmaps` | `heatmap` | str | None |
| `output.charts` | `charts` | str/bool | None |
| `output.anonymize` | `anonymize` | str | None |
| `output.anonymize_padding` | `anonymize_padding` | float | `0.3` |
| `output.conditions` | `conditions` | str | None |
| `output.pid_map` | `pid_map` | dict | None |

### Participants Section

| YAML Key Path | Argparse Attribute | Type | Default |
|---|---|---|---|
| `participants.csv` | `participant_csv` | str | None |
| `participants.ids` | `participant_ids` | str | None |

### Performance Section

| YAML Key Path | Argparse Attribute | Type | Default |
|---|---|---|---|
| `performance.fast` | `fast` | bool | `false` |
| `performance.skip_phenomena` | `skip_phenomena` | int | `0` |
| `performance.lite_overlay` | `lite_overlay` | bool | `false` |
| `performance.no_dashboard` | `no_dashboard` | bool | `false` |
| `performance.profile` | `profile` | bool | `false` |

### Phenomena Top-level Keys

| YAML Key Path | Argparse Attribute | Type | Default |
|---|---|---|---|
| `phenomena.ja_window` | `ja_window` | int | `0` |
| `phenomena.ja_window_thresh` | `ja_window_thresh` | float | `0.70` |
| `phenomena.ja_quorum` | `ja_quorum` | float | `1.0` |

---

## Phenomena List Format

The `phenomena` key can also be a YAML list of tracker toggles. Each entry is either a plain string (to enable with defaults) or a single-key dict (to enable with custom parameters):

```yaml
phenomena:
  # Simple toggle — enables with default parameters
  - mutual_gaze
  - scanpath

  # Toggle with parameters
  - joint_attention:
      ja_window: 30
      ja_quorum: 0.8

  - gaze_following:
      lag: 20

  - social_referencing:
      window: 90
```

### Toggle Strings

| YAML String | Argparse Attribute |
|---|---|
| `joint_attention` | `joint_attention` |
| `mutual_gaze` | `mutual_gaze` |
| `social_referencing` | `social_ref` |
| `gaze_following` | `gaze_follow` |
| `gaze_aversion` | `gaze_aversion` |
| `scanpath` | `scanpath` |
| `gaze_leadership` | `gaze_leader` |
| `attention_span` | `attn_span` |

### Per-Tracker Parameter Keys

| Param Key | Argparse Attribute |
|---|---|
| `ja_window` | `ja_window` |
| `ja_quorum` | `ja_quorum` |
| `ja_window_thresh` | `ja_window_thresh` |
| `window` | `social_ref_window` |
| `lag` | `gaze_follow_lag` |
| `aversion_window` | `aversion_window` |
| `aversion_conf` | `aversion_conf` |
| `dwell` | `scanpath_dwell` |

---

## Plugins Section

The `plugins` section is a pass-through: keys are mapped directly to argparse attributes with hyphens converted to underscores. This allows any plugin flag to be set without a hardcoded mapping.

```yaml
plugins:
  gazelle-model: /path/to/gazelle.pt
  gazelle-name: gazelle_dinov2_vitb14
  l2cs-model: /path/to/l2cs.pkl
```

The above sets `ns.gazelle_model`, `ns.gazelle_name`, and `ns.l2cs_model`.

---

## Auxiliary Streams Section

The `aux_streams` section defines optional per-participant video feeds that are frame-synchronised with the main source. Each entry requires three keys:

```yaml
aux_streams:
  - pid: "S70"
    stream_type: eye_camera
    source: /data/s70_eye.mp4
  - pid: "S71"
    stream_type: first_person_view
    source: /data/s71_fpv.mp4
```

These are parsed into `AuxStreamConfig` dataclass instances and made available in `FrameContext['aux_frames']`.

---

## Fully Annotated Example

```yaml
# pipeline.yaml — MindSight pipeline configuration

source: /data/experiment_01.mp4

detection:
  model: yolov8s.pt           # Use the small YOLO model
  conf: 0.40                   # Slightly higher confidence threshold
  classes: [person, chair]     # Only detect people and chairs
  skip_frames: 2               # Detect every 2nd frame for speed
  obj_persistence: 3           # Keep detections alive for 3 frames

gaze:
  ray_length: 1.2
  adaptive_ray: snap           # Enable snap mode
  snap_dist: 200.0
  snap_w_intersect: 0.8
  gaze_lock: true
  dwell_frames: 20
  lock_dist: 120
  detect_extend: 50.0
  detect_extend_scope: both

output:
  save_video: /results/annotated.mp4
  log_csv: /results/frames.csv
  summary_csv: /results/summary.csv
  heatmaps: /results/heatmap.png
  charts: /results/charts/
  anonymize: blur
  anonymize_padding: 0.4
  conditions: "Group A|Emotional"
  pid_map:
    1: "Alice"
    2: "Bob"

participants:
  ids: "1:Alice,2:Bob"

performance:
  fast: true
  skip_phenomena: 2
  lite_overlay: true

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
  - scanpath:
      dwell: 10
  - gaze_leadership
  - attention_span
  - gaze_aversion:
      aversion_window: 45
      aversion_conf: 0.6

plugins:
  gazelle-model: /models/gazelle.pt
  gazelle-name: gazelle_dinov2_vitb14

aux_streams:
  - pid: "S01"
    stream_type: eye_camera
    source: /data/s01_eye.mp4
```

## Precedence Rules

1. CLI flags always override YAML values.
2. A YAML value is only applied if the corresponding namespace attribute is at its "default" (None, False, 0, empty string, or empty list).
3. The `plugins` section and `aux_streams` section follow the same precedence rule.
