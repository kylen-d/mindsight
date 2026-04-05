# Pipeline Configuration (YAML)

## Overview

MindSight supports declarative pipeline configuration through YAML files as an alternative to passing CLI flags. Supply a configuration file with the `--pipeline` flag:

```bash
python MindSight.py --pipeline my_pipeline.yaml
```

This is especially useful for reproducible experiments, project mode, and sharing configurations across team members.

## How It Works

On startup, `ms/pipeline_loader.py` reads the YAML file, flattens any nested keys into dot-separated paths, and maps each key to its corresponding `argparse` attribute using the internal `_YAML_MAP` dictionary. The resulting values are applied to the argument namespace before the pipeline begins.

CLI flags always take precedence over YAML values, so you can use a YAML file as a baseline and override individual settings from the command line.

## Precedence

The resolution order for every setting is:

1. **CLI flags** -- highest priority. Anything explicitly passed on the command line wins.
2. **YAML values** -- applied if the CLI flag was not provided.
3. **argparse defaults** -- used when neither CLI nor YAML specifies a value.

MindSight uses an `_is_default()` heuristic to decide whether a CLI flag was explicitly set: values of `None`, `False`, `0`, and empty strings/lists are treated as "default" (i.e., not explicitly provided), so the YAML value will apply instead.

## YAML Sections

A pipeline YAML file is organized into the following top-level sections:

### `detection`

Object detection settings.

| Key | Description | Example |
|-----|-------------|---------|
| `model` | Model name or path | `yoloe-11l-seg` |
| `conf` | Confidence threshold | `0.25` |
| `classes` | List of COCO class IDs to detect | `[0]` |
| `device` | Inference device | `mps` |
| `imgsz` | Input image size | `640` |
| `tracker` | Tracker config file | `bytetrack.yaml` |

### `gaze`

Gaze estimation parameters.

| Key | Description | Example |
|-----|-------------|---------|
| `ray_length` | Default gaze ray length in pixels | `500` |
| `adaptive_ray` | Enable distance-adaptive ray length | `true` |
| `backend` | Gaze estimation backend | `gazelle` |

### `output`

Output and logging options.

| Key | Description | Example |
|-----|-------------|---------|
| `save_video` | Write annotated video to disk | `true` |
| `log_csv` | Write per-frame CSV data | `true` |
| `heatmap` | Generate gaze heatmap image | `true` |
| `output_dir` | Output directory path | `./results` |

### `phenomena`

A list of phenomena trackers to enable, optionally with per-tracker parameters (see format below).

### `plugins`

Pass-through configuration for loaded plugins. Each key is a plugin name; its value is a dict of plugin-specific settings.

### `participants`

Participant metadata such as IDs and labels used for multi-person tracking.

### `performance`

Performance tuning options such as frame skip, resize factor, and threading.

### `aux_streams`

Auxiliary video streams for multi-camera setups.

## Phenomena Format

Phenomena entries can be simple strings or dicts with parameters:

```yaml
phenomena:
  # Enable with default parameters
  - mutual_gaze
  - scanpath

  # Enable with custom parameters
  - joint_attention:
      ja_window: 10
      ja_threshold: 0.5

  - gaze_aversion:
      aversion_threshold: 30
```

When a string is provided, the tracker is enabled with its default settings. When a dict is provided, the key is the tracker name and the nested dict supplies parameter overrides.

## Annotated Example

```yaml
# pipeline.yaml -- MindSight pipeline configuration

detection:
  model: yoloe-11l-seg          # YOLOE model variant
  conf: 0.25                    # Detection confidence threshold
  classes: [0]                  # COCO class 0 = person
  device: mps                   # Apple Silicon GPU
  imgsz: 640                    # Input resolution
  tracker: bytetrack.yaml       # Multi-object tracker

gaze:
  backend: gazelle              # Gaze estimation backend
  ray_length: 500               # Gaze ray length (pixels)
  adaptive_ray: true            # Scale ray by face size

output:
  save_video: true              # Write annotated output video
  log_csv: true                 # Per-frame CSV log
  heatmap: true                 # Aggregate gaze heatmap
  output_dir: ./results         # Output directory

phenomena:
  - mutual_gaze                 # Detect mutual gaze events
  - joint_attention:            # Detect joint attention
      ja_window: 10
  - gaze_following              # Detect gaze following
  - scanpath                    # Scanpath analysis

plugins:
  my_custom_plugin:
    threshold: 0.8
    mode: fast

participants:
  - id: 1
    label: "Participant A"
  - id: 2
    label: "Participant B"

performance:
  frame_skip: 0                 # Process every frame
  resize: 1.0                   # No resize
```

## Reference

For the complete mapping between YAML keys and argparse attributes, see [Pipeline YAML Schema](../reference/pipeline-yaml-schema.md).
