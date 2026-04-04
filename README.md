# MindSight — Unified Eye-Gaze Intersection Tracker for Behavioural Neuroscience Research

> **Beta Release** — This is a pre-release version (v0.2.0-beta). APIs and features may change. Bug reports and feedback are welcome via [GitHub Issues](https://github.com/kylen-d/mindsight/issues).

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Gaze estimation combined with YOLO object detection for the studying of various gaze and attention-based psychological phenomena. MindSight determines what participants are looking at in real time, and provides a framework to use this information to study a wide-range of gaze-based behaviour, such as Joint-Attention.
---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [MindSight.py — CLI Usage](#mindsightpy--cli-usage)
- [MindSight_GUI.py — GUI Usage](#mindsight_guipy--gui-usage)
- [Visual Prompt (VP) Files](#visual-prompt-vp-files)
- [Output & Logging](#output--logging)
- [Gaze Backends](#gaze-backends)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [License](#license)
- [Contributing](#contributing)

---

## Features

### Core Functionality

- **Multi-participant gaze tracking** — independently tracks each detected face, colour-coded per person
- **Built-in gaze post-processing features** — gaze smoothing, extending rays to nearby objects, adaptive ray-object snapping, fixation lock-on, etc.
- **Gaze ray–bbox intersection** — determines which detected objects each participant is looking at
- **Gaze convergence detection** — detects when multiple gaze ray tips cluster near the same point
- **Visual Object Detection Prompting** — makes use of YOLOE to allow for visual object detection prompts
- **PyQt6 GUI** — full graphical front-end with live preview and a drag-and-drop Visual Prompt Builder

### Built-in Tracking of Various Attention-based Phenomena

- **Joint attention detection** — flags when all tracked participants share a gaze target, with optional temporal confirmation windowing
- **Mutual gaze detection** — detects when two participants look directly at each other
- **Social referencing** — identifies when a participant looks at a person and then shifts gaze to an object
- **Gaze following** — tracks when one participant follows another's gaze to the same object
- **Gaze aversion** — detects sustained avoidance of specific objects or regions
- **Scanpath analysis** — records fixation sequences and dwell patterns across objects
- **Gaze leadership** — identifies which participant's gaze shifts are followed by others
- **Attention span** — tracks per-participant per-object mean completed-glance duration

### Highly Extensible

- **Gaze Backend Support** — supports and includes MGaze out of the box, with ONNX, PyTorch, and Gazelle backends, and allows custom gaze estimation backends through plugin support
- **Custom Phenomena Plugins** — supports user-written plugins to detect custom phenomena alongside the default pack
- **Custom Data Plugins** — supports user-written plugins for data output in addition to the video, CSV, and heatmap features included by default

### Designed for Research

- **Built-in CSV event logging** — per-frame hit events and a customizable post-run summary with joint attention %, cosine similarity averages, and object look-time statistics
- **Video Dashboard** — graphically overlays and organizes gaze-based phenomena detection alongside input videos
- **Heatmap generation** — per-participant gaze heatmap accumulation and export
- **Project-based Workflow** — user-defined pipelines to easily customize functionality and behaviour for a specific task, input stacks, and organized data output

---

## Architecture

```
Camera / Video / Image
        │
        ▼
  YOLO / YOLOE ──► object bounding boxes
        │
        ▼
  RetinaFace ──────► face bounding boxes
        │
        ▼
  Gaze Estimator ──► pitch + yaw per face  (ONNX / PyTorch / Gazelle)
        │
        ▼
  Ray–BBox Intersection ──► hit list  (face_idx, object_idx)
        │
        ├──► Joint Attention
        ├──► Gaze Convergence
        ├──► Cosine Similarity
        └──► Lock-on / Smooth / Snap
```

---

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10+ recommended (tested on 3.14) |
| CUDA / CoreML | Optional — CPU works, GPU accelerates |
| Webcam | Required for live mode |

### Python packages

```
opencv-python
numpy
torch
torchvision
onnxruntime          # or onnxruntime-gpu for CUDA
ultralytics          # YOLO / YOLOE
uniface              # RetinaFace face detector
PyQt6                # GUI only
```

> **Note:** The `gaze-estimation` submodule ships its own virtual environment (`gaze-est/`). You do **not** need to activate it separately — MindSight inserts the `gaze-estimation/` directory into `sys.path` automatically at runtime.

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd MindSight
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install opencv-python numpy torch torchvision
pip install onnxruntime           # CPU
# pip install onnxruntime-gpu     # NVIDIA GPU
pip install ultralytics
pip install uniface
pip install PyQt6                 # GUI only
```

**macOS (Apple Silicon) — CoreML acceleration:**

```bash
pip install onnxruntime-silicon   # or onnxruntime with CoreML support
```

### 4. Download gaze model weights

The default gaze model is `gaze-estimation/weights/mobileone_s0_gaze.onnx`. Other ONNX and PyTorch weights (`resnet18`, `resnet34`, `resnet50`) are already included in the `gaze-estimation/weights/` directory.

To use the Gazelle backend, download a checkpoint separately and pass it via `--gazelle-model`.

### 5. YOLO weights

YOLO model weights (e.g. `yolov8n.pt`) are downloaded automatically by Ultralytics on first use. YOLOE weights (e.g. `yoloe-26l-seg.pt`) are also auto-downloaded.

---

## Project Structure

```
MindSight/
├── MindSight.py                  # Orchestrator + CLI entry point
├── pipeline_config.py            # Config dataclasses + FrameContext
├── pipeline_loader.py            # YAML pipeline config loader
├── project_runner.py             # Project-based batch processing
├── constants.py                  # Shared constants (colours, thresholds)
│
├── ObjectDetection/
│   ├── Detection_Pipeline.py     # YOLO detection step (ctx-based)
│   ├── Object_Detection.py       # YOLO wrapper, ObjectPersistenceCache
│   ├── detection.py              # Detection utilities
│   └── model_factory.py          # YOLO and RetinaFace factory
│
├── GazeTracking/
│   ├── Gaze_Pipeline.py          # Gaze step coordinator (ctx-based)
│   ├── Gaze_Processing.py        # GazeSmootherReID, snap, lock-on
│   └── gaze_factory.py           # Gaze engine factory
│
├── Phenomena/
│   ├── Phenomena_Pipeline.py     # Phenomena step (ctx-based unified loop)
│   ├── phenomena_config.py       # PhenomenaConfig dataclass
│   ├── helpers.py                # joint_attention, gaze_convergence
│   └── Default/                  # Built-in phenomena pack
│       ├── joint_attention.py    # Temporal JA confirmation filter
│       ├── mutual_gaze.py
│       ├── social_referencing.py
│       ├── gaze_following.py
│       ├── gaze_aversion.py
│       ├── scanpath.py
│       ├── gaze_leadership.py
│       └── attention_span.py
│
├── DataCollection/
│   ├── Data_Pipeline.py          # CSV logging, heatmap accumulation
│   ├── Dashboard_Output.py       # Frame overlay + dashboard compositor
│   ├── CSV_Output.py             # Summary CSV writer
│   └── Heatmap_Output.py         # Per-participant heatmap generation
│
├── Plugins/
│   ├── __init__.py               # Base classes + registries
│   ├── GazeTracking/             # Gaze backend plugins
│   │   ├── MGaze/                # Per-face estimation (default)
│   │   └── Gazelle/              # Scene-level DINOv2
│   ├── ObjectDetection/          # Detection plugins
│   ├── Phenomena/                # Community phenomena plugins
│   │   └── NovelSalience/        # Saccade detection
│   ├── DataCollection/           # Custom data output plugins
│   └── TEMPLATE/                 # Skeleton plugin for developers
│
├── Projects/
│   └── ExampleProject/           # Example project layout
│       ├── Inputs/Videos/
│       ├── Inputs/Prompts/
│       ├── Outputs/
│       └── Pipeline/pipeline.yaml
│
├── utils/
│   └── geometry.py               # Ray geometry, pitch/yaw, bbox ops
│
├── tests/                        # pytest test suite
│   ├── test_geometry.py
│   ├── test_frame_context.py
│   ├── test_pipeline_loader.py
│   └── test_phenomena_trackers.py
│
└── Outputs/                      # Default output directory
    ├── CSV Files/
    ├── Video/
    └── heatmaps/
```

---

## MindSight.py — CLI Usage

```bash
python MindSight.py [OPTIONS]
```

### Quick examples

```bash
# Webcam with default settings
python MindSight.py --source 0

# Video file, save output, log events to CSV
python MindSight.py --source video.mp4 --save --log events.csv

# Single image with gaze tip convergence markers
python MindSight.py --source image.jpg --gaze-tips

# Use Gazelle gaze backend
python MindSight.py --source 0 --gazelle-model ckpt.pt

# Use YOLOE Visual Prompt instead of text-class YOLO
python MindSight.py --source 0 --vp-file my_prompts.vp.json --vp-model yoloe-26l-seg.pt

# Export post-run summary CSV
python MindSight.py --source video.mp4 --summary results.csv
```

### Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Input: `0` = webcam, integer = camera index, path to video/image |
| `--model` | `yolov8n.pt` | YOLO model weights |
| `--mgaze-model` | `gaze-estimation/weights/mobileone_s0_gaze.onnx` | MGaze: ONNX or `.pt` gaze weights |
| `--mgaze-arch` | `None` | MGaze: Required for `.pt` models: `resnet18`, `resnet34`, `resnet50`, `mobilenetv2`, `mobileone_s0`–`s4` |
| `--mgaze-dataset` | `gaze360` | MGaze: Dataset config used for `.pt` models |
| `--l2cs-model` | `None` | L2CS-Net: Path to `.pkl` or `.onnx` weights |
| `--l2cs-arch` | `ResNet50` | L2CS-Net: Architecture (`ResNet18`–`ResNet152`) |
| `--l2cs-dataset` | `gaze360` | L2CS-Net: Dataset config key |
| `--unigaze-model` | `None` | UniGaze: Model variant (requires `pip install unigaze timm==0.3.2`) |
| `--conf` | `0.35` | YOLO detection confidence threshold |
| `--classes` | `None` | Filter YOLO to specific class names, e.g. `--classes person knife` |
| `--blacklist` | `[]` | Suppress specific YOLO classes, e.g. `--blacklist chair` |
| `--save` | `False` | Write annotated output video/image to disk |
| `--log` | `None` | Path to per-frame event CSV |
| `--summary` | `None` | Path to post-run summary CSV |
| `--ray-length` | `1.0` | Gaze ray length multiplier (× face width) |
| `--conf-ray` | `False` | Scale ray length by gaze confidence |
| `--adaptive-ray` | `False` | Snap ray tips to nearby object centres |
| `--snap-dist` | `150` | Snap radius in pixels for adaptive ray |
| `--gaze-tips` | `False` | Enable gaze-tip convergence detection |
| `--tip-radius` | `80` | Radius (px) for convergence clustering |
| `--no-lock` | `False` | Disable fixation lock-on |
| `--dwell-frames` | `15` | Frames of sustained gaze before lock activates |
| `--lock-dist` | `100` | Pixel radius for lock-on detection |
| `--gaze-debug` | `False` | Show pitch/yaw values next to each face |
| `--skip-frames` | `1` | Run detection every N frames (1 = every frame) |
| `--detect-scale` | `1.0` | Scale factor for the detection pass (< 1 = faster) |
| `--gazelle-model` | `None` | Path to Gazelle checkpoint; switches to Gazelle backend |
| `--gazelle-name` | `gazelle_dinov2_vitb14` | Gazelle model variant |
| `--gazelle-inout-threshold` | `0.5` | Gazelle in/out-of-frame confidence threshold |
| `--vp-file` | `None` | Visual Prompt file (`.vp.json`); switches to YOLOE detector |
| `--vp-model` | `yoloe-26l-seg.pt` | YOLOE model weights for VP mode |

### Pipeline configuration files

Instead of passing many CLI flags, you can define a reusable YAML pipeline config:

```yaml
# my_pipeline.yaml
detection:
  model: "yoloe-26l-seg.pt"
  conf: 0.4
gaze:
  ray_length: 1.5
  adaptive_ray: true
phenomena:
  - joint_attention: { ja_window: 10 }
  - mutual_gaze: {}
output:
  save_video: true
  summary_csv: true
```

```bash
python MindSight.py --pipeline my_pipeline.yaml --source video.mp4
```

CLI flags always override YAML values.

### Project mode

A Project is a directory with a standard layout for batch-processing multiple videos:

```
MyProject/
├── Inputs/
│   ├── Videos/         # Drop video files here
│   └── Prompts/        # VP files for this project
├── Outputs/            # Auto-populated with per-video results
│   ├── CSV Files/
│   └── Videos/
└── Pipeline/
    └── pipeline.yaml   # Project-specific pipeline config
```

```bash
python MindSight.py --project Projects/MyProject/
```

This loads the project's `pipeline.yaml`, processes every video in `Inputs/Videos/`, and writes per-video outputs to `Outputs/`.

### Plugin development

MindSight supports four plugin types: **Gaze**, **Object Detection**, **Phenomena**, and **Data Collection**.

To create a new phenomena plugin:

1. Copy `Plugins/TEMPLATE/` to `Plugins/Phenomena/YourPlugin/`
2. Rename and edit `my_plugin.py` — subclass `PhenomenaPlugin`, implement `update(**kwargs)`
3. Set a unique `name` class attribute and expose `PLUGIN_CLASS = YourClass`
4. Add `add_arguments(cls, parser)` / `from_args(cls, args)` for CLI activation

The plugin is auto-discovered on startup. See `Plugins/Phenomena/NovelSalience/` for a full reference implementation.

---

## MindSight_GUI.py — GUI Usage

```bash
python MindSight_GUI.py
```

The GUI has two tabs:

### Tab 1 — Gaze Tracker

A graphical front-end for all `MindSight.py` functionality.

**Sections:**

- **Source** — camera index, video file, or image file
- **Detection mode** — YOLO (text classes) or YOLOE Visual Prompt
- **Gaze backend** — ONNX, PyTorch, or Gazelle
- **Gaze parameters** — ray length, confidence threshold, adaptive ray, gaze tips, lock-on dwell, etc.
- **Output** — optional video save and CSV log paths
- **Live preview** — annotated frames displayed inside the GUI window
- **Log console** — status messages from the background worker thread

**Controls:**

- Click **Start** to begin tracking; **Stop** to halt.
- The live frame display updates in real time; resize the window as needed.

### Tab 2 — VP Builder

Drag-and-drop tool for creating and testing `.vp.json` Visual Prompt files.

**Workflow:**

1. **Add reference images** — click *Add Images* and select one or more images that contain the objects you want to detect.
2. **Draw bounding boxes** — click and drag on the image canvas to draw boxes around objects.
3. **Assign classes** — each box is assigned a class from the class list on the left. Add new classes with *Add Class*.
4. **Save prompt** — click *Save VP File* to write the `.vp.json`.
5. **Test inference** — select a YOLOE model and a folder of test images, then click *Run Inference* to preview detections with the saved prompt.

---

## Visual Prompt (VP) Files

VP files encode reference images and bounding-box annotations used by YOLOE for image-based (visual) class prompting. They are JSON with the extension `.vp.json`.

**Format:**

```json
{
  "version": 1,
  "classes": [
    {"id": 0, "name": "knife"},
    {"id": 1, "name": "plate"}
  ],
  "references": [
    {
      "image": "/absolute/path/to/reference.jpg",
      "annotations": [
        {"cls_id": 0, "bbox": [x1, y1, x2, y2]},
        {"cls_id": 1, "bbox": [x1, y1, x2, y2]}
      ]
    }
  ]
}
```

- `classes` — sequential IDs starting from `0`.
- `references` — list of reference images with annotated boxes.
- The **first** reference image is used to initialise YOLOE class embeddings. Additional reference images are currently reserved for future use.
- Class IDs must be contiguous and start at `0`.

---

## Output & Logging

### Per-frame event CSV (`--log events.csv`)

One row per gaze–object hit:

```
frame, face_idx, object, object_conf, bbox_x1, bbox_y1, bbox_x2, bbox_y2, joint_attention
```

### Post-run summary CSV (`--summary summary.csv`)

Three sections written to the same file:

| Section | Columns | Description |
|---|---|---|
| `joint_attention` | `category, participant, object, frames_active, total_frames, value_pct` | % of frames with shared gaze target |
| `cosine_similarity` | same | Per-pair and overall mean cosine similarity of gaze directions |
| `object_look_time` | same | Per-(participant, object-class) frame count and % |

### Saved video

When `--save` is passed, an annotated `.mp4` is written alongside the source, or to `gaze_output.mp4` for live webcam sessions.

---

## Gaze Backends

| Backend | Trigger | Notes |
|---|---|---|
| **MGaze ONNX** (default) | `--mgaze-model` with `.onnx` path | Fastest; uses CoreML on Apple Silicon, CUDA on NVIDIA, CPU otherwise |
| **MGaze PyTorch** | `--mgaze-model` with `.pt` + `--mgaze-arch` | Requires `--mgaze-arch` to identify the architecture |
| **L2CS-Net** | `--l2cs-model <weights.pkl>` | ResNet50 with dual classification heads; ~3x more accurate than MGaze on MPIIGaze (3.92 vs ~11 deg MAE) |
| **UniGaze** (optional) | `--unigaze-model <variant>` | ViT + MAE pre-training; best cross-dataset accuracy (~9.4 deg Gaze360). Requires `pip install unigaze timm==0.3.2` (non-commercial license) |
| **Gazelle** | `--gazelle-model <ckpt.pt>` | Scene-level DINOv2 model; processes all faces in a single forward pass; outputs a gaze heatmap rather than pitch/yaw |

**MGaze architectures** (`--mgaze-arch`):

`resnet18`, `resnet34`, `resnet50`, `mobilenetv2`, `mobileone_s0`, `mobileone_s1`, `mobileone_s2`, `mobileone_s3`, `mobileone_s4`

**L2CS-Net architectures** (`--l2cs-arch`):

`ResNet18`, `ResNet34`, `ResNet50` (default), `ResNet101`, `ResNet152`

**UniGaze model variants** (`--unigaze-model`):

`unigaze_b16_joint` (ViT-Base), `unigaze_l16_joint` (ViT-Large), `unigaze_h14_joint` (ViT-Huge, best accuracy)

**Gazelle model variants** (`--gazelle-name`):

`gazelle_dinov2_vitb14`, `gazelle_dinov2_vitl14`, `gazelle_dinov2_vitb14_inout`, `gazelle_dinov2_vitl14_inout`

The `_inout` variants add an in-frame / out-of-frame confidence score that modulates the gaze heatmap peak.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` | Quit (CLI video/webcam mode) |
| Any key | Close (CLI image mode) |

---

## On-screen Overlay Legend

| Visual element | Meaning |
|---|---|
| Coloured arrow | Gaze ray for each tracked person (P0, P1, …) |
| Thick box + `← JOINT` label | Object receiving joint attention from all participants |
| Gold box + `← LOCKED` label | Object under fixation lock-on |
| Green tip circle | Gaze ray snapped to a nearby object (adaptive ray) |
| Dwell arc around face dot | Progress toward fixation lock-on (0 → 100%) |
| Teal circle + `CONVERGE` label | Cluster where multiple gaze tips converge |
| Top-right HUD panel | Per-pair cosine similarity (instantaneous + running average) |
| Bottom-right HUD | Joint attention % and currently attended objects |
| Top-left overlay | FPS + hit event count |

---

## License

MindSight is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

This project uses [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) for YOLO-based object detection. If you distribute or provide network access to this software, you must make the complete corresponding source code available under the same license. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for a full list of third-party dependencies and their licenses.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/kylen-d/mindsight).

For bug reports and feature requests, use [GitHub Issues](https://github.com/kylen-d/mindsight/issues).
