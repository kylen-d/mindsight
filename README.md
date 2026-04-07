# MindSight — Unified Object Detection and Gaze Intersection Tracker for Cognitive Science Research 

<p align="center">
  <img src="mindsightlogo.png" alt="MindSight logo">
</p>
#

> **Beta Release** — This is a pre-release version (v0.4.0-beta). APIs and features may change. Bug reports and feedback are welcome via [GitHub Issues](https://github.com/kylen-d/mindsight/issues).

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

> Gaze estimation combined with YOLO object detection for the studying of various gaze and attention-based psychological phenomena. MindSight determines what participants are looking at in real time, and provides a framework to use this information to study a wide-range of gaze-based behaviour, such as Joint-Attention.

---

## Documentation

Full documentation is available at **[kylen-d.github.io/mindsight-docs](https://kylen-d.github.io/mindsight-docs/)**.

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
- **Built-in gaze post-processing** — gaze smoothing, extending rays to nearby objects, adaptive ray-object snapping, fixation lock-on, gaze cone mode, and forward gaze detection
- **Gaze ray–bbox intersection** — determines which detected objects each participant is looking at
- **Gaze convergence detection** — detects when multiple gaze ray tips cluster near the same point
- **Visual Object Detection Prompting** — makes use of YOLOE to allow for visual object detection prompts
- **Face anonymization** — blur or black-out faces in saved outputs (`--anonymize`)
- **Live dashboard** — real-time matplotlib dashboard with FPS, hit counts, cosine similarity, and phenomena panels
- **Performance modes** — `--fast`, `--lite-overlay`, and `--no-dashboard` flags for throughput tuning
- **Device auto-detection** — automatically selects CUDA, MPS (Apple Silicon), or CPU (`--device`)
- **PyQt6 GUI** — full graphical front-end with live preview, a drag-and-drop Visual Prompt Builder, and project management

> See the [pipeline overview](https://kylen-d.github.io/mindsight-docs/user-guide/pipeline-overview/) for how the four-stage pipeline (Detection → Gaze → Phenomena → Data Collection) works end to end.

### Built-in Tracking of Various Attention-based Phenomena

- **Joint attention** — two or more people simultaneously fixating on the same object. A core marker in early cognitive development, ASD screening, and collaborative task research.
- **Mutual gaze** — two people looking directly at each other (eye contact). Central to studies of social bonding, turn-taking, and shared intentionality.
- **Social referencing** — a person looks at another's face and then redirects their gaze to an object, as if checking before engaging. Studied in infant uncertainty resolution and emotional cueing.
- **Gaze following** — one person shifts their gaze to match where another is looking. A key indicator of theory of mind, social learning, and attention cueing.
- **Gaze aversion** — sustained avoidance of a visible salient object. Relevant to social anxiety, ASD, and phobia research.
- **Scanpath analysis** — the ordered sequence of fixation targets for each participant. Used in visual search strategy, expertise studies, and reading pattern analysis.
- **Gaze leadership** — one participant's gaze consistently directs others' attention first. Studied in group dynamics, social hierarchy, and leadership research.
- **Attention span** — the average duration of completed glances per participant and object. Used in sustained attention research, ADHD screening, and engagement measurement.

Each phenomenon has its own tuning parameters — see the [phenomena guide](https://kylen-d.github.io/mindsight-docs/phenomena/) for details.

### Highly Extensible

- **Gaze Backend Plugins** — supports and includes MGaze out of the box, with ONNX, PyTorch, L2CS-Net, UniGaze, and Gazelle backends, and allows custom gaze estimation backends through the plugin system
- **Object Detection Plugins** — custom detection post-processing (e.g. the included GazeBoost plugin)
- **Phenomena Plugins** — user-written plugins to detect custom phenomena alongside the default pack
- **Data Collection Plugins** — user-written plugins for custom data output in addition to video, CSV, heatmaps, and charts

> See the [plugin system guide](https://kylen-d.github.io/mindsight-docs/developer/plugin-system/) and [plugin tutorial](https://kylen-d.github.io/mindsight-docs/developer/plugin-tutorial/) for how to write your own.

### Designed for Research

- **Built-in CSV event logging** — per-frame hit events with participant labels, conditions, and a customizable post-run summary with joint attention %, cosine similarity averages, and object look-time statistics
- **Live & post-run dashboard** — real-time phenomena overlays and post-run time-series charts (`--charts`)
- **Heatmap generation** — per-participant gaze heatmap accumulation and export (`--heatmap`)
- **Participant ID mapping** — map track IDs to meaningful labels via `--participant-ids` or `--participant-csv`
- **Project-based workflow** — user-defined pipelines with batch processing, per-condition CSV aggregation, and organized data output

> See the [data output guide](https://kylen-d.github.io/mindsight-docs/user-guide/data-output/) for full details on all output types.

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
  Gaze Estimator ──► pitch + yaw per face  (MGaze / L2CS / UniGaze / Gazelle)
        │
        ▼
  Ray–BBox Intersection ──► hit list  (face_idx, object_idx)
        │
        ├──► Gaze Convergence
        ├──► Gaze Lock-on / Smooth / Snap 
        └──► Phenomena Pipeline (JA, Mutual Gaze, Social Ref, …)
```

> See the [architecture deep dive](https://kylen-d.github.io/mindsight-docs/developer/architecture/) for module dependency graphs and per-frame processing details.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10+ recommended (tested on 3.14) |
| CUDA / CoreML | Optional — CPU works, GPU accelerates |
| Webcam | Required for live mode |

### Python packages

All dependencies are listed in `requirements.txt`. Key packages:

```
torch / torchvision          # Deep learning
onnxruntime                  # ONNX inference (or onnxruntime-gpu for CUDA)
ultralytics                  # YOLO / YOLOE object detection
clip                         # Ultralytics CLIP fork (visual prompts)
uniface                      # RetinaFace face detector
timm                         # PyTorch Image Models (UniGaze backend)
opencv-python                # Computer vision
matplotlib                   # Charts and dashboard rendering
pandas                       # Data output
PyQt6                        # GUI
PyYAML                       # Pipeline configuration
tqdm                         # Progress bars
Pillow                       # Image handling
```

> **Note:** The UniGaze backend requires `pip install unigaze` separately (non-commercial license, pins `timm==0.3.2`).

> See the [installation guide](https://kylen-d.github.io/mindsight-docs/getting-started/installation/) for troubleshooting and platform-specific instructions.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kylen-d/mindsight.git
cd mindsight
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .                  # install MindSight as an editable package
```

> **Note:** `pip install -e .` is required -- it makes the `ms` package importable and installs the `mindsight` and `mindsight-gui` console commands.

**GPU acceleration (optional):** Install PyTorch with CUDA support *before* running the above — see [pytorch.org/get-started](https://pytorch.org/get-started/locally/). For Apple Silicon CoreML, replace `onnxruntime` with `onnxruntime-silicon`.

Alternatively, use the platform-aware helper:

```bash
python scripts/install_dependencies.py        # auto-detects CUDA / Apple Silicon
```

### 4. Download gaze model weights

All model weights are centralized in `Weights/{backend}/`. Download them with:

```bash
python scripts/download_weights.py            # all backends
python scripts/download_weights.py --backend MGaze   # specific backend
```

For L2CS-Net, download weights to `Weights/L2CS/` and pass the path via `--l2cs-model`.

For Gazelle, download a checkpoint separately and pass it via `--gazelle-model`.

For UniGaze, install separately (`pip install unigaze`) and pass the model variant via `--unigaze-model`.

### 5. YOLO weights

YOLO model weights (e.g. `yolov8n.pt`) are downloaded automatically by Ultralytics on first use. YOLOE weights (e.g. `yoloe-26l-seg.pt`) are also auto-downloaded.

> See the [full installation guide](https://kylen-d.github.io/mindsight-docs/getting-started/installation/) for download links, verification steps, and troubleshooting.

---

## Project Structure

```
MindSight/
├── MindSight.py                  # CLI entry point (thin wrapper)
├── MindSight_GUI.py              # GUI entry point (thin wrapper)
├── pyproject.toml                # Package config, console_scripts, linter settings
│
├── ms/                           # Core package (pip install -e .)
│   ├── cli.py                    # CLI orchestrator (main pipeline logic)
│   ├── gui.py                    # GUI launcher
│   ├── constants.py              # Shared constants (colours, thresholds)
│   ├── pipeline_config.py        # Config dataclasses + FrameContext
│   ├── pipeline_loader.py        # YAML pipeline config loader
│   ├── project_runner.py         # Project-based batch processing
│   ├── participant_ids.py        # Participant label mapping
│   ├── weights.py                # Centralized weight file resolution
│   │
│   ├── ObjectDetection/
│   │   ├── detection_pipeline.py # YOLO detection step (ctx-based)
│   │   ├── object_detection.py   # YOLO wrapper, ObjectPersistenceCache
│   │   ├── detection.py          # Detection dataclass
│   │   └── model_factory.py      # YOLO and RetinaFace factory
│   │
│   ├── GazeTracking/
│   │   ├── gaze_pipeline.py      # Gaze step coordinator (ctx-based)
│   │   ├── gaze_processing.py    # GazeSmootherReID, snap, lock-on
│   │   ├── gaze_factory.py       # Gaze engine factory
│   │   └── Backends/             # Built-in gaze backends (MGaze, L2CS, UniGaze)
│   │
│   ├── Phenomena/
│   │   ├── phenomena_pipeline.py # Phenomena step (ctx-based unified loop)
│   │   ├── phenomena_config.py   # PhenomenaConfig dataclass
│   │   ├── helpers.py            # joint_attention, gaze_convergence
│   │   └── Default/              # Built-in phenomena pack
│   │
│   ├── DataCollection/
│   │   ├── data_pipeline.py      # CSV logging, heatmap accumulation
│   │   ├── dashboard_output.py   # Frame overlay + dashboard compositor
│   │   ├── csv_output.py         # Summary CSV writer
│   │   └── heatmap_output.py     # Per-participant heatmap generation
│   │
│   ├── GUI/
│   │   ├── main_window.py        # PyQt6 main window (3-tab layout)
│   │   ├── gaze_tab/             # Gaze tracker tab (decomposed sections)
│   │   ├── project_tab/          # Project mode tab (decomposed sections)
│   │   └── widgets.py            # Reusable GUI components
│   │
│   └── utils/
│       ├── geometry.py           # Ray geometry, pitch/yaw, bbox ops
│       └── device.py             # Shared device detection for all backends
│
├── Plugins/
│   ├── __init__.py               # Base classes + registries
│   ├── GazeTracking/             # Gaze backend plugins (Gazelle, GazelleSnap)
│   ├── ObjectDetection/          # Detection plugins (GazeBoost)
│   ├── Phenomena/                # Community phenomena plugins (NovelSalience)
│   ├── DataCollection/           # Custom data output plugins
│   └── TEMPLATE/                 # Skeleton plugin for developers
│
├── Weights/                      # Model weights (git-ignored, download on demand)
├── Projects/                     # User project directories
├── scripts/                      # Utility scripts (download_weights, install_deps)
├── tests/                        # pytest test suite (171 tests)
├── docs/                         # MkDocs documentation
└── Outputs/                      # Default output directory
```

---

## MindSight.py — CLI Usage

```bash
mindsight                          # console command (after pip install -e .)
python MindSight.py [OPTIONS]      # or run the CLI directly
```

### Quick examples

```bash
# Webcam with default settings
python MindSight.py --source 0

# After `pip install -e .`, you can also use the `mindsight` command directly:
mindsight --source video.mp4 --save

# Video file, save output, log events to CSV
python MindSight.py --source video.mp4 --save --log events.csv

# Enable joint attention and mutual gaze tracking
python MindSight.py --source video.mp4 --joint-attention --mutual-gaze

# Enable all phenomena trackers at once
python MindSight.py --source video.mp4 --all-phenomena --save --log events.csv --summary

# Use L2CS-Net for higher accuracy gaze estimation
python MindSight.py --source video.mp4 --l2cs-model weights/l2cs.pkl

# Use gaze cone mode instead of a single ray
python MindSight.py --source video.mp4 --gaze-cone 15

# Save with face anonymization, heatmaps, and charts
python MindSight.py --source video.mp4 --save --anonymize blur --heatmap --charts

# Use YOLOE Visual Prompt instead of text-class YOLO
python MindSight.py --source 0 --vp-file my_prompts.vp.json

# Fast mode — bundled optimizations for throughput
python MindSight.py --source video.mp4 --fast

# Export post-run summary CSV with participant labels
python MindSight.py --source video.mp4 --summary results.csv --participant-ids "Alice,Bob,Carol"
```

### Key arguments (by category)

The table below covers the most commonly used flags. For the **complete reference** of all ~70 flags, see the [CLI flags reference](https://kylen-d.github.io/mindsight-docs/reference/cli-flags/).

#### Orchestration

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Input: `0` = webcam, integer = camera index, path to video/image |
| `--save` | off | Write annotated output video/image to disk |
| `--log` | — | Path to per-frame event CSV |
| `--summary` | — | Path to post-run summary CSV |
| `--heatmap` | — | Generate per-participant gaze heatmaps |
| `--charts` | — | Generate post-run time-series charts |
| `--pipeline` | — | Load a YAML pipeline config file |
| `--project` | — | Batch-process videos in a project directory |
| `--device` | `auto` | Compute device: `auto`, `cpu`, `cuda`, `mps` |
| `--anonymize` | off | Face anonymization mode: `blur` or `black` |

#### Detection

| Argument | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | YOLO model weights |
| `--conf` | `0.35` | Detection confidence threshold |
| `--classes` | — | Filter to specific class names, e.g. `--classes person knife` |
| `--blacklist` | — | Suppress specific classes, e.g. `--blacklist chair` |
| `--skip-frames` | `1` | Run detection every N frames (1 = every frame) |
| `--detect-scale` | `1.0` | Scale factor for detection pass (< 1 = faster) |
| `--vp-file` | — | Visual Prompt file (`.vp.json`); switches to YOLOE detector |
| `--obj-persistence` | — | Keep ghost detections alive for N frames |

#### Gaze

| Argument | Default | Description |
|---|---|---|
| `--ray-length` | `1.0` | Gaze ray length multiplier |
| `--adaptive-ray` | `off` | Ray mode: `off`, `extend`, or `snap` |
| `--gaze-cone` | `0` | Vision cone angle in degrees (0 = single ray) |
| `--gaze-lock` | off | Enable fixation lock-on |
| `--dwell-frames` | `15` | Frames of sustained gaze before lock activates |
| `--gaze-tips` | off | Enable gaze-tip convergence detection |
| `--forward-gaze-threshold` | — | Threshold for "looking at camera" classification |

#### Gaze Backends

| Argument | Description |
|---|---|
| `--mgaze-model` | MGaze: ONNX or `.pt` gaze weights (default backend) |
| `--l2cs-model` | L2CS-Net: Path to `.pkl` or `.onnx` weights |
| `--unigaze-model` | UniGaze: Model variant (requires `pip install unigaze`) |
| `--gazelle-model` | Gazelle: Path to checkpoint; switches to scene-level backend |

#### Phenomena

| Argument | Description |
|---|---|
| `--joint-attention` | Enable joint attention tracking |
| `--mutual-gaze` | Enable mutual gaze detection |
| `--social-ref` | Enable social referencing |
| `--gaze-follow` | Enable gaze following detection |
| `--gaze-aversion` | Enable gaze aversion detection |
| `--scanpath` | Enable scanpath tracking |
| `--gaze-leader` | Enable gaze leadership tracking |
| `--attn-span` | Track per-participant attention span |
| `--all-phenomena` | Enable all phenomena trackers at once |

#### Performance

| Argument | Description |
|---|---|
| `--fast` | Enable bundled optimizations |
| `--skip-phenomena` | Run phenomena trackers every N frames |
| `--lite-overlay` | Minimal overlay rendering |
| `--no-dashboard` | Skip dashboard composition |
| `--profile` | Print per-stage timing every 100 frames |

### Pipeline configuration files

Instead of passing many CLI flags, you can define a reusable YAML pipeline config:

```yaml
# my_pipeline.yaml
detection:
  model: "yoloe-26l-seg.pt"
  conf: 0.4
gaze:
  ray_length: 1.5
  adaptive_ray: snap
phenomena:
  - joint_attention: { ja_window: 10 }
  - mutual_gaze: {}
output:
  save_video: true
  summary_csv: true
  heatmaps: true
  anonymize: blur
performance:
  fast: true
participants:
  ids: ["Alice", "Bob"]
```

```bash
python MindSight.py --pipeline my_pipeline.yaml --source video.mp4
```

CLI flags always override YAML values. For the full YAML schema including `aux_streams` and `plugins` sections, see the [pipeline YAML reference](https://kylen-d.github.io/mindsight-docs/reference/pipeline-yaml-schema/).

### Project mode

A Project is a directory with a standard layout for batch-processing multiple videos:

```
MyProject/
├── Inputs/
│   ├── Videos/         # Drop video files here
│   └── Prompts/        # VP files for this project
├── Outputs/            # Auto-populated with per-video results
│   ├── CSV Files/      # Per-video + Global + By Condition CSVs
│   ├── Videos/
│   └── heatmaps/
├── Pipeline/
│   └── pipeline.yaml   # Project-specific pipeline config
└── project.yaml        # Optional: conditions, participants, output settings
```

```bash
python MindSight.py --project Projects/MyProject/
```

This loads the project's `pipeline.yaml`, processes every video in `Inputs/Videos/`, and writes per-video outputs to `Outputs/`. The optional `project.yaml` allows you to define conditions (per-video tags), participant label mappings, and output directory customization.

> See the [project mode guide](https://kylen-d.github.io/mindsight-docs/user-guide/project-mode/) for full details on project.yaml, conditions, participants, and auxiliary streams.

### Plugin development

MindSight supports four plugin types: **Gaze**, **Object Detection**, **Phenomena**, and **Data Collection**.

> See the [plugin tutorial](https://kylen-d.github.io/mindsight-docs/developer/plugin-tutorial/) for step-by-step guides for each plugin type.

---

## MindSight_GUI.py — GUI Usage

```bash
mindsight-gui                # console command (after pip install -e .)
python MindSight_GUI.py      # or run the wrapper script directly
```

The GUI has three tabs:

### Tab 1 — Gaze Tracker

A graphical front-end for all `MindSight.py` functionality.

**Sections:**

- **Source** — camera index, video file, or image file
- **Detection mode** — YOLO (text classes) or YOLOE Visual Prompt
- **Gaze backend** — MGaze, L2CS-Net, UniGaze, or Gazelle
- **Device selector** — auto, CPU, CUDA, or MPS
- **Gaze parameters** — ray length, adaptive ray, gaze cone, lock-on, etc.
- **Phenomena panel** — toggle individual phenomena trackers
- **Plugin panel** — activate and configure installed plugins
- **Output settings** — video save, CSV log, summary, heatmaps, charts, anonymization
- **Presets** — save and restore parameter presets
- **Live preview** — annotated frames with real-time dashboard
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

### Tab 3 — Project Mode

Manage and run batch-processing projects directly from the GUI.

**Sections:**

- **Pipeline selector** — load and edit pipeline YAML configurations
- **Participants table** — map track IDs to participant labels (auto-populate or manual entry)
- **Conditions table** — tag videos with experimental conditions
- **Output settings** — configure output directory, CSV aggregation, heatmaps, and charts
- **Monitoring** — source list, live preview, progress bars, and log output

> See the [GUI guide](https://kylen-d.github.io/mindsight-docs/user-guide/gui-guide/) for detailed instructions including the pipeline dialog and settings persistence.

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

> See the [visual prompts guide](https://kylen-d.github.io/mindsight-docs/user-guide/visual-prompts/) for tips on creating effective prompts.

---

## Output & Logging

### Per-frame event CSV (`--log events.csv`)

One row per gaze-object hit per frame:

| Column | Description |
|---|---|
| `video_name` | Source video filename |
| `conditions` | Experimental conditions (from project.yaml) |
| `frame` | Frame number |
| `face_idx` | Tracked face index |
| `participant_label` | Participant label (if mapped) |
| `object` | Detected object class name |
| `object_conf` | Detection confidence |
| `bbox_x1, bbox_y1, bbox_x2, bbox_y2` | Object bounding box |
| `joint_attention` | Whether joint attention is active |

### Post-run summary CSV (`--summary`)

Three core sections written to the same file:

| Section | Description |
|---|---|
| `joint_attention` | % of frames with shared gaze target |
| `cosine_similarity` | Per-pair and overall mean cosine similarity of gaze directions |
| `object_look_time` | Per-(participant, object-class) frame count and % |

Additional sections are appended for each active phenomena tracker.

### Heatmaps (`--heatmap`)

Per-participant gaze heatmaps with Gaussian blur (sigma=40), saved as images to the output directory.

### Time-series charts (`--charts`)

Post-run matplotlib charts showing phenomena metrics over time.

### Annotated video (`--save`)

When `--save` is passed, an annotated `.mp4` is written with bounding boxes, gaze rays, and dashboard overlays. Auto-named as `[stem]_Video_Output.mp4` in the output directory.

### Face anonymization (`--anonymize`)

Blur or black-out all detected faces in saved video output. Use `--anonymize blur` or `--anonymize black`, with adjustable padding via `--anonymize-padding` (default: 0.3).

### Participant ID mapping

Map track IDs to meaningful labels:

- **Inline:** `--participant-ids "Alice,Bob,Carol"`
- **From file:** `--participant-csv participants.csv`

Labels appear in CSV output and on-screen overlays.

> See the [data output guide](https://kylen-d.github.io/mindsight-docs/user-guide/data-output/) for full details on output directory structure, CSV column definitions, and project-mode aggregation.

---

## Gaze Backends

| Backend | Trigger | Accuracy | Notes |
|---|---|---|---|
| **MGaze ONNX** (default) | `--mgaze-model` with `.onnx` path | ~11 MAE | Fastest; uses CoreML on Apple Silicon, CUDA on NVIDIA, CPU otherwise |
| **MGaze PyTorch** | `--mgaze-model` with `.pt` + `--mgaze-arch` | ~11 MAE | Requires `--mgaze-arch` to identify the architecture |
| **L2CS-Net** | `--l2cs-model <weights>` | ~3.92 MAE | ResNet50 with dual classification heads; ~3x more accurate than MGaze on MPIIGaze |
| **UniGaze** | `--unigaze-model <variant>` | ~9.4 (Gaze360) | ViT + MAE pre-training; best cross-dataset accuracy. Requires `pip install unigaze` (non-commercial) |
| **Gazelle** | `--gazelle-model <ckpt.pt>` | — | Scene-level DINOv2 model; processes all faces in a single forward pass; outputs a gaze heatmap |
| **GazelleSnap** | `--gazelle-model` + `--adaptive-ray snap` | — | Gazelle with adaptive snapping for object-level gaze assignment |

**MGaze architectures** (`--mgaze-arch`):
`resnet18`, `resnet34`, `resnet50`, `mobilenetv2`, `mobileone_s0`–`s4`

**L2CS-Net architectures** (`--l2cs-arch`):
`ResNet18`, `ResNet34`, `ResNet50` (default), `ResNet101`, `ResNet152`

**UniGaze model variants** (`--unigaze-model`):
`unigaze_b16_joint` (ViT-Base), `unigaze_l16_joint` (ViT-Large), `unigaze_h14_joint` (ViT-Huge, best accuracy)

**Gazelle model variants** (`--gazelle-name`):
`gazelle_dinov2_vitb14`, `gazelle_dinov2_vitl14`, `gazelle_dinov2_vitb14_inout`, `gazelle_dinov2_vitl14_inout`

The `_inout` variants add an in-frame / out-of-frame confidence score that modulates the gaze heatmap peak.

> See the [gaze estimation guide](https://kylen-d.github.io/mindsight-docs/user-guide/gaze-estimation/) for detailed parameter tuning, adaptive ray modes, smoothing, re-ID, and intersection algorithms.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` | Quit (CLI video/webcam mode) |
| Any key | Close (CLI image mode) |

### On-screen Overlay Legend

> See the [keyboard shortcuts & overlay reference](https://kylen-d.github.io/mindsight-docs/reference/keyboard-shortcuts/) for the full legend.

---

## License

MindSight is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

This project uses [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) for YOLO-based object detection. If you distribute or provide network access to this software, you must make the complete corresponding source code available under the same license. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for a full list of third-party dependencies and their licenses.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/kylen-d/mindsight).

For bug reports and feature requests, use [GitHub Issues](https://github.com/kylen-d/mindsight/issues).
