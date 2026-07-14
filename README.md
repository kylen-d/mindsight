# MindSight — Multi-Person Attention Tracking for Cognitive Science Research

<p align="center">
  <img src="mindsightlogo.png" alt="MindSight logo">
</p>

> **v1.0.0** -- first stable release. Bug reports and feedback are welcome via [GitHub Issues](https://github.com/kylen-d/mindsight/issues).

MindSight combines multi-person gaze estimation with YOLO object detection to determine *where* and *what* every participant in a scene is looking at, frame by frame, and turns that signal into measurements of attention-based psychological phenomena — such as joint attention, mutual gaze, social referencing, and more.

**Featured at PURC 2026📚🎉** — This project has been featured at the **University of British Columbia's** 28th Annual *Psychology Undergraduate Research Conference!* Massive thanks to the folks at the [UBC Motivated Cognition Lab](https://mclab.psych.ubc.ca/) for this incredible opportunity, and for all their help in supervising and supporting this project!

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![CI](https://github.com/kylen-d/mindsight/actions/workflows/ci.yml/badge.svg)](https://github.com/kylen-d/mindsight/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v1.0.0-success.svg)](https://github.com/kylen-d/mindsight/releases)



Full documentation lives at **[kylen-d.github.io/mindsight-docs](https://kylen-d.github.io/mindsight-docs/)**.

---

## What it does

- **Multi-person gaze + object intersection** — independently tracks every detected face (colour-coded per person), estimates each one's gaze, and resolves which detected object each participant is looking at via ray–bounding-box intersection.
- **Gaze-LLE Blend as the primary gaze mode** — a per-face pitch/yaw backend (MobileGaze) is periodically corrected against scene-level Gaze-LLE heatmaps, with One-Euro smoothing and fixation-aware anchoring. Plain per-face and plain scene-level modes are also available.
- **Eight built-in phenomena** — joint attention, mutual gaze, social referencing, gaze following, gaze aversion, scanpath, gaze leadership, and attention span, each with its own tuning parameters.
- **Projects, batch processing, and resume** — organize studies as project directories, batch every video, aggregate per condition, and resume interrupted runs from a ledger.
- **Visual prompts + depth** — detect study-specific objects from example images with YOLOE visual prompts, and optionally add monocular depth estimation to inform ray length and object snapping.
- **Extensible everywhere** — a plugin system for gaze backends, detection post-processing, phenomena, and data collection, plus a schema-driven CLI and YAML pipeline configs.

> See the [pipeline overview](https://kylen-d.github.io/mindsight-docs/concepts/pipeline/) for how the stages fit together end to end.

---

## Installation

The fastest way to run MindSight — **no Python setup required** — is the double-click installer for your platform. If you want an editable source checkout (running the tests, contributing, or working on a platform without a prebuilt installer), skip to [Developer install](#developer-install).

### Quick install (recommended)

The installer provisions a self-contained Python, installs MindSight with locked dependencies, downloads the required model weights, and creates a launcher. You do not need Python or anything else installed first.

1. **Download the release zip** for your platform — `MindSight-1.0.0-mac.zip` or `MindSight-1.0.0-win.zip` — from the [GitHub Releases](https://github.com/kylen-d/mindsight/releases) page.
2. **Extract** it somewhere you can find again (Desktop or Downloads is fine).
3. **Run the installer:**
   - **Windows:** double-click `Install-MindSight.bat`. If the blue **"Windows protected your PC"** SmartScreen box appears, click **More info → Run anyway** (expected — the in-house tool is unsigned).
   - **macOS:** right-click (or Control-click) `Install-MindSight.command` and choose **Open**, then click **Open** in the Gatekeeper dialog. (A plain double-click only offers "Move to Trash"; right-click → Open is the way past this.)
4. A console/Terminal window walks through setup and finishes with `MindSight install: PASS`. It creates **`/Applications/MindSight.app` plus a Desktop link** on macOS, and **Start Menu and Desktop shortcuts** on Windows. It is safe to re-run — re-running **updates** an existing install and skips finished work.

Platform-specific details (SmartScreen / Gatekeeper, first-launch notes) are in [`installer/INSTALL-WINDOWS.md`](installer/INSTALL-WINDOWS.md) and [`installer/INSTALL-MACOS.md`](installer/INSTALL-MACOS.md).

### Developer install

**Requirements:** Python 3.10+ (tested on 3.14) and [uv](https://docs.astral.sh/uv/) recommended. A GPU is optional — CPU works, while CUDA (NVIDIA) or MPS/CoreML (Apple Silicon) accelerate inference. All dependencies are declared in `pyproject.toml` and pinned in the committed `uv.lock` (torch/torchvision, onnxruntime, ultralytics, uniface/RetinaFace, timm, opencv, matplotlib, pandas, PyQt6, PyYAML).

For an editable source checkout:

```bash
git clone https://github.com/kylen-d/mindsight.git
cd mindsight

uv sync                    # exact locked versions from uv.lock (recommended)
# --- or ---
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e .           # resolves everything from pyproject.toml
```

This installs the `mindsight`, `mindsight-gui`, and `mindsight-weights` console commands. Download the required model weights and launch the app:

```bash
mindsight-weights --required     # the required weights for the default pipeline
mindsight-gui                    # launch the desktop app
```

`mindsight-weights --all` fetches every weight in the checksummed manifest; `--verify-only` checks checksums without downloading. YOLO/YOLOE detector weights (e.g. `yolov8n.pt`) are fetched on first use.

> Full platform notes, GPU/CoreML setup, and troubleshooting: [installation guide](https://kylen-d.github.io/mindsight-docs/getting-started/installation/).

---

## Feature tour — the desktop app

`mindsight-gui` (or the installed **MindSight** launcher) opens a six-tab window: **Analyze Footage · Projects · VP Builder · Inference Tuning · Models · About**. A menu bar adds project management (File), a light/dark **View → Theme** toggle, the **Inference Settings** dialog (Tools), and in-app documentation (Help).

> 🎬 **Demo coming soon** — SHOT:gui-tour — full-window walkthrough across all six tabs and the menu bar.

### Analyze Footage

The run surface, with three modes: **Project** (batch a whole study), **Video File** (drop a single clip for a quick analysis), and **Camera** (live capture, saved as an importable session sidecar). Runs show a live preview plus a tabbed panel — Log, Charts, Output CSVs, and a live dashboard — that renders each run's outputs as they are written.

> 🎬 **Demo coming soon** — SHOT:quick-analysis — drag a clip into Video File mode; the output folder auto-fills and live charts fill the pane.

Guide: [Analyze Footage](https://kylen-d.github.io/mindsight-docs/guides/analyze-footage/)

### Projects

Create and manage studies. The **Build New Project** wizard steps through Study → Videos → Tag → Pipeline → Review; you can **Plan a session** (a run awaiting footage), **Record a Live Session** (records from a camera, then auto-analyzes), and use **Crop & Adjust** to non-destructively crop/re-fps footage with an optional YOLOE-based auto-crop.

> 🎬 **Demo coming soon** — SHOT:record-live-session — the Record Session dialog: pick a camera, choose a planned session, record, then auto-analysis begins.

Guides: [Projects and Sessions](https://kylen-d.github.io/mindsight-docs/guides/projects-and-sessions/), [Crop and Adjust](https://kylen-d.github.io/mindsight-docs/guides/crop-and-adjust/)

### VP Builder

Build and test YOLOE **Visual Prompt** files: add reference images, draw and label bounding boxes, then run inference to preview detections. **Extract Frames…** pulls stills from a video to annotate, and **Export Portable…** writes a self-contained `.vp.zip` archive (image paths rewritten archive-relative) for sharing between machines.

> 🎬 **Demo coming soon** — SHOT:vp-annotate — add a reference image, add a class, drag a box, assign it, save the VP file.

Guide: [Visual Prompts](https://kylen-d.github.io/mindsight-docs/guides/visual-prompts/)

### Inference Tuning

A live **playground** for dialing in detection, gaze, and phenomena parameters against a clip or camera, with a real-time preview and dashboard, a plugin panel, and preset/YAML round-tripping. This tab is a decoupled scratchpad — the authority for what an actual study run uses is the **Inference Settings** dialog (Tools → Inference Settings).

> 🎬 **Demo coming soon** — SHOT:tuning-live — load a clip, press Start, watch the overlay and dashboard update as a slider is dragged.

Guide: [Inference Settings and Tuning](https://kylen-d.github.io/mindsight-docs/guides/inference-settings-and-tuning/)

### Models

A manifest-driven manager for model weights: per-weight backend, whether the current config needs it, on-disk state and size, with **Install**, **Verify** (checksums), and **Re-download** actions.

Guide: [Quickstart (GUI)](https://kylen-d.github.io/mindsight-docs/getting-started/quickstart-gui/)

### About

An offline documentation reader that renders the bundled guides in-app, plus version and license info. Pairs with the **View → Theme** toggle (auto / light / dark).

> 🎬 **Demo coming soon** — SHOT:about-reader — click a guide card, the doc opens in the in-app reader; SHOT:theme-toggle — View → Theme recolours the whole window live.

Guides: [About and Theming](https://kylen-d.github.io/mindsight-docs/guides/about-and-theming/), [Where Things Live](https://kylen-d.github.io/mindsight-docs/guides/where-things-live/)

---

## CLI quickstart

```bash
mindsight                          # launches on the webcam (--source 0)
python MindSight.py [OPTIONS]       # or run the CLI wrapper directly
```

```bash
# Analyze one video with every phenomenon, saving an annotated video + summary CSV
mindsight --source video.mp4 --all-phenomena --save --summary

# Use the pre-tuned Gaze-LLE Blend config, with heatmaps
mindsight --source video.mp4 --pipeline configs/pipeline_known_good.yaml --save --heatmap

# Anonymize faces and label participants positionally (track 0 -> S70, track 1 -> S71)
mindsight --source video.mp4 --save --anonymize blur --participant-ids S70,S71

# Batch-process a whole study (resumes from the ledger by default)
mindsight --project Projects/MyStudy/
```

The CLI exposes **over 150 flags** across detection, gaze, ray-forming (Gaze-LLE Blend), depth, phenomena, performance, and plugin families. Rather than reproduce them here, see the full **[CLI flags reference](https://kylen-d.github.io/mindsight-docs/reference/cli-flags/)**.

### Pipeline configuration

Instead of long flag lists, define a reusable YAML pipeline config and point `--pipeline` at it (CLI flags always override YAML values):

```bash
mindsight --pipeline my_pipeline.yaml --source video.mp4
```

A ready-to-use, pre-tuned config — Gaze-LLE Blend wiring plus detection and ray-geometry values validated on classroom-style footage — ships as [`configs/pipeline_known_good.yaml`](configs/pipeline_known_good.yaml). A lighter-weight variant is [`configs/pipeline_low_power.yaml`](configs/pipeline_low_power.yaml). See the [pipeline YAML schema](https://kylen-d.github.io/mindsight-docs/reference/pipeline-yaml-schema/) for the full structure, and the [first-project guide](https://kylen-d.github.io/mindsight-docs/getting-started/first-project/) for project.yaml, conditions, and participants.

### Architecture

```
Camera / Video / Image
        │
        ▼
  YOLO / YOLOE ─────────► object bounding boxes
        │
  RetinaFace ───────────► face bounding boxes
        │
  Depth Estimation ─────► per-scene depth map (optional; MiDaS)
        │
        ▼
  Gaze Estimation ──────► pitch + yaw per face  (MobileGaze / Gaze-LLE)
        │
  Ray Forming ──────────► Gaze-LLE Blend, One-Euro smoothing, fixation anchoring
        │
        ▼
  Ray–BBox Intersection ► hit list  (face_idx, object_idx)
        │
        ├──► Gaze convergence / snap / lock-on
        ├──► Phenomena engine (JA, Mutual Gaze, Social Ref, …)
        └──► Data collection (video, CSV, heatmaps, charts)
```

> Deeper dive: [architecture guide](https://kylen-d.github.io/mindsight-docs/developer/architecture/).

### Gaze modes

MindSight supports three gaze paths. Model weights live under `Weights/` (e.g. `Weights/MGaze/`) and are resolved from the checksummed manifest.

| Mode | How to enable | Notes |
|---|---|---|
| **MobileGaze** (per-face) | `--mgaze-model` (`.onnx`, or `.pt` with `--mgaze-arch`) | Fast per-face pitch/yaw; ONNX uses CoreML/CUDA/CPU. Architectures: `resnet18/34/50`, `mobilenetv2`, `mobileone_s0`–`s4`. |
| **Gaze-LLE** (scene-level) | `--gazelle-model <ckpt.pt>` (`--gazelle-name` variant) | Single DINOv2 forward pass over the whole scene; outputs a gaze heatmap. |
| **Gaze-LLE Blend** (primary) | `--rf-gazelle-model` + a per-face backend | Periodically corrects per-face rays against Gaze-LLE heatmaps with One-Euro smoothing and fixation anchoring. Pre-wired in `configs/pipeline_known_good.yaml`. |

Gaze-LLE `--gazelle-name` variants: `gazelle_dinov2_vitb14`, `gazelle_dinov2_vitl14`, and their `_inout` counterparts (which add an in/out-of-frame confidence score).

---

## Phenomena

MindSight tracks eight attention-based phenomena out of the box (each with its own tuning parameters and CLI flag):

| Phenomenon | Flag | What it measures |
|---|---|---|
| **Joint attention** | `--joint-attention` | Two or more people fixating the same object at once — a core marker in early development, ASD screening, and collaboration research. |
| **Mutual gaze** | `--mutual-gaze` | Two people looking directly at each other (eye contact) — social bonding, turn-taking, shared intentionality. |
| **Social referencing** | `--social-ref` | Looking at another's face, then redirecting to an object — infant uncertainty resolution and emotional cueing. |
| **Gaze following** | `--gaze-follow` | Shifting gaze to match where another is looking — theory of mind, social learning, attention cueing. |
| **Gaze aversion** | `--gaze-aversion` | Sustained avoidance of a visible salient object — social anxiety, ASD, phobia research. |
| **Scanpath** | `--scanpath` | The ordered sequence of fixation targets per participant — visual search, expertise, reading patterns. |
| **Gaze leadership** | `--gaze-leader` | One participant consistently directing others' attention first — group dynamics and leadership research. |
| **Attention span** | `--attn-span` | Mean duration of completed glances per participant and object — sustained attention, ADHD screening, engagement. |

Enable everything at once with `--all-phenomena`. Full definitions and parameters: [phenomena guide](https://kylen-d.github.io/mindsight-docs/phenomena/).

---

## Outputs

A run can produce:

- **Annotated video** (`--save`) — bounding boxes, gaze rays, and dashboard overlays.
- **Summary CSV** (`--summary`) — one tidy long-format table (`video_name, conditions, phenomenon, participant, partner, object, metric, value`).
- **Per-frame events CSV** (`--log`) — one row per gaze–object hit per frame.
- **Phenomena episodes CSV** — merged start/end/duration episodes across trackers.
- **Heatmaps** (`--heatmap`) — per-participant gaze accumulation images.
- **Charts** (`--charts`) — post-run phenomena time-series.
- **`Global_*` aggregates** — in project mode, per-study rollups across all videos and conditions.

> Full directory layout and column definitions: [outputs guide](https://kylen-d.github.io/mindsight-docs/concepts/outputs/).

---

## Project structure

```
MindSight/
├── MindSight.py / MindSight_GUI.py   # CLI + GUI entry-point shims
├── pyproject.toml                    # package config, console scripts, linter
├── mindsight/                        # core package (pip install -e .)
│   ├── ObjectDetection/              # YOLO / YOLOE detection
│   ├── GazeTracking/                 # gaze backends + processing
│   ├── PostProcessing/RayForming/    # Gaze-LLE Blend ray forming
│   ├── DepthEstimation/              # monocular depth (MiDaS)
│   ├── Phenomena/                    # built-in phenomena pack
│   ├── outputs/                      # video, CSV, heatmaps, charts
│   ├── project/                      # project batch runner + resume ledger
│   ├── GUI/                          # PyQt6 six-tab desktop app
│   ├── io/, utils/                   # sources/writers, geometry, device
│   └── config*.py, cli*.py           # schema, YAML loader, CLI frontend
├── Plugins/                          # gaze, detection, phenomena, data-collection plugins
│   ├── GazeTracking/                 # Gazelle, IrisRefinedGaze
│   ├── ObjectDetection/              # GazeBoost
│   ├── Phenomena/                    # EyeMovement, Pupillometry, NovelSalience
│   ├── DataCollection/               # custom data output
│   └── TEMPLATE/                     # skeleton plugin for developers
├── configs/                          # known-good + low-power pipeline YAMLs
├── installer/                        # release-zip installers + build scripts
├── Weights/                          # model weights (download on demand)
├── tests/                            # pytest suite (930 tests)
└── docs/                             # MkDocs documentation
```

---

## Development

```bash
uv sync                              # dev environment
uv run pytest                        # run the 930-test suite
uv run pytest -m "not slow"          # skip the slow-marked tests for a fast loop
uv run ruff check .                  # lint
```

MindSight is built to be extended. Plugins register at four points — **gaze backends, object-detection post-processing, phenomena, and data collection** — and contribute their own CLI flags automatically. See the [plugin system](https://kylen-d.github.io/mindsight-docs/developer/plugin-system/) and the [plugin tutorial](https://kylen-d.github.io/mindsight-docs/developer/plugin-tutorial/) to write your own; `Plugins/TEMPLATE/` is a working skeleton.

Contributions are welcome. Please open an issue (the repo provides issue templates) or a pull request on [GitHub](https://github.com/kylen-d/mindsight); use [GitHub Issues](https://github.com/kylen-d/mindsight/issues) for bugs and feature requests.

---

## Documentation

The full docs site — [kylen-d.github.io/mindsight-docs](https://kylen-d.github.io/mindsight-docs/) — is the authority for everything summarized here:

- **Get started:** [Install](https://kylen-d.github.io/mindsight-docs/getting-started/installation/) · [Run a Study (tutorial)](https://kylen-d.github.io/mindsight-docs/studies/run-a-study-tutorial/) · [Quickstart CLI](https://kylen-d.github.io/mindsight-docs/getting-started/quickstart-cli/) · [Quickstart GUI](https://kylen-d.github.io/mindsight-docs/getting-started/quickstart-gui/)
- **Concepts:** [The pipeline](https://kylen-d.github.io/mindsight-docs/concepts/pipeline/) · [Understanding outputs](https://kylen-d.github.io/mindsight-docs/concepts/outputs/) · [Phenomena](https://kylen-d.github.io/mindsight-docs/phenomena/)
- **Reference:** [CLI flags](https://kylen-d.github.io/mindsight-docs/reference/cli-flags/) · [pipeline.yaml schema](https://kylen-d.github.io/mindsight-docs/reference/pipeline-yaml-schema/) · [Inference settings](https://kylen-d.github.io/mindsight-docs/reference/inference-settings/)
- **Develop:** [Architecture](https://kylen-d.github.io/mindsight-docs/developer/architecture/) · [Plugin system](https://kylen-d.github.io/mindsight-docs/developer/plugin-system/) · [Testing](https://kylen-d.github.io/mindsight-docs/developer/testing/)

---

## License & acknowledgments

MindSight is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0). It uses [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) for YOLO-based detection — if you distribute or provide network access to this software, you must make the complete corresponding source available under the same license. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the full dependency list and licenses.

This project builds on the MobileGaze and Gaze-LLE gaze-estimation methods, the RetinaFace face detector, and Ultralytics YOLO/YOLOE. Deepest thanks to the [UBC Motivated Cognition Lab](https://mclab.psych.ubc.ca/) for supervising and supporting the work.
