# Installation

There are two ways to install MindSight:

- **[One-click install](#one-click-install-recommended)** -- the recommended
  path for researchers and lab machines. A double-click installer sets up
  everything: a self-contained Python, MindSight with locked dependencies, the
  required model weights, and an app launcher. **You do not need Python -- or
  anything else -- installed first.**
- **[Developer install](#developer-install)** -- an editable source checkout
  for contributing, running the tests, or platforms without a prebuilt
  installer.

---

## One-click install (recommended)

1. **Download the release zip** for your platform --
   `MindSight-1.3.2-mac.zip` or `MindSight-1.3.2-win.zip` -- from the
   [GitHub Releases page](https://github.com/kylen-d/mindsight/releases).
2. **Extract** it somewhere you can find again (Desktop or Downloads is fine).
3. **Run the installer:**

    === "macOS"

        Right-click (or Control-click) `Install-MindSight.command` and choose
        **Open**, then click **Open** in the Gatekeeper dialog. A plain
        double-click only offers "Move to Trash" -- right-click → Open is the
        way past this for an unsigned in-house tool.

    === "Windows"

        Double-click `Install-MindSight.bat`. If the blue **"Windows protected
        your PC"** SmartScreen box appears, click **More info → Run anyway**
        (expected -- the in-house tool is unsigned).

4. A console/Terminal window walks through setup and finishes with
   `MindSight install: PASS`.

When it finishes you get **`/Applications/MindSight.app` plus a Desktop link**
on macOS, and **Start Menu and Desktop shortcuts** on Windows. Launch it like
any other app and you land on the [GUI Tour](quickstart-gui.md).

!!! tip "Re-running is safe -- and how you update"
    Running the installer again **updates** an existing install and skips work
    that is already done. When a new MindSight release ships, download the new
    zip and re-run.

Platform-specific detail (Gatekeeper/SmartScreen notes, first-launch camera
permissions, install locations) lives in
[`installer/INSTALL-MACOS.md`](https://github.com/kylen-d/mindsight/blob/main/installer/INSTALL-MACOS.md)
and
[`installer/INSTALL-WINDOWS.md`](https://github.com/kylen-d/mindsight/blob/main/installer/INSTALL-WINDOWS.md).
Where the installed files end up is covered in
[Where things live](../guides/where-things-live.md).

---

## Developer install

### Prerequisites

- **Python 3.10** or newer
- A GPU is optional -- CPU works; CUDA (NVIDIA) or MPS/CoreML (Apple Silicon)
  accelerate inference

!!! note "Apple Silicon users"
    The standard `onnxruntime` dependency includes the CoreML execution
    provider on macOS -- ONNX inference is accelerated out of the box, and no
    CUDA installation is needed.

### Clone the repository

```bash
git clone https://github.com/kylen-d/mindsight.git
cd mindsight
```

### Install dependencies

All dependencies are declared in `pyproject.toml` (the single source of truth,
with a committed `uv.lock` pinning exact versions). With
[uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                           # exact locked versions
```

or with plain pip in a virtual environment:

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

=== "Windows"

    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    pip install -e .
    ```

!!! note "Editable install"
    The editable install registers the `mindsight` package and installs the
    `mindsight`, `mindsight-gui`, and `mindsight-weights` commands.

!!! note "PyTorch acceleration"
    The default PyPI wheels give you MPS on Apple Silicon and CUDA on Linux
    out of the box; Windows installs run PyTorch on CPU. For NVIDIA ONNX
    inference, replace `onnxruntime` with `onnxruntime-gpu`.

### Download model weights

Weights live in `Weights/{backend}/` and are managed by a checksummed manifest
(`weights_manifest.json`) -- the same one the GUI's Models tab uses:

```bash
mindsight-weights                     # the 6 required weights (default)
mindsight-weights --all               # every downloadable weight
mindsight-weights --backend MGaze     # one backend (repeatable)
mindsight-weights --verify-only       # check checksums, download nothing
mindsight-weights --dry-run           # show what would be downloaded
```

(`python scripts/download_weights.py` is an equivalent wrapper for a checkout
where the console command is not on PATH.)

#### What gets downloaded

| Backend | Required set | Notes |
|---------|--------------|-------|
| **YOLO / YOLOE** | `yolov8n.pt` | Larger YOLOv8 and YOLOE variants are optional, fetched with `--all` or on demand |
| **MobileGaze** | `resnet50_gaze.onnx`, `mobileone_s0_gaze.onnx` | PyTorch variants optional, in `Weights/MGaze/` |
| **Gaze-LLE** | `gazelle_hgnetv2_pico_inout_distill_1x3x640x640_1xNx4.onnx` (default blend engine), `gazelle_dinov2_vitb14.pt` (torch fallback + standalone backend) | The larger `vitl14` checkpoint and the DINOv3 ViT tiny-plus / ViT-S/16 quality tiers (incl. single-face exports for Apple-GPU CoreML) are optional, in `Weights/Gazelle/` |
| **MobileClip** | -- | `mobileclip_blt.ts` is auto-fetched by Ultralytics on first visual-prompt use |
| **MediaPipe** | -- | `face_landmarker.task` (468-point landmarker feeding the head-pose-normalized gaze backends) is optional, in `Weights/Mediapipe/`; fetch with `mindsight-weights --backend Mediapipe` |
| **MPIIFaceGaze / Adas** | -- | `mpiifacegaze_resnet_simple.pth` (research-provenance) and `gaze-estimation-adas-0002.onnx` (Apache-2.0) power the opt-in head-pose-normalized backends; in `Weights/MPIIFaceGaze/` and `Weights/AdasGaze/` |

### Verify the installation

```bash
mindsight --help          # or: python MindSight.py --help
```

You should see the full list of command-line flags. Then take the
[CLI quickstart](quickstart-cli.md) for a first run.

---

## Troubleshooting

### CUDA not found

```
RuntimeError: CUDA not available
```

- Verify your NVIDIA driver is installed: `nvidia-smi`
- Ensure you installed the CUDA-compatible PyTorch build. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct install command.
- Confirm `onnxruntime-gpu` is installed instead of the CPU-only `onnxruntime`.

### Missing model weights

```
FileNotFoundError: .../mobileone_s0_gaze.onnx
```

- Run `mindsight-weights --verify-only` to see what is present, mismatched, or missing.
- Download the missing backend with `mindsight-weights --backend MGaze` (or use the GUI's **Models** tab).

### Reinstall doesn't seem to pick up an update

If you re-ran the installer but the app still looks out of date, a cached
package may have been reused. Clear it and re-run the installer:

```bash
uv cache clean mindsight
```

(The installer's `uv` lives on your PATH after any install; on Windows run the
same command in the installer's console.)

### Import errors

```
ModuleNotFoundError: No module named 'cv2'
```

- Make sure your virtual environment is activated.
- Re-run `pip install opencv-python` (or the missing package).
- On headless servers, use `opencv-python-headless` instead of `opencv-python`.

### PyQt6 issues on Linux

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

- Install system-level Qt dependencies:
  ```bash
  sudo apt install libxcb-xinerama0 libxcb-cursor0
  ```

!!! info "Still stuck?"
    Open an issue on the repository with the full error traceback and your `pip list` output.
