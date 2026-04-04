# Installation

This guide walks you through setting up MindSight on your local machine.

---

## Prerequisites

- **Python 3.10** or newer
- **PyTorch** (CPU is sufficient; GPU accelerates inference)
- *Optional:* CUDA toolkit (NVIDIA GPUs) or CoreML support (Apple Silicon)

!!! note "Apple Silicon users"
    MindSight supports CoreML acceleration via `onnxruntime-silicon`. No CUDA installation is needed on macOS with Apple Silicon.

---

## Clone the Repository

```bash
git clone https://github.com/kylen-d/mindsight.git
cd MindSight
```

---

## Create a Virtual Environment

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows"

    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    ```

---

## Install Dependencies

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

!!! note "PyTorch GPU support"
    The `requirements.txt` installs CPU-only PyTorch by default. For GPU acceleration, install PyTorch **first** using the appropriate command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.

!!! note "ONNX Runtime variants"
    For NVIDIA GPU inference, replace `onnxruntime` with `onnxruntime-gpu`. For Apple Silicon CoreML, use `onnxruntime-silicon`.

Alternatively, use the helper script for platform-aware installation:

```bash
python install_dependencies.py          # auto-detects CUDA / Apple Silicon
python install_dependencies.py --dry-run  # preview without installing
```

---

## Download Gaze Model Weights

MindSight supports multiple gaze-estimation backends. Each requires its own model weights.

### MGaze (default)

Download ONNX or PyTorch weights and place them in:

```
GazeTracking/Backends/MGaze/gaze-estimation/weights/
```

You can download weights using the included script:

```bash
cd GazeTracking/Backends/MGaze/gaze-estimation
bash download.sh
```

### Gazelle

Download the Gazelle model weights separately and pass the path at launch:

```bash
python MindSight.py --gazelle-model /path/to/gazelle_weights.pth
```

### L2CS

Download the L2CS weights and pass the path at launch:

```bash
python MindSight.py --l2cs-model /path/to/l2cs_weights.pkl
```

### UniGaze

!!! warning "Non-commercial license"
    UniGaze is released under a non-commercial license. Review its terms before use.

UniGaze dependencies (`timm`) are included in `requirements.txt`. Download the UniGaze model weights and pass the path at launch:

```bash
python MindSight.py --unigaze-model /path/to/unigaze_weights.pth
```

---

## YOLO Weights

YOLO and YOLOE weights are **auto-downloaded** by the Ultralytics library on first use. No manual download is required. The weights are cached locally after the initial download.

---

## Verify Installation

Run the following command to confirm MindSight is installed correctly:

```bash
python MindSight.py --help
```

You should see a list of available command-line arguments and their descriptions.

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

- Check that the `GazeTracking/Backends/MGaze/gaze-estimation/weights/` directory exists and contains the expected weight files.
- Download weights using `bash GazeTracking/Backends/MGaze/gaze-estimation/download.sh`.

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
