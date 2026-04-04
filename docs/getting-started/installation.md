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

### Core packages

```bash
pip install opencv-python numpy torch torchvision
```

### ONNX Runtime

Install **one** of the following, depending on your hardware:

=== "CPU (any platform)"

    ```bash
    pip install onnxruntime
    ```

=== "NVIDIA GPU (CUDA)"

    ```bash
    pip install onnxruntime-gpu
    ```

=== "Apple Silicon (CoreML)"

    ```bash
    pip install onnxruntime-silicon
    ```

### Object Detection

```bash
pip install ultralytics   # YOLO / YOLOE
pip install uniface        # RetinaFace face detection
```

### GUI (optional)

```bash
pip install PyQt6
```

### Data Output

```bash
pip install matplotlib pandas scipy
```

!!! tip "One-liner"
    If a `requirements.txt` is provided in the repository root, you can install everything at once:
    ```bash
    pip install -r requirements.txt
    ```

---

## Download Gaze Model Weights

MindSight supports multiple gaze-estimation backends. Each requires its own model weights.

### Default (MobileOne)

The default weights ship with the repository:

```
gaze-estimation/weights/mobileone_s0_gaze.onnx
```

Other ONNX and PyTorch weight files can also be placed in `gaze-estimation/weights/`.

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

```bash
pip install unigaze timm==0.3.2
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

- Check that the `gaze-estimation/weights/` directory exists and contains the expected `.onnx` file.
- If you cloned with `--depth 1`, large files tracked by Git LFS may not have been pulled. Run `git lfs pull`.

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
