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

All dependencies are declared in `pyproject.toml` (the single source of truth,
with a committed `uv.lock` pinning exact versions). Install with either:

```bash
uv sync                           # exact locked versions (recommended)
```

or plain pip:

```bash
pip install -e .                  # editable install; resolves from pyproject.toml
```

!!! note "Editable install"
    The editable install registers the `mindsight` package and installs the
    `mindsight`, `mindsight-gui`, and `mindsight-weights` commands.

!!! note "PyTorch acceleration"
    The default PyPI wheels give you MPS on Apple Silicon and CUDA on Linux
    out of the box; Windows installs run PyTorch on CPU. For NVIDIA ONNX
    inference, replace `onnxruntime` with `onnxruntime-gpu`.

---

## Download Model Weights

All model weights are stored in `Weights/{backend}/`. Download them with:

```bash
python scripts/download_weights.py            # all backends
python scripts/download_weights.py --backend MGaze   # specific backend
python scripts/download_weights.py --dry-run         # preview only
```

### What gets downloaded

| Backend | Method | Notes |
|---------|--------|-------|
| **MobileGaze** (default) | Download script | ONNX + PyTorch variants in `Weights/MGaze/` |
| **Gaze-LLE** | Download script | Checkpoints in `Weights/Gazelle/` |
| **YOLO** | Auto (Ultralytics) | Downloaded on first use, cached in `Weights/YOLO/` |

---

## Verify Installation

Run the following command to confirm MindSight is installed correctly:

```bash
python MindSight.py --help
```

Or use the console command: `mindsight --help`

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

- Check that the `Weights/MGaze/` directory exists and contains the expected weight files.
- Download weights using `python scripts/download_weights.py --backend MGaze`.

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
