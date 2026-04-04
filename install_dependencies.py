#!/usr/bin/env python3
"""
MindSight dependency installer.

Installs all packages required by:
  - gaze_tracker.py / ObjectDetection/YOLO/yolo_tracking.py  (the MindSight orchestration layer)
  - GazeTracking/Backends/MGaze/gaze-estimation/  (the embedded gaze-estimation submodule)

Usage:
    python install_dependencies.py
    python install_dependencies.py --dry-run
    python install_dependencies.py --no-torch   # skip heavy PyTorch install
"""

import argparse
import platform
import subprocess
import sys

# ---------------------------------------------------------------------------
# Package lists
# ---------------------------------------------------------------------------

# Packages needed by gaze_tracker.py and yolo_tracking.py (MindSight layer).
# ultralytics was missing from gaze-estimation/requirements.txt but is
# imported by both top-level modules.
MINDSIGHT_PACKAGES = [
    "ultralytics",          # YOLOv8 — used by gaze_tracker.py and yolo_tracking.py
]

# Packages from GazeTracking/Backends/MGaze/gaze-estimation/requirements.txt (pinned versions kept
# to match the embedded venv that ships with the submodule).
GAZE_ESTIMATION_PACKAGES = [
    "onnxruntime==1.19.0",
    "opencv-python==4.10.0.84",
    "pillow==10.2.0",
    "tqdm==4.66.5",
    "uniface==1.1.0",       # RetinaFace face detector
]

# PyTorch / TorchVision are listed separately so the caller can skip them
# (e.g. when only running ONNX inference and wanting a lighter install).
TORCH_PACKAGES_DEFAULT = [
    "torch==2.4.0",
    "torchvision==0.19.0",
]

# Apple-Silicon Macs ship a separate PyTorch index for MPS acceleration.
TORCH_PACKAGES_APPLE_SILICON = [
    "torch==2.4.0",
    "torchvision==0.19.0",
]
TORCH_INDEX_APPLE_SILICON = "https://download.pytorch.org/whl/cpu"

# CUDA 12.1 index (adjust the cu### suffix for other CUDA versions).
TORCH_INDEX_CUDA = "https://download.pytorch.org/whl/cu121"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_pip(packages: list[str], extra_args: list[str] | None = None, dry_run: bool = False) -> None:
    """Call `pip install` for the given package list."""
    if not packages:
        return
    cmd = [sys.executable, "-m", "pip", "install"] + (extra_args or []) + packages
    print(f"\n  $ {' '.join(cmd)}")
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] pip exited with code {result.returncode}.")
        sys.exit(result.returncode)


def detect_platform() -> str:
    """Return 'apple_silicon', 'cuda', or 'cpu'."""
    machine = platform.machine().lower()
    system  = platform.system().lower()
    if system == "darwin" and machine == "arm64":
        return "apple_silicon"
    # Rough heuristic: if nvcc is on PATH assume CUDA is available.
    try:
        subprocess.run(["nvcc", "--version"], capture_output=True, check=True)
        return "cuda"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "cpu"


def install_torch(platform_tag: str, dry_run: bool) -> None:
    packages = (
        TORCH_PACKAGES_APPLE_SILICON if platform_tag == "apple_silicon"
        else TORCH_PACKAGES_DEFAULT
    )
    if platform_tag == "cuda":
        run_pip(packages, extra_args=["--index-url", TORCH_INDEX_CUDA], dry_run=dry_run)
    elif platform_tag == "apple_silicon":
        # pip resolves the correct MPS/CPU wheel from the default index on macOS ARM.
        run_pip(packages, dry_run=dry_run)
    else:
        run_pip(packages, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Install MindSight Python dependencies.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip commands that would be run without executing them.",
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="Skip PyTorch / TorchVision installation (use when running ONNX-only inference).",
    )
    args = parser.parse_args()

    dry_run  = args.dry_run
    no_torch = args.no_torch

    plat = detect_platform()

    print("=" * 60)
    print("  MindSight — dependency installer")
    print("=" * 60)
    print(f"  Python  : {sys.version}")
    print(f"  Platform: {plat}")
    if dry_run:
        print("  Mode    : DRY RUN (no packages will be installed)")
    print()

    # 1. Upgrade pip first to avoid resolver issues with older pip versions.
    print("[1/4] Upgrading pip …")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    print(f"\n  $ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)

    # 2. GazeTracking/gaze-estimation submodule requirements.
    print("\n[2/4] Installing gaze-estimation packages …")
    run_pip(GAZE_ESTIMATION_PACKAGES, dry_run=dry_run)

    # 3. PyTorch (optional).
    if no_torch:
        print("\n[3/4] Skipping PyTorch (--no-torch specified).")
    else:
        print(f"\n[3/4] Installing PyTorch for platform '{plat}' …")
        install_torch(plat, dry_run)

    # 4. MindSight top-level packages (ultralytics / YOLO).
    print("\n[4/4] Installing MindSight packages …")
    run_pip(MINDSIGHT_PACKAGES, dry_run=dry_run)

    print()
    print("=" * 60)
    if dry_run:
        print("  Dry run complete — no packages were installed.")
    else:
        print("  All dependencies installed successfully.")
        print()
        print("  Quick-start:")
        print("    python gaze_tracker.py --help")
    print("=" * 60)


if __name__ == "__main__":
    main()
