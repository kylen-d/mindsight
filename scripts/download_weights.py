#!/usr/bin/env python
"""
scripts/download_weights.py — Download model weights for MindSight backends.

Usage:
    python scripts/download_weights.py             # download all backends
    python scripts/download_weights.py --backend MGaze
    python scripts/download_weights.py --backend YOLO --backend L2CS
    python scripts/download_weights.py --dry-run    # show what would be downloaded
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Resolve project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_ROOT = PROJECT_ROOT / "Weights"

# ── Weight manifest ──────────────────────────────────────────────────────────
# Each entry: (backend_dir, filename, download_url)
# Backends that auto-download (YOLO via Ultralytics, UniGaze via HuggingFace)
# are not listed here — they handle their own downloads at runtime.

MGAZE_BASE = "https://github.com/yakhyo/gaze-estimation/releases/download/weights"
GAZELLE_BASE = "https://github.com/fkryan/gazelle/releases/download/v1.0.0"

MANIFEST: list[tuple[str, str, str]] = [
    # MGaze — default ONNX model
    ("MGaze", "mobileone_s0_gaze.onnx", f"{MGAZE_BASE}/mobileone_s0_gaze.onnx"),
    ("MGaze", "mobileone_s0.pt",        f"{MGAZE_BASE}/mobileone_s0.pt"),
    # MGaze — ResNet variants (PyTorch + ONNX)
    ("MGaze", "resnet18.pt",            f"{MGAZE_BASE}/resnet18.pt"),
    ("MGaze", "resnet18_gaze.onnx",     f"{MGAZE_BASE}/resnet18_gaze.onnx"),
    ("MGaze", "resnet34.pt",            f"{MGAZE_BASE}/resnet34.pt"),
    ("MGaze", "resnet34_gaze.onnx",     f"{MGAZE_BASE}/resnet34_gaze.onnx"),
    ("MGaze", "resnet50.pt",            f"{MGAZE_BASE}/resnet50.pt"),
    ("MGaze", "resnet50_gaze.onnx",     f"{MGAZE_BASE}/resnet50_gaze.onnx"),
    # Gazelle
    ("Gazelle", "gazelle_dinov2_vitb14_inout.pt",
     f"{GAZELLE_BASE}/gazelle_dinov2_vitb14_inout.pt"),
    ("Gazelle", "gazelle_dinov2_vitb14_hub.pt",
     f"{GAZELLE_BASE}/gazelle_dinov2_vitb14_hub.pt"),
    ("Gazelle", "gazelle_dinov2_vitl14.pt",
     f"{GAZELLE_BASE}/gazelle_dinov2_vitl14.pt"),
    ("Gazelle", "gazelle_dinov2_vitl14_inout.pt",
     f"{GAZELLE_BASE}/gazelle_dinov2_vitl14_inout.pt"),
]

# L2CS does not have a public download URL — user must obtain weights
# manually. See: https://github.com/Ahmednull/L2CS-Net

# YOLO weights are auto-downloaded by Ultralytics at runtime.
# UniGaze weights are auto-downloaded from HuggingFace at runtime.
# MobileClip weights must be obtained separately.


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress indication."""
    print(f"  Downloading {dest.name} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f" {size_mb:.1f} MB")
    except Exception as e:
        print(f" FAILED: {e}")
        if dest.exists():
            dest.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description="Download MindSight model weights.")
    parser.add_argument(
        "--backend", action="append", default=None,
        help="Backend(s) to download (e.g. MGaze, Gazelle). "
             "Omit to download all. Can be specified multiple times.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file already exists.",
    )
    args = parser.parse_args()

    requested = set(args.backend) if args.backend else None
    entries = MANIFEST
    if requested:
        entries = [(b, f, u) for b, f, u in MANIFEST if b in requested]
        unknown = requested - {b for b, _, _ in MANIFEST}
        if unknown:
            print(f"Warning: unknown backend(s): {', '.join(sorted(unknown))}")
            available = sorted({b for b, _, _ in MANIFEST})
            print(f"Available: {', '.join(available)}")

    if not entries:
        print("Nothing to download.")
        return

    skipped = 0
    downloaded = 0
    failed = 0

    for backend, filename, url in entries:
        dest_dir = WEIGHTS_ROOT / backend
        dest = dest_dir / filename

        if dest.exists() and not args.force:
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [dry-run] {backend}/{filename}  <-  {url}")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            download_file(url, dest)
            downloaded += 1
        except Exception:
            failed += 1

    if args.dry_run:
        print(f"\n{len(entries)} file(s) would be downloaded.")
    else:
        parts = []
        if downloaded:
            parts.append(f"{downloaded} downloaded")
        if skipped:
            parts.append(f"{skipped} already present")
        if failed:
            parts.append(f"{failed} failed")
        print(f"\nDone: {', '.join(parts)}.")

    # Remind about manual-download backends
    print("\nNote: L2CS weights must be downloaded manually.")
    print("  See: https://github.com/Ahmednull/L2CS-Net")
    print("  Place in: Weights/L2CS/")
    print("\nYOLO and UniGaze weights are auto-downloaded at runtime.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
