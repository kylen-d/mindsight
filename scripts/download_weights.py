#!/usr/bin/env python
"""
scripts/download_weights.py — Download model weights for MindSight backends.

Thin wrapper over :func:`mindsight.weights.main` (the ``mindsight-weights``
console script).  The checksummed manifest (``weights_manifest.json``) and all
download/verify logic live in :mod:`mindsight.weights`; this script exists so a
source checkout can still run ``python scripts/download_weights.py``.

Usage:
    python scripts/download_weights.py --required       # the 6 required weights
    python scripts/download_weights.py --all            # every downloadable weight
    python scripts/download_weights.py --backend MGaze  # one backend
    python scripts/download_weights.py --verify-only    # check checksums only
    python scripts/download_weights.py --dry-run        # show what would download
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindsight.weights import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
