"""
tests/conftest.py — pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so all imports resolve
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
