#!/usr/bin/env python
"""
MindSight_GUI.py — Launch the MindSight graphical user interface.

Usage:  python MindSight_GUI.py
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from GUI.main_window import main

if __name__ == "__main__":
    main()
