"""
constants.py — Project-wide constants shared across all MindSight modules.

Centralizes tuning values, colour palettes, and path definitions so that
adjustments only need to be made in one place.  Organized by domain:

    - Paths & file types
    - Colour palettes (BGR format for OpenCV)
    - Gaze estimation thresholds
    - Dashboard & overlay drawing
    - UI labels & strings
    - Heatmap rendering
"""
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Paths & file types
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent

# Recognized still-image extensions — used to distinguish image vs video input.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Default root for all run-time outputs (video, CSV, heatmaps).
OUTPUTS_ROOT = PROJECT_ROOT / "Outputs"


# ═══════════════════════════════════════════════════════════════════════════════
# Colour palettes  (BGR format — OpenCV convention)
# ═══════════════════════════════════════════════════════════════════════════════

# Per-class colour cycle for detected objects (wraps after 20 classes).
PALETTE_BGR = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (147, 69, 52),  (255, 115, 100),(236, 24, 0),   (255, 56, 132),
    (133, 0, 82),   (255, 56, 203), (200, 149, 255),(199, 55, 255),
]


def get_colour(class_id: int) -> tuple:
    """Return a BGR colour for a YOLO class ID (cycles through the palette)."""
    return PALETTE_BGR[class_id % len(PALETTE_BGR)]


# ═══════════════════════════════════════════════════════════════════════════════
# Gaze estimation thresholds
# ═══════════════════════════════════════════════════════════════════════════════

EYE_CONF_THRESH = 0.20       # minimum face-detection score to use eye-centre origin
SMOOTH_ALPHA    = 0.30       # EMA alpha for temporal gaze smoother (lower = more lag)
RAY_EXT_LENGTH  = 8000.0     # default ray extension length in pixels (face-to-face tests)

# Confidence-ray scaling bounds — maps gaze confidence [0, 1] to a ray-length
# multiplier so high-confidence estimates produce longer rays.
CR_MIN = 0.3                 # multiplier at confidence = 0
CR_MAX = 2.5                 # multiplier at confidence = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard & overlay drawing
# ═══════════════════════════════════════════════════════════════════════════════

DASH_WIDTH      = 280         # side-panel width in pixels
DASH_FONT_SCALE = 0.40       # font scale for panel text (OpenCV putText)
DASH_PADDING    = 5           # internal padding in pixels

# Overlay blend weights for the semi-transparent gaze cone fill.
OVERLAY_BLEND_ALPHA = 0.18   # cone polygon opacity
OVERLAY_BLEND_BETA  = 0.82   # background retention (must equal 1 - alpha)

# Dwell-progress indicator (circular arc drawn at gaze origin during lock-on).
DWELL_INDICATOR_RADIUS = 14  # radius of the dwell arc in pixels
DWELL_MIN_FRACTION     = 0.05  # minimum dwell fraction before the arc appears

# Bounding-box drawing defaults.
BOX_THICKNESS   = 2           # default rectangle thickness
LABEL_PAD_X     = 2           # horizontal padding inside label background
LABEL_PAD_Y     = 4           # vertical padding inside label background


# ═══════════════════════════════════════════════════════════════════════════════
# UI labels & strings
# ═══════════════════════════════════════════════════════════════════════════════

UI_ARROW_LEFT   = "\u2190"    # ← used next to object labels to show looker
UI_ARROW_RIGHT  = "\u2192"    # → used in console output for gaze hits
UI_LABEL_JOINT  = "JOINT"     # tag for joint-attention objects
UI_LABEL_LOCKED = "LOCKED"    # tag for gaze-locked objects
UI_LABEL_GHOST  = "[ghost]"   # suffix for persisted (stale) detections
UI_LABEL_CONVERGE = "CONVERGE"  # prefix for gaze convergence annotations
UI_DEGREE_SIGN  = "\u00b0"    # ° symbol appended to angle readouts


# ═══════════════════════════════════════════════════════════════════════════════
# Heatmap rendering
# ═══════════════════════════════════════════════════════════════════════════════

HEATMAP_SIGMA   = 40          # Gaussian sigma in pixels — controls heat-blob spread
HEATMAP_ALPHA   = 0.65        # maximum heatmap blend weight [0–1] over background
