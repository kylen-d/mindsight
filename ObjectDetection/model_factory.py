"""
ObjectDetection/model_factory.py — Shared model loading for YOLO and RetinaFace.

Both the CLI (MindSight.py) and GUI (MindSight_GUI.py) use these helpers
so that detector initialization logic is defined in one place.
"""
import sys
from pathlib import Path

from constants import PROJECT_ROOT
from ObjectDetection.object_detection import YOLOEVPDetector
from ObjectDetection.YOLO.yolo_tracking import BLACKLISTED_CLASSES, resolve_classes

# Default directory for YOLO weight files.  Models referenced by bare
# filename (e.g. "yolov8n.pt") are resolved here first; any models that
# Ultralytics auto-downloads will also be placed here.
_YOLO_WEIGHTS_DIR = PROJECT_ROOT / "Weights" / "YOLO"


def _resolve_yolo_path(model_path: str) -> str:
    """Resolve a YOLO model path, preferring the Weights/YOLO/ directory.

    If *model_path* is a bare filename (no directory component) and a file
    with that name exists under ``Weights/YOLO/``, the full path is returned
    so that Ultralytics uses the local copy instead of downloading a new one
    to the working directory.

    If the file doesn't exist yet (first run), the path is still pointed at
    ``Weights/YOLO/`` so that the auto-downloaded model lands there.
    """
    p = Path(model_path)
    # If the user supplied an absolute or relative path with directories,
    # respect it as-is — they know where their model is.
    if p.parent != Path("."):
        return model_path
    # Bare filename → resolve against the weights directory
    weights_path = _YOLO_WEIGHTS_DIR / p.name
    _YOLO_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(weights_path)


def create_yolo_detector(
    model_path: str = "yolov8n.pt",
    classes: list | None = None,
    blacklist_names: list | None = None,
    vp_file: str | None = None,
    vp_model: str = "yoloe-26l-seg.pt",
    device: str = "auto",
):
    """
    Create a YOLO (or YOLOE VP) detector and resolve class/blacklist config.

    Model paths given as bare filenames are resolved against ``Weights/YOLO/``
    so that auto-downloaded models are stored there rather than in the
    project root directory.

    Returns ``(yolo, class_ids, blacklist_set)``.
    """
    from utils.device import resolve_device
    resolved_dev = str(resolve_device(device))

    if vp_file:
        if not Path(vp_file).exists():
            raise FileNotFoundError(f"VP file not found: {vp_file}")
        resolved_vp_model = _resolve_yolo_path(vp_model)
        print(f"Loading YOLOE VP detector: {resolved_vp_model}  +  {vp_file}")
        yolo = YOLOEVPDetector(resolved_vp_model, vp_file, device=resolved_dev)
        return yolo, None, set()

    from ultralytics import YOLO
    resolved = _resolve_yolo_path(model_path)
    print(f"Loading YOLO: {resolved}")
    yolo = YOLO(resolved)
    if resolved_dev != "cpu":
        try:
            yolo.to(resolved_dev)
        except Exception:
            pass  # Ultralytics may not support this device; fall back
    try:
        yolo.set_classes(classes)
    except AttributeError:
        pass
    class_ids = resolve_classes(yolo, classes or None)
    bl = (set(BLACKLISTED_CLASSES)
          | {n.lower() for n in (blacklist_names or [])}) - {"person"}
    return yolo, class_ids, bl


def create_face_detector():
    """Create and return a RetinaFace instance."""
    _GAZE_DIR = Path(__file__).parent.parent / "GazeTracking" / "gaze-estimation"
    if str(_GAZE_DIR) not in sys.path:
        sys.path.insert(0, str(_GAZE_DIR))
    from uniface import RetinaFace
    print("Loading RetinaFace…")
    return RetinaFace()
