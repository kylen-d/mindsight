"""
ObjectDetection/model_factory.py — Shared model loading for YOLO and RetinaFace.

Both the CLI (MindSight.py) and GUI (MindSight_GUI.py) use these helpers
so that detector initialization logic is defined in one place.
"""
import sys
from pathlib import Path

from mindsight.ObjectDetection.object_detection import YOLOEVPDetector
from mindsight.ObjectDetection.YOLO.yolo_tracking import BLACKLISTED_CLASSES, resolve_classes
from mindsight.weights import resolve_weight


def _resolve_yolo_path(model_path: str) -> str:
    """Resolve a YOLO model path, preferring the Weights/YOLO/ directory."""
    return str(resolve_weight("YOLO", model_path))


class NullDetector:
    """LP2 ``--no-detector``: the same duck-type as YOLO/YOLOEVPDetector
    (called per detection frame, exposes ``.names``), but finds nothing.

    Faces, gaze rays, and tip-based phenomena (tip-convergence joint
    attention, mutual gaze, aversion) run normally; object hits and object
    lock-on simply never fire.  Keeping the ``parse_dets(yolo(...),
    yolo.names, ...)`` contract intact means zero None-checks on the hot
    path."""

    names: dict = {}

    def __call__(self, frame, **kwargs):
        return []


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
    from mindsight.utils.device import resolve_device
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
    # Only narrow the vocabulary when classes were requested: set_classes(None)
    # raises TypeError inside ultralytics on YOLOE models (plain YOLO models
    # have no set_classes at all -- that AttributeError stays tolerated). A
    # YOLOE model with no classes and no VP prompt runs prompt-free on its
    # built-in vocabulary.
    if classes:
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
    _GAZE_DIR = Path(__file__).parent.parent / "GazeTracking" / "Backends" / "MGaze" / "gaze-estimation"
    if str(_GAZE_DIR) not in sys.path:
        sys.path.insert(0, str(_GAZE_DIR))
    from uniface import RetinaFace
    print("Loading RetinaFace…")
    return RetinaFace()
