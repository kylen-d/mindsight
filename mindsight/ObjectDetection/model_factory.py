"""
ObjectDetection/model_factory.py — Shared model loading for YOLO and RetinaFace.

Both the CLI (MindSight.py) and GUI (MindSight_GUI.py) use these helpers
so that detector initialization logic is defined in one place.
"""
import contextlib
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


@contextlib.contextmanager
def _uniface_download_lock():
    """Serialize concurrent first-use RetinaFace weight downloads.

    uniface's model store is not concurrency-safe: two processes fetching
    the same backbone into ``~/.uniface`` race, and one crashes mid-verify
    on the other's partial file (hit by three concurrent gate smokes on a
    fresh HOME once r34 became the default).  An exclusive flock on a
    sidecar lock file makes the first construction finish its download
    before the others start; best-effort no-op where flock is unavailable
    (Windows), where GUI launches are single-process anyway.
    """
    try:
        import fcntl
        lock_dir = Path.home() / ".uniface"
        lock_dir.mkdir(parents=True, exist_ok=True)
        fh = open(lock_dir / ".mindsight-download.lock", "w")
    except Exception:
        yield
        return
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


# --face-model short names -> uniface RetinaFaceWeights enum values.
_FACE_MODEL_NAMES = {
    "mnet025": "retinaface_mnet025",
    "mnet050": "retinaface_mnet050",
    "mnet_v1": "retinaface_mnet_v1",
    "mnet_v2": "retinaface_mnet_v2",
    "r18":     "retinaface_r18",
    "r34":     "retinaface_r34",
}


def create_face_detector(conf_thresh: float = 0.5, input_size: int = 640,
                         model_name: str | None = None):
    """Create and return a RetinaFace instance.

    v1.1 W2.4: the confidence threshold and (square) input size are
    configurable via --face-conf / --face-input-size; the defaults are the
    uniface library defaults, so an unconfigured build is byte-unchanged.
    Faces feed BOTH the per-face gaze model and Gaze-LLE's head bboxes, so
    these are the first knobs to reach for on distant/small-face footage.
    v1.1 W3X adds --face-model to pick the backbone (r18/r34 for
    small/distant faces); None keeps the library default (mnet_v2).
    """
    _GAZE_DIR = Path(__file__).parent.parent / "GazeTracking" / "Backends" / "MGaze" / "gaze-estimation"
    if str(_GAZE_DIR) not in sys.path:
        sys.path.insert(0, str(_GAZE_DIR))
    from uniface import RetinaFace
    print("Loading RetinaFace…")
    kwargs = {}
    if model_name:
        from uniface.constants import RetinaFaceWeights
        kwargs["model_name"] = RetinaFaceWeights(
            _FACE_MODEL_NAMES.get(model_name, model_name))
    with _uniface_download_lock():
        return RetinaFace(conf_thresh=float(conf_thresh),
                          input_size=(int(input_size), int(input_size)),
                          **kwargs)
