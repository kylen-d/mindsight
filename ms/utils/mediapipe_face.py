"""
utils/mediapipe_face.py -- Lazy singleton for MediaPipe Face Mesh with iris landmarks.

Shared by Pupillometry, EyeMovement (iris mode), IrisRefinedGaze, and
BlinkDetection plugins.  ``mediapipe`` is an optional dependency -- each
consumer prints a helpful error if the package is missing.

Supports both the legacy ``mp.solutions.face_mesh`` API (mediapipe <0.10.14)
and the new task-based ``FaceLandmarker`` API (mediapipe >=0.10.14).

Key landmark indices:
    Right iris: 468 (center), 469-472 (contour)
    Left iris:  473 (center), 474-477 (contour)
    Right eye contour: [33, 133, 159, 145, 160, 144]
    Left eye contour:  [362, 263, 386, 374, 385, 373]
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Landmark indices
RIGHT_IRIS_CENTER = 468
RIGHT_IRIS_CONTOUR = [469, 470, 471, 472]
LEFT_IRIS_CENTER = 473
LEFT_IRIS_CONTOUR = [474, 475, 476, 477]

RIGHT_EYE_CONTOUR = [33, 133, 159, 145, 160, 144]
LEFT_EYE_CONTOUR = [362, 263, 386, 374, 385, 373]

# EAR landmarks (6-point model per eye)
RIGHT_EYE_EAR = [33, 160, 159, 133, 144, 145]  # P1-P6
LEFT_EYE_EAR = [362, 385, 386, 263, 373, 374]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_FILENAME = "face_landmarker.task"


@dataclass
class IrisData:
    """Extracted iris and eye contour data from a single face crop."""
    right_iris_center: np.ndarray | None = None
    left_iris_center: np.ndarray | None = None
    right_iris_contour: np.ndarray | None = None
    left_iris_contour: np.ndarray | None = None
    right_eye_contour: np.ndarray | None = None
    left_eye_contour: np.ndarray | None = None
    right_eye_ear_pts: np.ndarray | None = None
    left_eye_ear_pts: np.ndarray | None = None
    right_valid: bool = False
    left_valid: bool = False
    landmarks: list = field(default_factory=list)


# ── Singleton state ──────────────────────────────────────────────────────────

_instance = None
_api_mode: str | None = None  # "legacy" or "tasks"


def _ensure_model_file() -> str:
    """Download the FaceLandmarker .task model if not cached. Return path."""
    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / _MODEL_FILENAME
    if not model_path.exists():
        print(f"Downloading MediaPipe face_landmarker model to {model_path}...")
        urllib.request.urlretrieve(_MODEL_URL, str(model_path))
    return str(model_path)


def get_face_mesh():
    """Return the singleton face landmark detector (created on first call).

    Tries the new task-based API first, falls back to legacy mp.solutions.
    """
    global _instance, _api_mode
    if _instance is not None:
        return _instance

    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            "mediapipe is required for iris-based features. "
            "Install it with: pip install mediapipe>=0.10"
        )

    # Try new task-based API first (mediapipe >= 0.10.14)
    try:
        from mediapipe.tasks.python import vision, BaseOptions

        model_path = _ensure_model_file()
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        _instance = vision.FaceLandmarker.create_from_options(opts)
        _api_mode = "tasks"
        return _instance
    except (ImportError, AttributeError, Exception):
        pass

    # Fall back to legacy API (mediapipe < 0.10.14)
    try:
        _fm = mp.solutions.face_mesh
        _instance = _fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        _api_mode = "legacy"
        return _instance
    except AttributeError:
        raise ImportError(
            "mediapipe is installed but neither the task-based API nor "
            "the legacy solutions API is available. "
            "Try: pip install --upgrade mediapipe>=0.10"
        )


def extract_iris_data(face_crop: np.ndarray) -> IrisData | None:
    """
    Run MediaPipe Face Mesh on a BGR face crop and extract iris data.

    Returns ``None`` if no face is detected in the crop.
    """
    import cv2

    if face_crop is None or face_crop.size == 0:
        return None

    detector = get_face_mesh()
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    h, w = face_crop.shape[:2]

    if _api_mode == "tasks":
        return _extract_tasks_api(detector, rgb, h, w)
    else:
        return _extract_legacy_api(detector, rgb, h, w)


def _extract_tasks_api(landmarker, rgb, h, w) -> IrisData | None:
    """Extract iris data using the new task-based FaceLandmarker API."""
    import mediapipe as mp

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    if len(lms) < 478:
        return None  # no iris landmarks

    def to_px(idx):
        lm = lms[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def to_px_array(indices):
        return np.array([to_px(i) for i in indices], dtype=np.float32)

    data = IrisData()
    data.landmarks = lms

    data.right_iris_center = to_px(RIGHT_IRIS_CENTER)
    data.right_iris_contour = to_px_array(RIGHT_IRIS_CONTOUR)
    data.right_eye_contour = to_px_array(RIGHT_EYE_CONTOUR)
    data.right_eye_ear_pts = to_px_array(RIGHT_EYE_EAR)
    data.right_valid = True

    data.left_iris_center = to_px(LEFT_IRIS_CENTER)
    data.left_iris_contour = to_px_array(LEFT_IRIS_CONTOUR)
    data.left_eye_contour = to_px_array(LEFT_EYE_CONTOUR)
    data.left_eye_ear_pts = to_px_array(LEFT_EYE_EAR)
    data.left_valid = True

    return data


def _extract_legacy_api(mesh, rgb, h, w) -> IrisData | None:
    """Extract iris data using the legacy mp.solutions.face_mesh API."""
    result = mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    lms = result.multi_face_landmarks[0].landmark

    def to_px(idx):
        lm = lms[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def to_px_array(indices):
        return np.array([to_px(i) for i in indices], dtype=np.float32)

    data = IrisData()
    data.landmarks = lms

    data.right_iris_center = to_px(RIGHT_IRIS_CENTER)
    data.right_iris_contour = to_px_array(RIGHT_IRIS_CONTOUR)
    data.right_eye_contour = to_px_array(RIGHT_EYE_CONTOUR)
    data.right_eye_ear_pts = to_px_array(RIGHT_EYE_EAR)
    data.right_valid = True

    data.left_iris_center = to_px(LEFT_IRIS_CENTER)
    data.left_iris_contour = to_px_array(LEFT_IRIS_CONTOUR)
    data.left_eye_contour = to_px_array(LEFT_EYE_CONTOUR)
    data.left_eye_ear_pts = to_px_array(LEFT_EYE_EAR)
    data.left_valid = True

    return data
