"""
utils/mediapipe_face.py -- Lazy singleton for MediaPipe Face Mesh with iris landmarks.

Shared by Pupillometry, EyeMovement (iris mode), IrisRefinedGaze, and
BlinkDetection plugins.  ``mediapipe`` is an optional dependency -- each
consumer prints a helpful error if the package is missing.

Key landmark indices (refine_landmarks=True):
    Right iris: 468 (center), 469-472 (contour)
    Left iris:  473 (center), 474-477 (contour)
    Right eye contour: [33, 133, 159, 145, 160, 144]
    Left eye contour:  [362, 263, 386, 374, 385, 373]
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


_face_mesh_instance = None


def get_face_mesh():
    """Return the singleton MediaPipe FaceMesh instance (created on first call)."""
    global _face_mesh_instance
    if _face_mesh_instance is not None:
        return _face_mesh_instance

    try:
        import mediapipe as mp
        _fm = mp.solutions.face_mesh
    except (ImportError, AttributeError):
        raise ImportError(
            "mediapipe>=0.10 is required for iris-based features. "
            "Install it with: pip install mediapipe>=0.10"
        )

    _face_mesh_instance = _fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    return _face_mesh_instance


def extract_iris_data(face_crop: np.ndarray) -> IrisData | None:
    """
    Run MediaPipe Face Mesh on a BGR face crop and extract iris data.

    Returns ``None`` if no face is detected in the crop.
    """
    import cv2

    if face_crop is None or face_crop.size == 0:
        return None

    mesh = get_face_mesh()
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    lms = result.multi_face_landmarks[0].landmark
    h, w = face_crop.shape[:2]

    def to_px(idx):
        lm = lms[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def to_px_array(indices):
        return np.array([to_px(i) for i in indices], dtype=np.float32)

    data = IrisData()
    data.landmarks = lms

    # Right iris
    data.right_iris_center = to_px(RIGHT_IRIS_CENTER)
    data.right_iris_contour = to_px_array(RIGHT_IRIS_CONTOUR)
    data.right_eye_contour = to_px_array(RIGHT_EYE_CONTOUR)
    data.right_eye_ear_pts = to_px_array(RIGHT_EYE_EAR)
    data.right_valid = True

    # Left iris
    data.left_iris_center = to_px(LEFT_IRIS_CENTER)
    data.left_iris_contour = to_px_array(LEFT_IRIS_CONTOUR)
    data.left_eye_contour = to_px_array(LEFT_EYE_CONTOUR)
    data.left_eye_ear_pts = to_px_array(LEFT_EYE_EAR)
    data.left_valid = True

    return data
