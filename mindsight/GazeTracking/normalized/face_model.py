"""
GazeTracking/normalized/face_model.py — 3D face model fitting (head pose via solvePnP).

Vendored ptgaze math (hysts/pytorch_mpiigaze_demo, MIT License — see
THIRD_PARTY_LICENSES.md) reimplemented on numpy + cv2 only (no scipy):
fits the canonical MediaPipe 468-point model to detected 2D landmarks and
derives the face / eye centers that the head-pose normalizer needs.
"""
from __future__ import annotations

import cv2
import numpy as np

from .face_model_data import LANDMARKS

# Landmark index groups (ptgaze FaceModelMediaPipe).
REYE_INDICES  = np.array([33, 133])
LEYE_INDICES  = np.array([362, 263])
MOUTH_INDICES = np.array([78, 308])
NOSE_INDICES  = np.array([240, 460])


def estimate_head_pose(landmarks2d, camera_matrix):
    """Fit the canonical model to 2D landmarks; return (rot, tvec, model3d).

    Parameters
    ----------
    landmarks2d   : (468, 2) pixel coordinates from the face landmarker.
    camera_matrix : (3, 3) pinhole intrinsics for the frame the landmarks
                    live in.  Landmarks are treated as undistorted, so no
                    distortion coefficients are applied (ptgaze contract).

    Returns
    -------
    rot     : (3, 3) head rotation matrix (model -> camera).
    tvec    : (3,) head translation in camera coordinates (meters).
    model3d : (468, 3) posed model points in camera coordinates.
    """
    # solvePnP can be unstable with an unconstrained start; seed it with an
    # unrotated head 1 m in front of the camera (ptgaze does the same).
    rvec = np.zeros(3, dtype=float)
    tvec = np.array([0.0, 0.0, 1.0])
    _, rvec, tvec = cv2.solvePnP(
        LANDMARKS, np.asarray(landmarks2d, dtype=float),
        np.asarray(camera_matrix, dtype=float), np.zeros((1, 5)),
        rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    rot = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    tvec = np.asarray(tvec, dtype=float).reshape(3)
    model3d = LANDMARKS @ rot.T + tvec
    return rot, tvec, model3d


def compute_face_center(model3d, mode: str = "mpiifacegaze"):
    """Face center in camera coordinates.

    MPIIFaceGaze defines it as the mean of the six eye+mouth corner
    points; ETH-XGaze substitutes the nose points for the mouth.
    """
    extra = NOSE_INDICES if mode == "eth-xgaze" else MOUTH_INDICES
    idx = np.concatenate([REYE_INDICES, LEYE_INDICES, extra])
    return model3d[idx].mean(axis=0)


def compute_eye_centers(model3d):
    """(right, left) eye centers: the mean of each eye's corner points."""
    return (model3d[REYE_INDICES].mean(axis=0),
            model3d[LEYE_INDICES].mean(axis=0))
