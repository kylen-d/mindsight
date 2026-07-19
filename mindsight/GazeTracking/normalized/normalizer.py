"""
GazeTracking/normalized/normalizer.py — Head-pose normalization + gaze denormalization.

Vendored ptgaze math (hysts/pytorch_mpiigaze_demo, MIT License — see
THIRD_PARTY_LICENSES.md) on numpy + cv2 only.  The normalizer warps a
camera frame into a canonical patch whose virtual camera looks straight
at the face center from a fixed distance (Zhang et al., "Revisiting Data
Normalization for Appearance-Based Gaze Estimation") — the input contract
of MPIIFaceGaze-style estimators.  The inverse rotation carries the
predicted gaze back into real camera coordinates.
"""
from __future__ import annotations

import cv2
import numpy as np

# Normalized virtual camera for MPIIFaceGaze face patches
# (ptgaze data/normalized_camera_params/mpiifacegaze.yaml).
MPIIFACEGAZE_NORM_CAMERA = np.array([[1600.0, 0.0, 112.0],
                                     [0.0, 1600.0, 112.0],
                                     [0.0, 0.0, 1.0]])
MPIIFACEGAZE_PATCH_SIZE = (224, 224)                     # (width, height)
MPIIFACEGAZE_NORM_DISTANCE = 1.0                         # meters


def _unit(v):
    return v / np.linalg.norm(v)


def compute_normalizing_rotation(center, head_rot):
    """Rotation (3, 3) aligning the camera z-axis with the face center.

    The x-axis stays parallel to the head's x-axis so the normalized
    patch has no in-plane roll (ptgaze head_pose_normalizer).
    """
    z_axis = _unit(np.asarray(center, dtype=float).ravel())
    head_x_axis = np.asarray(head_rot, dtype=float)[:, 0]
    y_axis = _unit(np.cross(z_axis, head_x_axis))
    x_axis = _unit(np.cross(y_axis, z_axis))
    return np.vstack([x_axis, y_axis, z_axis])


def normalize_image(image, camera_matrix, normalizing_rot, distance,
                    norm_camera_matrix=MPIIFACEGAZE_NORM_CAMERA,
                    patch_size=MPIIFACEGAZE_PATCH_SIZE,
                    norm_distance=MPIIFACEGAZE_NORM_DISTANCE):
    """Warp *image* into the normalized patch for a face at *distance*."""
    scale = np.diag([1.0, 1.0, norm_distance / float(distance)])
    projection = (np.asarray(norm_camera_matrix, dtype=float)
                  @ (scale @ np.asarray(normalizing_rot, dtype=float))
                  @ np.linalg.inv(np.asarray(camera_matrix, dtype=float)))
    return cv2.warpPerspective(image, projection, patch_size)


def normalized_head_rot2d(head_rot, normalizing_rot):
    """Normalized head pose (pitch, yaw) of the rotated head z-axis.

    The MPIIGaze annotation convention: for z = (R_norm @ R_head)[:, 2],
    returns (arcsin(z_y), arctan2(z_x, z_z)).
    """
    z = (np.asarray(normalizing_rot, dtype=float)
         @ np.asarray(head_rot, dtype=float))[:, 2]
    return np.array([np.arcsin(z[1]), np.arctan2(z[0], z[2])])


def gaze_angles_to_vector(angles):
    """(pitch, yaw) -> unit gaze vector, MPIIGaze convention (-z forward)."""
    pitch, yaw = angles
    return -np.array([np.cos(pitch) * np.sin(yaw),
                      np.sin(pitch),
                      np.cos(pitch) * np.cos(yaw)])


def gaze_vector_to_angles(vector):
    """Unit gaze vector -> (pitch, yaw), inverse of gaze_angles_to_vector."""
    x, y, z = vector
    return np.array([np.arcsin(-y), np.arctan2(-x, -z)])


def denormalize_gaze_vector(gaze_vector, normalizing_rot):
    """Rotate a normalized-space gaze vector back into camera coordinates."""
    # Row vector times R == R^-1 @ column vector (R is orthogonal).
    return np.asarray(gaze_vector, dtype=float) @ np.asarray(
        normalizing_rot, dtype=float)


# Camera coords (x right, y down, z away) -> the OMZ adas head frame
# (OX face-to-camera, OY image-right, OZ up; head-pose-estimation-adas-0001
# README).  Right-handed: OX x OY = OZ.
_ADAS_BASIS = np.array([[0.0, 0.0, -1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])


def head_pose_angles_adas(head_rot):
    """Head pose (yaw, pitch, roll) in DEGREES, adas-0001 convention.

    Decomposes a camera-frame head rotation (as returned by
    ``estimate_head_pose``; identity = frontal) using the OMZ
    parameterization ``R = Yaw_ccw(OZ) @ Pitch_ccw(OY) @ Roll_cw(OX)``
    from the head-pose-estimation-adas-0001 README — the angle triplet
    gaze-estimation-adas-0002 consumes.
    """
    r = _ADAS_BASIS @ np.asarray(head_rot, dtype=float) @ _ADAS_BASIS.T
    yaw = np.arctan2(r[1, 0], r[0, 0])
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    roll = np.arctan2(-r[2, 1], r[2, 2])
    return np.degrees(np.array([yaw, pitch, roll]))


def vector_to_pipeline_pitchyaw(gaze_vector):
    """Camera-frame gaze vector -> (pitch, yaw) in the MindSight ray convention.

    The pitch/yaw pipeline projects angles to 2D via
    ``pitch_yaw_to_2d(p, y) = normalize([-sin(p)cos(y), -sin(y)])``
    (mindsight.utils.geometry).  This inverse picks the (p, y) whose 2D
    projection is parallel to the vector's image-plane component
    (g_x, g_y), so normalized-backend rays render/snap/smooth exactly
    like MGaze rays.  A gaze straight at (or away from) the camera has a
    near-zero image-plane component and maps to near-zero angles, which
    the pipeline's forward-gaze dead zone already handles.
    """
    g = np.asarray(gaze_vector, dtype=float)
    n = np.linalg.norm(g)
    if n < 1e-9:
        return 0.0, 0.0
    g = g / n
    yaw = float(np.arcsin(np.clip(-g[1], -1.0, 1.0)))
    cos_yaw = float(np.cos(yaw))
    if cos_yaw < 1e-9:
        return 0.0, yaw
    pitch = float(np.arcsin(np.clip(-g[0] / cos_yaw, -1.0, 1.0)))
    return pitch, yaw
