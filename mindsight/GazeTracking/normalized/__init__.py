"""
GazeTracking/normalized — Head-pose-normalized gaze estimation core.

Shared machinery for estimator backends that consume normalized inputs
(MPIIFaceGaze face patches, Intel adas-0002 eye crops + head pose):
canonical MediaPipe face model fitting, the normalizing warp, and gaze
denormalization.  Vendored ptgaze math (MIT) — see THIRD_PARTY_LICENSES.md.
"""
from .face_model import (
    LANDMARKS,
    LEYE_INDICES,
    MOUTH_INDICES,
    NOSE_INDICES,
    REYE_INDICES,
    compute_eye_centers,
    compute_face_center,
    estimate_head_pose,
)
from .normalizer import (
    MPIIFACEGAZE_NORM_CAMERA,
    MPIIFACEGAZE_NORM_DISTANCE,
    MPIIFACEGAZE_PATCH_SIZE,
    compute_normalizing_rotation,
    denormalize_gaze_vector,
    gaze_angles_to_vector,
    gaze_vector_to_angles,
    normalize_image,
    normalized_head_rot2d,
    vector_to_pipeline_pitchyaw,
)

__all__ = [
    "LANDMARKS", "REYE_INDICES", "LEYE_INDICES", "MOUTH_INDICES",
    "NOSE_INDICES", "estimate_head_pose", "compute_face_center",
    "compute_eye_centers", "MPIIFACEGAZE_NORM_CAMERA",
    "MPIIFACEGAZE_NORM_DISTANCE", "MPIIFACEGAZE_PATCH_SIZE",
    "compute_normalizing_rotation", "normalize_image",
    "normalized_head_rot2d", "gaze_angles_to_vector",
    "gaze_vector_to_angles", "denormalize_gaze_vector",
    "vector_to_pipeline_pitchyaw",
]
