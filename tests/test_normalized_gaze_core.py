"""Unit tests for the head-pose-normalized gaze core (v1.1 W4B).

The vendored ptgaze math (mindsight.GazeTracking.normalized) is welded to
two independent anchors:

1. A synthetic-pose round trip: a known head pose posed through the
   canonical face model and projected through a pinhole camera must be
   recovered by ``estimate_head_pose`` and produce the PINNED
   normalization outputs below.  The pins were computed with an
   independent scipy.spatial.transform reference implementation (the
   validated W4B prototype route), so a cv2/numpy transcription slip in
   the vendored core cannot silently pass.
2. Algebraic invariants (rotation orthonormality, angle/vector round
   trips, warp geometry) that hold for any correct implementation.

Pure math — no models, no video, no mediapipe.
"""

import cv2
import numpy as np
import pytest

from mindsight.GazeTracking.normalized import (
    LANDMARKS,
    MPIIFACEGAZE_NORM_CAMERA,
    compute_eye_centers,
    compute_face_center,
    compute_normalizing_rotation,
    denormalize_gaze_vector,
    estimate_head_pose,
    gaze_angles_to_vector,
    gaze_vector_to_angles,
    normalize_image,
    normalized_head_rot2d,
    vector_to_pipeline_pitchyaw,
)
from mindsight.utils.geometry import pitch_yaw_to_2d

# ── Synthetic scene shared by the pinned tests ────────────────────────────────
# Head pose: intrinsic XYZ Euler (5°, 15°, -3°), 0.9 m in front of a
# 1280x720 pinhole camera.  R0 is the explicit rotation matrix (pinned
# from the scipy reference) so the test does not depend on any Euler
# convention helper.
R0 = np.array([
    [0.9646020585, 0.0505526518, 0.2588190451],
    [-0.0296101504, 0.9960100197, -0.0841859828],
    [-0.2620421869, 0.0735423015, 0.9622501869],
])
T0 = np.array([0.03, -0.02, 0.9])
CAM = np.array([[1280.0, 0.0, 640.0],
                [0.0, 1280.0, 360.0],
                [0.0, 0.0, 1.0]])


def _project(model3d):
    proj = model3d @ CAM.T
    return proj[:, :2] / proj[:, 2:3]


@pytest.fixture(scope="module")
def synthetic_landmarks():
    model3d = LANDMARKS @ R0.T + T0
    return _project(model3d)


# ── Face model data ───────────────────────────────────────────────────────────

def test_landmarks_shape_and_scale():
    assert LANDMARKS.shape == (468, 3)
    # Canonical model contract: outer eye corner distance is 90 mm.
    outer = np.linalg.norm(LANDMARKS[33] - LANDMARKS[263])
    assert outer == pytest.approx(0.0889, abs=2e-4)
    # Spot-pin one vendored row against the upstream source.
    assert LANDMARKS[33] == pytest.approx(
        [-0.04445859, -0.03790856, 0.04302182], abs=1e-8)


# ── Head pose recovery ────────────────────────────────────────────────────────

def test_estimate_head_pose_recovers_synthetic_pose(synthetic_landmarks):
    rot, tvec, model3d = estimate_head_pose(synthetic_landmarks, CAM)
    assert rot == pytest.approx(R0, abs=1e-5)
    assert tvec == pytest.approx(T0, abs=1e-5)
    assert model3d == pytest.approx(LANDMARKS @ R0.T + T0, abs=1e-5)


def test_face_and_eye_centers(synthetic_landmarks):
    rot, tvec, model3d = estimate_head_pose(synthetic_landmarks, CAM)
    center = compute_face_center(model3d, mode="mpiifacegaze")
    # Pinned from the scipy reference implementation.
    assert center == pytest.approx(
        [0.0391510336, -0.0376689683, 0.9356823716], abs=1e-6)
    # Direct definition: mean of the six eye+mouth corner points.
    idx = np.array([33, 133, 362, 263, 78, 308])
    assert center == pytest.approx(model3d[idx].mean(axis=0), abs=1e-12)
    # ETH-XGaze mode substitutes nose for mouth.
    idx_x = np.array([33, 133, 362, 263, 240, 460])
    assert compute_face_center(model3d, mode="eth-xgaze") == pytest.approx(
        model3d[idx_x].mean(axis=0), abs=1e-12)
    reye, leye = compute_eye_centers(model3d)
    assert reye == pytest.approx(model3d[[33, 133]].mean(axis=0), abs=1e-12)
    assert leye == pytest.approx(model3d[[362, 263]].mean(axis=0), abs=1e-12)


# ── Normalizing rotation ──────────────────────────────────────────────────────

def test_normalizing_rotation_pinned_and_orthonormal(synthetic_landmarks):
    rot, tvec, model3d = estimate_head_pose(synthetic_landmarks, CAM)
    center = compute_face_center(model3d)
    nr = compute_normalizing_rotation(center, rot)
    # Pinned from the scipy reference implementation.
    assert nr == pytest.approx(np.array([
        [0.998281472, -0.0394237738, -0.0433574529],
        [0.0411000444, 0.9984139835, 0.038474718],
        [0.0417718687, -0.0401905914, 0.9983185],
    ]), abs=1e-6)
    # Proper rotation.
    assert nr @ nr.T == pytest.approx(np.eye(3), abs=1e-10)
    assert np.linalg.det(nr) == pytest.approx(1.0, abs=1e-10)
    # It aims the z-axis at the face center...
    aligned = nr @ center
    assert aligned[:2] == pytest.approx([0.0, 0.0], abs=1e-10)
    assert aligned[2] == pytest.approx(np.linalg.norm(center), abs=1e-10)
    # ...and keeps the head x-axis in the xz-plane (no roll).
    assert float(nr[1] @ rot[:, 0]) == pytest.approx(0.0, abs=1e-10)


def test_normalized_head_rot2d_pinned(synthetic_landmarks):
    rot, _, model3d = estimate_head_pose(synthetic_landmarks, CAM)
    nr = compute_normalizing_rotation(compute_face_center(model3d), rot)
    hr2d = normalized_head_rot2d(rot, nr)
    # Pinned from the scipy reference implementation.
    assert hr2d == pytest.approx([-0.0364007216, 0.2219357258], abs=1e-6)


# ── Patch warp geometry ───────────────────────────────────────────────────────

def test_normalize_image_maps_face_center_to_patch_center(synthetic_landmarks):
    rot, _, model3d = estimate_head_pose(synthetic_landmarks, CAM)
    center = compute_face_center(model3d)
    nr = compute_normalizing_rotation(center, rot)

    # Paint a dot at the face center's image projection; after the warp it
    # must land on the normalized camera's principal point (112, 112).
    img = np.zeros((720, 1280, 3), np.uint8)
    cpx = _project(center[None])[0]
    cv2.circle(img, (int(round(cpx[0])), int(round(cpx[1]))), 3,
               (255, 255, 255), -1)
    patch = normalize_image(img, CAM, nr, np.linalg.norm(center))
    assert patch.shape == (224, 224, 3)
    ys, xs = np.nonzero(patch[:, :, 0])
    assert xs.mean() == pytest.approx(112.0, abs=2.0)
    assert ys.mean() == pytest.approx(112.0, abs=2.0)
    assert MPIIFACEGAZE_NORM_CAMERA[0, 2] == 112.0


# ── Gaze angle / vector conversions ───────────────────────────────────────────

def test_gaze_angle_vector_round_trip():
    for angles in ([0.1, -0.2], [0.0, 0.0], [-0.4, 0.35]):
        vec = gaze_angles_to_vector(angles)
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-12)
        assert gaze_vector_to_angles(vec) == pytest.approx(angles, abs=1e-12)
    # Zero angles look straight down the -z axis (toward the camera).
    assert gaze_angles_to_vector([0.0, 0.0]) == pytest.approx([0, 0, -1])


def test_denormalize_gaze_vector_pinned(synthetic_landmarks):
    rot, _, model3d = estimate_head_pose(synthetic_landmarks, CAM)
    nr = compute_normalizing_rotation(compute_face_center(model3d), rot)
    gcam = denormalize_gaze_vector(gaze_angles_to_vector([0.1, -0.2]), nr)
    # Pinned from the scipy reference implementation.
    assert gcam == pytest.approx(
        [0.1524992538, -0.0682755729, -0.9859424039], abs=1e-6)
    # Inverse relationship: rotating back recovers the normalized vector.
    assert nr @ gcam == pytest.approx(gaze_angles_to_vector([0.1, -0.2]),
                                      abs=1e-10)


# ── MindSight ray-convention bridge ───────────────────────────────────────────

def test_vector_to_pipeline_pitchyaw_projects_parallel():
    rng = np.random.default_rng(42)
    for _ in range(50):
        g = rng.normal(size=3)
        if abs(g[2]) < 0.1 or np.hypot(g[0], g[1]) < 1e-3:
            continue
        pitch, yaw = vector_to_pipeline_pitchyaw(g)
        d = pitch_yaw_to_2d(pitch, yaw)
        img = np.array([g[0], g[1]]) / np.linalg.norm([g[0], g[1]])
        assert d == pytest.approx(img, abs=1e-9)


def test_vector_to_pipeline_pitchyaw_degenerate_cases():
    # Straight at / away from the camera: no image-plane component; the
    # zero angles fall into the pipeline's forward-gaze dead zone.
    assert vector_to_pipeline_pitchyaw([0, 0, -1.0]) == pytest.approx([0, 0])
    assert vector_to_pipeline_pitchyaw([0, 0, 1.0]) == pytest.approx([0, 0])
    assert vector_to_pipeline_pitchyaw([0, 0, 0]) == pytest.approx([0, 0])
    # Straight down: yaw saturates, cos(yaw) -> 0 guard path.
    pitch, yaw = vector_to_pipeline_pitchyaw([0, 1.0, 0])
    assert yaw == pytest.approx(-np.pi / 2, abs=1e-9)
    d = pitch_yaw_to_2d(pitch, yaw)
    assert d == pytest.approx([0.0, 1.0], abs=1e-9)
