"""Tests for the Intel adas-0002 head-pose-normalized backend (v1.1 W4B).

Same layering as the MPIIFaceGaze backend tests: CLI protocol, the OMZ
preprocessing geometry (eye boxes, roll alignment), and the engine chain
against the real converted ONNX (skipped when the optional weight is not
installed).  The IR->ONNX conversion itself was verified offline against
the OpenVINO runtime (max abs diff 3.6e-7 over 20 random inputs at f32).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mindsight.constants import PROJECT_ROOT
from mindsight.GazeTracking.normalized import head_pose_angles_adas
from Plugins.GazeTracking.AdasGaze.adas_gaze_backend import (
    AdasGazeEngine,
    AdasGazePlugin,
    _eye_crop,
)

WEIGHT = PROJECT_ROOT / "Weights" / "AdasGaze" / "gaze-estimation-adas-0002.onnx"

# Camera -> OMZ adas head-frame basis (kept in sync with the normalizer).
_M = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _omz_rotation(yaw, pitch, roll):
    """R = Yaw_ccw(OZ) @ Pitch_ccw(OY) @ Roll_cw(OX), the exact
    parameterization printed in the head-pose-estimation-adas-0001
    README, built in the OMZ frame."""
    y, p, r = np.radians([yaw, pitch, roll])
    ry = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    rp = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    rr = np.array([[1, 0, 0], [0, np.cos(r), np.sin(r)],
                   [0, -np.sin(r), np.cos(r)]])
    return ry @ rp @ rr


# ── Head pose angle decomposition ─────────────────────────────────────────────

def test_head_pose_angles_adas_round_trip():
    for yaw, pitch, roll in ((21.0, 2.0, -4.0), (0.0, 0.0, 0.0),
                             (-35.0, 15.0, 10.0)):
        r_cam = _M.T @ _omz_rotation(yaw, pitch, roll) @ _M
        got = head_pose_angles_adas(r_cam)
        assert got == pytest.approx([yaw, pitch, roll], abs=1e-9)


def test_head_pose_angles_frontal_is_zero():
    assert head_pose_angles_adas(np.eye(3)) == pytest.approx([0, 0, 0])


# ── Eye crop geometry ─────────────────────────────────────────────────────────

def test_eye_crop_square_60px_and_none_on_collapse():
    frame = np.zeros((200, 300, 3), np.uint8)
    frame[95:105, 140:160] = 255                      # bright strip at the eye
    crop = _eye_crop(frame, (140, 100), (160, 100), roll_deg=0.0)
    assert crop.shape == (60, 60, 3)
    assert crop.max() == 255
    # Collapsed corners -> None (OMZ treats zero-area as closed eye).
    assert _eye_crop(frame, (150, 100), (150, 100), roll_deg=0.0) is None


def test_eye_crop_deroll_straightens_eye_line():
    # An adas roll of +20 tilts the eye line to +20 deg in y-down image
    # coords (verified against the OMZ head-pose parameterization), and
    # cv2.getRotationMatrix2D(angle=+20) maps image angle t -> t - 20,
    # so de-rolling with roll_deg=+20 must straighten it.
    frame = np.zeros((300, 300, 3), np.uint8)
    c = np.array([150.0, 150.0])
    d = np.array([np.cos(np.radians(20)), np.sin(np.radians(20))])
    import cv2
    p1, p2 = c - d * 20, c + d * 20
    cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)),
             (255, 255, 255), 1)
    crop = _eye_crop(frame, tuple(p1), tuple(p2), roll_deg=20.0)
    ys, xs = np.nonzero(crop[:, :, 0])
    # After de-rolling, the lit pixels form a horizontal band.
    assert ys.max() - ys.min() <= 4
    assert xs.max() - xs.min() > 30


# ── CLI protocol ──────────────────────────────────────────────────────────────

def test_from_args_inactive_without_flag():
    assert AdasGazePlugin.from_args(SimpleNamespace()) is None
    assert AdasGazePlugin.from_args(
        SimpleNamespace(adas_gaze_model=None)) is None


def test_from_args_missing_weight_raises_with_install_hint(tmp_path):
    ns = SimpleNamespace(adas_gaze_model=str(tmp_path / "nope.onnx"))
    with pytest.raises(FileNotFoundError, match="mindsight-weights"):
        AdasGazePlugin.from_args(ns)


# ── Engine chain on the real converted model ──────────────────────────────────

pytestmark_weight = pytest.mark.skipif(
    not WEIGHT.exists(),
    reason="optional adas-0002 weight not installed in the vault")


class _SyntheticLandmarker:
    def detect(self, frame_bgr, bbox):
        from mindsight.GazeTracking.normalized import LANDMARKS
        h, w = frame_bgr.shape[:2]
        cam = np.array([[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]])
        model3d = LANDMARKS + np.array([0.0, 0.0, 0.9])
        proj = model3d @ cam.T
        return proj[:, :2] / proj[:, 2:3]


@pytestmark_weight
def test_engine_estimates_on_synthetic_pose():
    eng = AdasGazeEngine(WEIGHT, landmarker=_SyntheticLandmarker())
    frame = np.full((480, 640, 3), 128, np.uint8)
    pitch, yaw, conf = eng.estimate_in_frame(frame, (270, 190, 370, 290))
    assert conf == 1.0
    assert np.isfinite(pitch) and np.isfinite(yaw)
    assert abs(pitch) < np.pi / 2 and abs(yaw) < np.pi / 2


@pytestmark_weight
def test_engine_zero_conf_stub_when_landmarker_finds_nothing():
    class _NoneLandmarker:
        def detect(self, frame_bgr, bbox):
            return None

    eng = AdasGazeEngine(WEIGHT, landmarker=_NoneLandmarker())
    frame = np.zeros((100, 100, 3), np.uint8)
    assert eng.estimate_in_frame(frame, (10, 10, 60, 60)) == (0.0, 0.0, 0.0)
