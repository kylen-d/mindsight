"""Tests for the MPIIFaceGaze head-pose-normalized backend (v1.1 W4B).

Three layers:

1. Plugin CLI protocol — activation flag gating and the missing-weight
   error message.
2. The ``estimate_in_frame`` pipeline seam — a per-face backend that
   declares frame-context estimation must receive the FULL frame + bbox
   from ``run_pitchyaw_pipeline`` (the crop-only ``estimate`` must not be
   called), and backends without the seam keep the crop path untouched.
3. The engine chain against the real vault checkpoint (skipped when the
   optional weight is not installed): a strict state-dict load pins the
   vendored architecture to the shipped weights, and a synthetic-pose
   fake landmarker drives the full normalize -> model -> denormalize ->
   ray-convention chain without MediaPipe.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mindsight.constants import PROJECT_ROOT
from Plugins.GazeTracking.MPIIFaceGaze.mpiifacegaze_backend import (
    MPIIFaceGazeEngine,
    MPIIFaceGazePlugin,
)

WEIGHT = PROJECT_ROOT / "Weights" / "MPIIFaceGaze" / "mpiifacegaze_resnet_simple.pth"


# ── CLI protocol ──────────────────────────────────────────────────────────────

def test_from_args_inactive_without_flag():
    assert MPIIFaceGazePlugin.from_args(SimpleNamespace()) is None
    assert MPIIFaceGazePlugin.from_args(
        SimpleNamespace(mpiifacegaze_model=None)) is None


def test_from_args_missing_weight_raises_with_install_hint(tmp_path):
    ns = SimpleNamespace(mpiifacegaze_model=str(tmp_path / "nope.pth"),
                         device="cpu")
    with pytest.raises(FileNotFoundError, match="mindsight-weights"):
        MPIIFaceGazePlugin.from_args(ns)


# ── estimate_in_frame pipeline seam ───────────────────────────────────────────

def _gaze_cfg():
    from mindsight.pipeline_config import GazeConfig
    return GazeConfig()


def _run(engine, frame, faces):
    from mindsight.GazeTracking.pitchyaw_pipeline import run_pitchyaw_pipeline
    return run_pitchyaw_pipeline(frame=frame, faces=faces, gaze_eng=engine,
                                 objects=[], gaze_cfg=_gaze_cfg())


def test_pipeline_prefers_estimate_in_frame():
    calls = []

    class _FrameEngine:
        def estimate_in_frame(self, frame, bbox):
            calls.append((frame.shape, bbox))
            return 0.3, 0.2, 0.9

        def estimate(self, crop):                    # must NOT be used
            raise AssertionError("crop path used despite estimate_in_frame")

    frame = np.zeros((200, 300, 3), np.uint8)
    faces = [{"bbox": [40, 50, 90, 110, 0.99]}]
    persons_gaze, confs, bboxes, *_ = _run(_FrameEngine(), frame, faces)
    assert calls == [((200, 300, 3), (40, 50, 90, 110))]
    assert len(persons_gaze) == 1
    assert confs == [0.9]
    assert bboxes == [(40, 50, 90, 110)]


def test_core_estimate_pitchyaw_prefers_clean_frame():
    """Path A (core ray forming) must hand estimate_in_frame the
    pre-annotation frame: ctx['frame'] carries drawn person boxes by the
    time gaze runs, and box edges inside a face crop break the
    landmarker."""
    from mindsight.GazeTracking.gaze_pipeline import _estimate_pitchyaw

    seen = []

    class _FrameEngine:
        def estimate_in_frame(self, frame, bbox):
            seen.append((frame[0, 0, 0], bbox))
            return 0.3, 0.2, 0.9

        def estimate(self, crop):
            raise AssertionError("crop path used despite estimate_in_frame")

    drawn = np.full((200, 300, 3), 255, np.uint8)      # "annotated" frame
    clean = np.zeros((200, 300, 3), np.uint8)          # pre-annotation copy
    faces = [{"bbox": [40, 50, 90, 110, 0.99]}]
    (raw_faces, smoothed, _fw, confs, bboxes, _tids, _objs) = \
        _estimate_pitchyaw(drawn, faces, _FrameEngine(), smoother=None,
                           clean_frame=clean)
    assert seen == [(0, (40, 50, 90, 110))]            # clean pixels won
    assert confs == [0.9]
    assert bboxes == [(40, 50, 90, 110)]
    # Without a clean frame it falls back to the (annotated) frame.
    _estimate_pitchyaw(drawn, faces, _FrameEngine(), smoother=None)
    assert seen[-1][0] == 255


def test_pipeline_crop_path_unchanged_without_seam():
    crops = []

    class _CropEngine:
        def estimate(self, crop):
            crops.append(crop.shape)
            return 0.3, 0.2, 0.8

    frame = np.zeros((200, 300, 3), np.uint8)
    persons_gaze, confs, *_ = _run(
        _CropEngine(), frame, [{"bbox": [40, 50, 90, 110, 0.99]}])
    assert crops == [(60, 50, 3)]
    assert confs == [0.8]


# ── Engine chain on the real checkpoint ───────────────────────────────────────

pytestmark_weight = pytest.mark.skipif(
    not WEIGHT.exists(),
    reason="optional MPIIFaceGaze weight not installed in the vault")


class _SyntheticLandmarker:
    """Returns the canonical face model projected at a known pose."""

    def __init__(self):
        self.calls = []

    def detect(self, frame_bgr, bbox):
        from mindsight.GazeTracking.normalized import LANDMARKS
        self.calls.append(bbox)
        h, w = frame_bgr.shape[:2]
        cam = np.array([[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]])
        model3d = LANDMARKS + np.array([0.0, 0.0, 0.9])   # head-on, 0.9 m
        proj = model3d @ cam.T
        return proj[:, :2] / proj[:, 2:3]


@pytestmark_weight
def test_engine_loads_checkpoint_strict_and_estimates():
    lmk = _SyntheticLandmarker()
    eng = MPIIFaceGazeEngine(WEIGHT, device="cpu", landmarker=lmk)

    frame = np.full((480, 640, 3), 128, np.uint8)
    pitch, yaw, conf = eng.estimate_in_frame(frame, (270, 190, 370, 290))
    assert conf == 1.0
    assert np.isfinite(pitch) and np.isfinite(yaw)
    # A face patch of flat gray is out-of-distribution, but the angles
    # must stay in the sane range the denormalization guarantees.
    assert abs(pitch) < np.pi / 2 and abs(yaw) < np.pi / 2
    assert lmk.calls == [(270, 190, 370, 290)]


@pytestmark_weight
def test_engine_zero_conf_stub_when_landmarker_finds_nothing():
    class _NoneLandmarker:
        def detect(self, frame_bgr, bbox):
            return None

    eng = MPIIFaceGazeEngine(WEIGHT, device="cpu",
                             landmarker=_NoneLandmarker())
    frame = np.zeros((100, 100, 3), np.uint8)
    assert eng.estimate_in_frame(frame, (10, 10, 60, 60)) == (0.0, 0.0, 0.0)


@pytestmark_weight
def test_plugin_from_args_activates_with_bare_filename(monkeypatch):
    ns = SimpleNamespace(mpiifacegaze_model="mpiifacegaze_resnet_simple.pth",
                         device="cpu")
    created = {}

    def _fake_engine(path, device="auto"):
        created["path"] = Path(path)
        created["device"] = device
        return SimpleNamespace()

    import sys
    mod = sys.modules[MPIIFaceGazePlugin.__module__]
    monkeypatch.setattr(mod, "MPIIFaceGazeEngine", _fake_engine)
    plugin = MPIIFaceGazePlugin.from_args(ns)
    assert plugin is not None
    assert created["path"] == WEIGHT
    assert created["device"] == "cpu"
