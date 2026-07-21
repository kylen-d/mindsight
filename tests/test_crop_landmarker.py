"""Tests for the RetinaFace-crop MediaPipe landmarker wrapper (v1.1 W4B).

The real FaceLandmarker needs its 3.8 MB task asset and a real face, so
these tests drive ``CropFaceLandmarker`` through its detector seam with a
fake: what is under test is the wrapper's own logic — padded-crop
geometry, BGR->RGB handoff, crop->frame coordinate mapping, and the
nearest-candidate pick when a padded crop catches a neighboring face.
The short-range caveat (crops only, never full frames) is baked into the
API shape itself: ``detect`` requires a bbox.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mindsight.GazeTracking.normalized.landmarks import CropFaceLandmarker


def _fake_result(*landmark_sets):
    return SimpleNamespace(face_landmarks=[
        [SimpleNamespace(x=float(x), y=float(y)) for x, y in pts]
        for pts in landmark_sets
    ])


class _FakeDetector:
    def __init__(self, result):
        self.result = result
        self.images = []
        self.closed = False

    def detect(self, image):
        self.images.append(image)
        return self.result

    def close(self):
        self.closed = True


def test_padded_crop_box_pads_half_size_and_clamps():
    # Interior box: half the box size added on every side.
    assert CropFaceLandmarker.padded_crop_box(
        (100, 200, 160, 280), 1280, 720) == (70, 160, 190, 320)
    # Frame-edge box: clamped to the frame bounds.
    assert CropFaceLandmarker.padded_crop_box(
        (10, 10, 90, 90), 100, 100) == (0, 0, 100, 100)


def test_detect_maps_crop_landmarks_back_to_frame_coords():
    # One candidate whose 468 landmarks all sit at crop fraction (0.25, 0.75).
    det = _FakeDetector(_fake_result([(0.25, 0.75)] * 468))
    lmk = CropFaceLandmarker(task_path="unused", _detector=det)

    frame = np.zeros((720, 1280, 3), np.uint8)
    pts = lmk.detect(frame, (100, 200, 160, 280))
    assert pts.shape == (468, 2)
    # Padded crop is (70, 160)-(190, 320): 120x160 px.
    assert pts[0] == pytest.approx([70 + 0.25 * 120, 160 + 0.75 * 160])

    # The detector saw the padded crop, RGB-swapped and contiguous.
    crop = det.images[0]
    assert crop.shape == (160, 120, 3)
    assert crop.flags["C_CONTIGUOUS"]


def test_detect_converts_bgr_to_rgb():
    det = _FakeDetector(_fake_result([(0.5, 0.5)] * 468))
    lmk = CropFaceLandmarker(task_path="unused", _detector=det)
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :, 0] = 255                       # pure blue in BGR
    lmk.detect(frame, (25, 25, 75, 75))
    crop = det.images[0]
    assert crop[0, 0, 2] == 255 and crop[0, 0, 0] == 0   # blue now last


def test_detect_picks_candidate_nearest_requested_box():
    # Two faces in the padded crop: one centered left, one centered right.
    left = [(0.2, 0.5)] * 468
    right = [(0.8, 0.5)] * 468
    det = _FakeDetector(_fake_result(left, right))
    lmk = CropFaceLandmarker(task_path="unused", _detector=det)

    frame = np.zeros((200, 400, 3), np.uint8)
    # Requested box center (100, 100); padded crop (50,50)-(150,150).
    pts = lmk.detect(frame, (75, 75, 125, 125))
    # Left candidate maps to x = 50 + 0.2*100 = 70 (|70-100| < |130-100|).
    assert pts[0] == pytest.approx([70.0, 100.0])


def test_detect_returns_none_when_no_faces_or_degenerate_box():
    det = _FakeDetector(_fake_result())
    lmk = CropFaceLandmarker(task_path="unused", _detector=det)
    frame = np.zeros((100, 100, 3), np.uint8)
    assert lmk.detect(frame, (25, 25, 75, 75)) is None
    assert lmk.detect(frame, (50, 50, 50, 50)) is None    # zero-size box
    lmk.close()
    assert det.closed


def test_missing_task_asset_raises_with_install_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="mindsight-weights"):
        CropFaceLandmarker(task_path=tmp_path / "nope.task")
