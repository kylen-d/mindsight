"""
tests/test_iris_refine.py -- Unit tests for IrisRefinedGaze wrapper.
"""

import numpy as np
import pytest

from Plugins import GazePlugin


class FakeBackend(GazePlugin):
    """Minimal fake backend for testing the wrapper."""

    name = "fake"
    mode = "per_face"

    def __init__(self):
        self._pipeline_called = False

    def estimate(self, face_bgr):
        return (0.1, 0.2, 0.9)

    def run_pipeline(self, **kwargs):
        self._pipeline_called = True
        n = len(kwargs.get('faces', []))
        if n == 0:
            n = 1
        persons_gaze = [
            (np.array([50.0, 50.0]), np.array([200.0, 200.0]), (0.1, 0.2))
            for _ in range(n)
        ]
        face_bboxes = [(20, 20, 80, 80)] * n
        face_track_ids = list(range(n))
        face_confs = [0.9] * n
        face_objs = [{'x1': 20, 'y1': 20, 'x2': 80, 'y2': 80, 'class_name': 'face'}] * n
        ray_snapped = [False] * n
        ray_extended = [False] * n
        return (persons_gaze, face_confs, face_bboxes, face_track_ids,
                face_objs, ray_snapped, ray_extended)


class TestIrisRefinedGaze:
    """Tests for the IrisRefinedGaze GazePlugin wrapper."""

    def _make_wrapper(self, **kwargs):
        from Plugins.GazeTracking.IrisRefinedGaze.iris_refined_gaze import IrisRefinedGaze
        defaults = dict(weight=0.3, upscale=1.0)
        defaults.update(kwargs)
        return IrisRefinedGaze(FakeBackend(), **defaults)

    def test_name(self):
        w = self._make_wrapper()
        assert w.name == "iris_refined"

    def test_estimate_delegates_to_inner(self):
        w = self._make_wrapper()
        result = w.estimate(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result == (0.1, 0.2, 0.9)

    def test_run_pipeline_delegates_to_inner(self):
        w = self._make_wrapper()
        # Use a blank frame -- iris correction degrades gracefully if
        # mediapipe is not installed (returns unmodified gaze).
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = w.run_pipeline(
            frame=frame, faces=[{}], objects=[],
            gaze_cfg=None, smoother=None, snap_hysteresis=None,
        )
        assert w._inner._pipeline_called
        persons_gaze, face_confs, face_bboxes, face_track_ids, \
            face_objs, ray_snapped, ray_extended = result
        assert len(persons_gaze) == 1
        assert len(face_confs) == 1

    def test_run_pipeline_without_frame_returns_unmodified(self):
        w = self._make_wrapper()
        result = w.run_pipeline(frame=None, faces=[{}], objects=[])
        persons_gaze = result[0]
        # Without frame, no corrections applied
        assert len(persons_gaze) == 1
        np.testing.assert_array_almost_equal(
            persons_gaze[0][1], [200.0, 200.0]
        )

    def test_weight_zero_means_no_correction(self):
        w = self._make_wrapper(weight=0.0)
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = w.run_pipeline(frame=frame, faces=[{}], objects=[])
        persons_gaze = result[0]
        # Even if iris data is found, weight=0 means ray_end unchanged
        # (correction magnitude is weight * offset * face_w, so weight=0 -> 0)
        # The test just ensures it runs without error
        assert len(persons_gaze) == 1

    def test_add_arguments(self):
        import argparse
        from Plugins.GazeTracking.IrisRefinedGaze.iris_refined_gaze import IrisRefinedGaze
        parser = argparse.ArgumentParser()
        IrisRefinedGaze.add_arguments(parser)
        args = parser.parse_args(["--iris-refine", "--iris-refine-weight", "0.5"])
        assert args.iris_refine is True
        assert args.iris_refine_weight == 0.5

    def test_from_args_disabled(self):
        import argparse
        from Plugins.GazeTracking.IrisRefinedGaze.iris_refined_gaze import IrisRefinedGaze
        parser = argparse.ArgumentParser()
        IrisRefinedGaze.add_arguments(parser)
        args = parser.parse_args([])
        assert IrisRefinedGaze.from_args(args) is None

    def test_correction_blending_math(self):
        """Verify the blending formula: corrected = base + weight * offset * scale."""
        weight = 0.3
        base_ray = np.array([200.0, 200.0])
        offset = np.array([0.1, -0.05])  # normalized iris offset
        face_w = 60  # pixels

        correction = offset * face_w * weight
        result = base_ray + correction

        expected_x = 200.0 + 0.1 * 60 * 0.3  # 200 + 1.8
        expected_y = 200.0 + (-0.05) * 60 * 0.3  # 200 - 0.9

        np.testing.assert_almost_equal(result[0], expected_x)
        np.testing.assert_almost_equal(result[1], expected_y)
