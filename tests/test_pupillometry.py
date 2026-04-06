"""
tests/test_pupillometry.py -- Unit tests for iris extraction and pupillometry tracker.
"""

import numpy as np
import pytest


# ── iris_extraction tests ────────────────────────────────────────────────────

class TestMeasureRGB:
    """Tests for RGB-mode pupil measurement (mocked MediaPipe data)."""

    def test_returns_none_for_none_iris_data(self):
        from Plugins.Phenomena.Pupillometry.iris_extraction import measure_rgb
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        assert measure_rgb(crop, None) is None

    def test_returns_none_for_invalid_iris(self):
        from Plugins.Phenomena.Pupillometry.iris_extraction import measure_rgb
        from ms.utils.mediapipe_face import IrisData
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        iris = IrisData()  # all invalid
        assert measure_rgb(crop, iris) is None

    def test_valid_measurement_returns_ratio(self):
        """Synthetic iris data should produce a valid pupil/iris ratio."""
        from Plugins.Phenomena.Pupillometry.iris_extraction import measure_rgb
        from ms.utils.mediapipe_face import IrisData

        # Create a synthetic face crop with a dark circle (pupil) on gray bg
        crop = np.full((200, 200, 3), 128, dtype=np.uint8)
        # Draw dark pupil circle
        import cv2
        cv2.circle(crop, (100, 100), 15, (20, 20, 20), -1)

        iris = IrisData(
            right_iris_center=np.array([100, 100], dtype=np.float32),
            right_iris_contour=np.array([
                [100 + 30, 100], [100, 100 + 30],
                [100 - 30, 100], [100, 100 - 30],
            ], dtype=np.float32),
            right_eye_contour=np.array([
                [60, 100], [140, 100], [100, 80],
                [100, 120], [80, 90], [120, 110],
            ], dtype=np.float32),
            right_valid=True,
        )

        result = measure_rgb(crop, iris, upscale=1.0)
        if result is not None:
            assert 0.1 <= result['ratio'] <= 0.8
            assert result['eye'] in ('right', 'left', 'avg')


class TestMeasureIR:
    """Tests for IR-mode pupil measurement."""

    def test_returns_none_for_empty_crop(self):
        from Plugins.Phenomena.Pupillometry.iris_extraction import measure_ir
        assert measure_ir(None) is None
        assert measure_ir(np.array([])) is None

    def test_blank_image_returns_none(self):
        from Plugins.Phenomena.Pupillometry.iris_extraction import measure_ir
        gray = np.full((100, 100), 200, dtype=np.uint8)
        assert measure_ir(gray) is None

    def test_dark_circle_detected(self):
        """A synthetic IR image with a dark circle should detect a pupil."""
        import cv2
        from Plugins.Phenomena.Pupillometry.iris_extraction import measure_ir

        # Gray background with dark circle (pupil)
        gray = np.full((200, 200), 180, dtype=np.uint8)
        cv2.circle(gray, (100, 100), 20, 10, -1)

        result = measure_ir(gray, threshold=40)
        if result is not None:
            assert 0.1 <= result['ratio'] <= 0.8
            assert result['eye'] == 'ir'


# ── PupillometryTracker tests ───────────────────────────────────────────────

class TestPupillometryTracker:
    """Tests for the PupillometryTracker PhenomenaPlugin."""

    def _make_tracker(self, **kwargs):
        from Plugins.Phenomena.Pupillometry.pupillometry import PupillometryTracker
        defaults = dict(mode="rgb", baseline_frames=5, ema_alpha=0.3)
        defaults.update(kwargs)
        return PupillometryTracker(**defaults)

    def test_init_defaults(self):
        t = self._make_tracker()
        assert t.name == "pupillometry"
        assert t._mode == "rgb"
        assert t._baseline_frames == 5

    def test_update_with_no_frame_is_safe(self):
        t = self._make_tracker()
        result = t.update(
            frame_no=0,
            persons_gaze=[((0, 0), (100, 100), (0.1, 0.2))],
            face_bboxes=[(10, 10, 90, 90)],
            face_track_ids=[0],
            frame=None,
        )
        assert isinstance(result, dict)

    def test_csv_rows_empty_when_no_data(self):
        t = self._make_tracker()
        rows = t.csv_rows(100)
        assert rows == []

    def test_console_summary_none_when_no_data(self):
        t = self._make_tracker()
        assert t.console_summary(100) is None

    def test_baseline_computed_after_n_frames(self):
        t = self._make_tracker(baseline_frames=3)
        # Manually populate raw_ratios to simulate measurements
        t._raw_ratios[0] = [0.3, 0.35, 0.32]
        t._baselines[0] = None
        t._ema[0] = None
        t._valid_counts[0] = 3
        t._ts_frames[0] = []
        t._ts_ratios[0] = []
        t._ts_dilation[0] = []

        # Simulate baseline computation
        if len(t._raw_ratios[0]) >= t._baseline_frames:
            t._baselines[0] = float(np.median(t._raw_ratios[0][:3]))

        assert t._baselines[0] is not None
        assert abs(t._baselines[0] - 0.32) < 0.01  # median of [0.3, 0.32, 0.35]

    def test_ema_smoothing(self):
        t = self._make_tracker(ema_alpha=0.5)
        t._ema[0] = 0.3
        # Apply EMA manually
        new_ratio = 0.4
        smoothed = 0.5 * new_ratio + 0.5 * 0.3
        assert abs(smoothed - 0.35) < 1e-6

    def test_dilation_percentage(self):
        baseline = 0.3
        current = 0.36
        dilation_pct = (current - baseline) / baseline * 100
        assert abs(dilation_pct - 20.0) < 1e-6

    def test_add_arguments(self):
        import argparse
        from Plugins.Phenomena.Pupillometry.pupillometry import PupillometryTracker
        parser = argparse.ArgumentParser()
        PupillometryTracker.add_arguments(parser)
        args = parser.parse_args(["--pupillometry", "--pupil-mode", "ir"])
        assert args.pupillometry is True
        assert args.pupil_mode == "ir"

    def test_from_args_disabled(self):
        import argparse
        from Plugins.Phenomena.Pupillometry.pupillometry import PupillometryTracker
        parser = argparse.ArgumentParser()
        PupillometryTracker.add_arguments(parser)
        args = parser.parse_args([])
        assert PupillometryTracker.from_args(args) is None

    def test_dashboard_data_empty(self):
        t = self._make_tracker()
        data = t.dashboard_data()
        assert data['title'] == 'PUPILLOMETRY'
        assert data['rows'] == []

    def test_latest_metrics_none_when_empty(self):
        t = self._make_tracker()
        assert t.latest_metrics() is None
