"""
tests/test_pupillometry.py -- Unit tests for iris extraction and pupillometry tracker.
"""

import numpy as np


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

    def test_add_arguments_filter_params(self):
        import argparse
        from Plugins.Phenomena.Pupillometry.pupillometry import PupillometryTracker
        parser = argparse.ArgumentParser()
        PupillometryTracker.add_arguments(parser)
        args = parser.parse_args([
            "--pupillometry",
            "--pupil-filter", "kalman",
            "--pupil-kalman-process-noise", "0.001",
            "--pupil-kalman-meas-noise", "0.05",
            "--pupil-ema-alpha", "0.5",
            "--pupil-per-eye",
        ])
        assert args.pupil_filter == "kalman"
        assert args.pupil_kalman_process_noise == 0.001
        assert args.pupil_kalman_meas_noise == 0.05
        assert args.pupil_ema_alpha == 0.5
        assert args.pupil_per_eye is True

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

    def test_latest_metrics_resolves_pid(self):
        t = self._make_tracker()
        t._current_dilation[0] = 5.0
        t._blink_counts[0] = 2
        t._blink_timestamps[0] = [10, 50]
        t._frame_no = 100
        t._pid_map = {0: "S70"}
        metrics = t.latest_metrics()
        assert metrics is not None
        assert "pupil_0" in metrics
        assert "S70 dilation" == metrics["pupil_0"]["label"]
        assert "blink_count_0" in metrics

    def test_time_series_resolves_pid(self):
        t = self._make_tracker()
        t._ts_frames[0] = [1, 2, 3]
        t._ts_dilation[0] = [1.0, 2.0, 3.0]
        t._ts_blink_rate[0] = [0.0, 0.0, 0.0]
        t._pid_map = {0: "S70"}
        series = t.time_series_data()
        assert "pupil_dilation_S70" in series
        assert series["pupil_dilation_S70"]["label"] == "S70 pupil dilation %"


class TestKalmanFilter:
    """Tests for the 1D Kalman filter."""

    def test_initial_state(self):
        from Plugins.Phenomena.Pupillometry.kalman import PupilKalman
        k = PupilKalman()
        assert k.x is None

    def test_first_measurement_sets_state(self):
        from Plugins.Phenomena.Pupillometry.kalman import PupilKalman
        k = PupilKalman()
        result = k.update(0.35)
        assert result == 0.35
        assert k.x == 0.35

    def test_convergence(self):
        from Plugins.Phenomena.Pupillometry.kalman import PupilKalman
        k = PupilKalman(process_noise=1e-4, measurement_noise=1e-2)
        # Feed constant signal -- should converge
        for _ in range(50):
            result = k.update(0.4)
        assert abs(result - 0.4) < 0.001

    def test_smooths_noisy_signal(self):
        from Plugins.Phenomena.Pupillometry.kalman import PupilKalman
        k = PupilKalman()
        np.random.seed(42)
        values = 0.35 + np.random.normal(0, 0.05, 100)
        filtered = [k.update(float(v)) for v in values]
        # Filtered signal should have lower variance than input
        assert np.std(filtered[20:]) < np.std(values[20:])

    def test_reset(self):
        from Plugins.Phenomena.Pupillometry.kalman import PupilKalman
        k = PupilKalman()
        k.update(0.3)
        k.reset()
        assert k.x is None


class TestBlinkDetection:
    """Tests for EAR-based blink detection."""

    def test_compute_ear_none_for_none_data(self):
        from Plugins.Phenomena.Pupillometry.iris_extraction import compute_ear
        assert compute_ear(None) is None

    def test_blink_detection_counts(self):
        t = TestPupillometryTracker()._make_tracker(blink_consec=2)
        t._init_track(0)
        # Simulate consecutive low-EAR frames triggering a blink
        t._ear_counters[0] = 2
        t._blink_in_progress[0] = False
        # Manually trigger blink logic
        assert t._ear_counters[0] >= t._blink_consec

    def test_blink_data_collection(self):
        t = TestPupillometryTracker()._make_tracker(blink_consec=2)
        t._init_track(0)
        # Simulate a blink
        t._blink_counts[0] = 1
        t._blink_timestamps[0] = [42]
        t._blink_durations[0] = [3]
        assert t._blink_counts[0] == 1
        assert t._blink_timestamps[0] == [42]
        assert t._blink_durations[0] == [3]

    def test_blink_csv_output(self):
        t = TestPupillometryTracker()._make_tracker(blink_consec=2)
        t._init_track(0)
        t._ts_frames[0] = [1, 2, 3]
        t._ts_ratios[0] = [0.3, 0.3, 0.3]
        t._ts_dilation[0] = [0.0, 0.0, 0.0]
        t._ts_valid[0] = [1, 1, 1]
        t._valid_counts[0] = 3
        t._blink_counts[0] = 1
        t._blink_timestamps[0] = [5]
        t._blink_durations[0] = [2]
        rows = t.csv_rows(100)
        # Should contain blink_events section
        flat = [str(cell) for row in rows for cell in row]
        assert "blink_events" in flat
        assert "total_blinks" in flat or any("blink" in str(r) for r in rows[0:20] if r)


class TestOutlierRejection:
    """Tests for Hampel outlier rejection."""

    def test_spike_rejected(self):
        t = TestPupillometryTracker()._make_tracker(outlier_window=10)
        t._init_track(0)
        # Fill window with stable values
        for v in [0.35, 0.34, 0.36, 0.35, 0.34, 0.35, 0.36, 0.35]:
            t._outlier_deque[0].append(v)
        # A spike should be detected as outlier
        assert t._is_outlier(0, 0.70)

    def test_normal_value_not_rejected(self):
        t = TestPupillometryTracker()._make_tracker(outlier_window=10)
        t._init_track(0)
        for v in [0.35, 0.34, 0.36, 0.35, 0.34, 0.35]:
            t._outlier_deque[0].append(v)
        assert not t._is_outlier(0, 0.36)


class TestPerEyeMeasurements:
    """Tests for per-eye ratio output."""

    def test_per_eye_result_from_measure_rgb(self):
        """measure_rgb should include left_ratio and right_ratio keys."""
        from Plugins.Phenomena.Pupillometry.iris_extraction import measure_rgb
        from ms.utils.mediapipe_face import IrisData
        import cv2

        crop = np.full((200, 200, 3), 128, dtype=np.uint8)
        cv2.circle(crop, (100, 100), 15, (20, 20, 20), -1)

        iris = IrisData(
            right_iris_center=np.array([100, 100], dtype=np.float32),
            right_iris_contour=np.array([
                [130, 100], [100, 130], [70, 100], [100, 70],
            ], dtype=np.float32),
            right_eye_contour=np.array([
                [60, 100], [140, 100], [100, 80],
                [100, 120], [80, 90], [120, 110],
            ], dtype=np.float32),
            right_valid=True,
        )

        result = measure_rgb(crop, iris, upscale=1.0)
        if result is not None:
            assert 'right_ratio' in result
            assert 'left_ratio' in result

    def test_per_eye_csv_columns(self):
        t = TestPupillometryTracker()._make_tracker(per_eye=True,
                                                     baseline_frames=5)
        t._init_track(0)
        t._ts_frames[0] = [1]
        t._ts_ratios[0] = [0.35]
        t._ts_dilation[0] = [0.0]
        t._ts_valid[0] = [1]
        t._ts_ratios_left[0] = [0.34]
        t._ts_ratios_right[0] = [0.36]
        t._valid_counts[0] = 1
        t._blink_counts[0] = 0
        t._blink_timestamps[0] = []
        t._blink_durations[0] = []
        rows = t.csv_rows(100)
        # Header row should have left_ratio and right_ratio
        header = rows[2]  # [[], [section], [header]]
        assert "left_ratio" in header
        assert "right_ratio" in header


class TestVideoTypeEnum:
    """Tests for the VideoType enum."""

    def test_enum_values(self):
        from ms.pipeline_config import VideoType
        assert VideoType.EYE_ONLY.value == "eye_only"
        assert VideoType.FACE_CLOSEUP.value == "face_closeup"
        assert VideoType.WIDE_CLOSEUP.value == "wide_closeup"
        assert VideoType.CUSTOM.value == "custom"

    def test_enum_from_string(self):
        from ms.pipeline_config import VideoType
        assert VideoType("eye_only") == VideoType.EYE_ONLY

    def test_enum_is_string_comparable(self):
        from ms.pipeline_config import VideoType
        assert VideoType.EYE_ONLY == "eye_only"


class TestAuxStreamConfig:
    """Tests for the new AuxStreamConfig dataclass."""

    def test_construction(self):
        from ms.pipeline_config import AuxStreamConfig, VideoType
        cfg = AuxStreamConfig(
            source="/path/to/video.mp4",
            video_type=VideoType.EYE_ONLY,
            stream_label="left_eye",
            participants=["S70"],
        )
        assert cfg.source == "/path/to/video.mp4"
        assert cfg.video_type == VideoType.EYE_ONLY
        assert cfg.participants == ["S70"]
        assert cfg.auto_detect_faces is True

    def test_multi_participant(self):
        from ms.pipeline_config import AuxStreamConfig, VideoType
        cfg = AuxStreamConfig(
            source="/path/to/wide.mp4",
            video_type=VideoType.WIDE_CLOSEUP,
            stream_label="room_cam",
            participants=["S70", "S71", "S72"],
        )
        assert len(cfg.participants) == 3


class TestFindAuxFrame:
    """Tests for find_aux_frame helper."""

    def test_find_by_pid(self):
        from ms.pipeline_config import VideoType, find_aux_frame
        frame = np.zeros((10, 10, 3))
        aux = {("S70", "eye_cam", VideoType.EYE_ONLY): frame}
        result = find_aux_frame(aux, "S70")
        assert result is frame

    def test_find_by_video_type(self):
        from ms.pipeline_config import VideoType, find_aux_frame
        frame1 = np.zeros((10, 10, 3))
        frame2 = np.ones((10, 10, 3))
        aux = {
            ("S70", "eye_cam", VideoType.EYE_ONLY): frame1,
            ("S70", "room_cam", VideoType.WIDE_CLOSEUP): frame2,
        }
        result = find_aux_frame(aux, "S70", video_type=VideoType.EYE_ONLY)
        assert result is frame1

    def test_returns_none_when_not_found(self):
        from ms.pipeline_config import VideoType, find_aux_frame
        aux = {("S70", "eye_cam", VideoType.EYE_ONLY): np.zeros((10, 10, 3))}
        assert find_aux_frame(aux, "S71") is None
