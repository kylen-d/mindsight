"""
Plugins/Phenomena/Pupillometry/pupillometry.py -- Pupillometry PhenomenaPlugin.

Measures pupil dilation via pupil/iris diameter ratio, supporting RGB
(MediaPipe iris landmarks) and IR (dark-pupil threshold) modes.  Outputs
per-participant dilation percentage relative to a per-session baseline.

This plugin is a pure observer -- it never modifies gaze pipeline output.

Activation: ``--pupillometry``
"""

from __future__ import annotations

import collections

import cv2
import numpy as np

from ms.pipeline_config import VideoType, find_aux_frame, resolve_display_pid
from Plugins import PhenomenaPlugin


# Dashboard drawing helpers (lazy import)
def _dash():
    from ms.DataCollection.dashboard_output import (
        _DASH_DIM,
        _dash_line_h,
        _draw_panel_section,
    )
    return _draw_panel_section, _dash_line_h, _DASH_DIM


class PupillometryTracker(PhenomenaPlugin):
    """
    Tracks pupil dilation via pupil/iris ratio measurement.

    Supports RGB mode (MediaPipe + thresholding on face crops) and IR mode
    (dark-pupil technique on auxiliary IR eye camera streams).

    Features:
    - Kalman or EMA filtering for ratio smoothing
    - Hampel outlier rejection (sliding-window MAD)
    - EAR-based blink detection with full blink data collection
    - Optional per-eye measurements
    """

    name = "pupillometry"
    dashboard_panel = "right"

    _COL = (200, 120, 255)  # light purple

    preferred_video_types = [VideoType.EYE_ONLY, VideoType.FACE_CLOSEUP]

    def __init__(
        self,
        mode: str = "rgb",
        baseline_frames: int = 90,
        ema_alpha: float = 0.3,
        upscale: float = 2.0,
        ir_threshold: int = 40,
        filter_mode: str = "kalman",
        kalman_process_noise: float = 1e-4,
        kalman_measurement_noise: float = 1e-2,
        ear_threshold: float = 0.21,
        blink_consec: int = 2,
        outlier_window: int = 15,
        per_eye: bool = False,
    ) -> None:
        self._mode = mode
        self._baseline_frames = baseline_frames
        self._ema_alpha = ema_alpha
        self._upscale = upscale
        self._ir_threshold = ir_threshold
        self._filter_mode = filter_mode
        self._kalman_process_noise = kalman_process_noise
        self._kalman_measurement_noise = kalman_measurement_noise
        self._ear_threshold = ear_threshold
        self._blink_consec = blink_consec
        self._outlier_window = outlier_window
        self._per_eye = per_eye

        # Per-track state
        self._raw_ratios: dict[int, list[float]] = {}
        self._baselines: dict[int, float | None] = {}
        self._ema: dict[int, float | None] = {}
        self._kalman: dict[int, object] = {}
        self._valid_counts: dict[int, int] = {}
        self._outlier_deque: dict[int, collections.deque] = {}

        # Per-eye state (when enabled)
        self._ts_ratios_left: dict[int, list[float]] = {}
        self._ts_ratios_right: dict[int, list[float]] = {}
        self._kalman_left: dict[int, object] = {}
        self._kalman_right: dict[int, object] = {}
        self._ema_left: dict[int, float | None] = {}
        self._ema_right: dict[int, float | None] = {}

        # Blink detection state
        self._ear_counters: dict[int, int] = {}
        self._blink_flags: dict[int, bool] = {}
        self._blink_in_progress: dict[int, bool] = {}
        self._blink_start_frame: dict[int, int] = {}

        # Blink data collection
        self._blink_counts: dict[int, int] = {}
        self._blink_timestamps: dict[int, list[int]] = {}
        self._blink_durations: dict[int, list[int]] = {}
        self._blink_frame_counter: dict[int, int] = {}

        # Time-series storage
        self._ts_frames: dict[int, list[int]] = {}
        self._ts_ratios: dict[int, list[float]] = {}
        self._ts_dilation: dict[int, list[float]] = {}
        self._ts_valid: dict[int, list[int]] = {}
        self._ts_blink_rate: dict[int, list[float]] = {}

        # Current-frame cache for dashboard/draw
        self._current_dilation: dict[int, float] = {}
        self._current_ratio: dict[int, float] = {}
        self._current_iris_offset: dict[int, tuple] = {}
        self._last_face_bboxes: list = []
        self._last_face_track_ids: list = []
        self._frame_no: int = 0
        self._fps: float = 30.0

        # Participant ID map (updated each frame from kwargs)
        self._pid_map: dict | None = None

        # Cross-plugin references (discovered at runtime)
        self._eye_movement_tracker = None
        self._eye_widget = None

    def _init_track(self, tid: int) -> None:
        if tid not in self._raw_ratios:
            self._raw_ratios[tid] = []
            self._baselines[tid] = None
            self._ema[tid] = None
            self._valid_counts[tid] = 0
            self._outlier_deque[tid] = collections.deque(
                maxlen=self._outlier_window)
            self._ts_frames[tid] = []
            self._ts_ratios[tid] = []
            self._ts_dilation[tid] = []
            self._ts_valid[tid] = []
            self._ts_blink_rate[tid] = []

            # Blink state
            self._ear_counters[tid] = 0
            self._blink_flags[tid] = False
            self._blink_in_progress[tid] = False
            self._blink_start_frame[tid] = 0
            self._blink_counts[tid] = 0
            self._blink_timestamps[tid] = []
            self._blink_durations[tid] = []
            self._blink_frame_counter[tid] = 0

            # Per-eye
            if self._per_eye:
                self._ts_ratios_left[tid] = []
                self._ts_ratios_right[tid] = []
                self._ema_left[tid] = None
                self._ema_right[tid] = None

            # Kalman filters
            if self._filter_mode == "kalman":
                from Plugins.Phenomena.Pupillometry.kalman import PupilKalman
                kw = dict(process_noise=self._kalman_process_noise,
                          measurement_noise=self._kalman_measurement_noise)
                self._kalman[tid] = PupilKalman(**kw)
                if self._per_eye:
                    self._kalman_left[tid] = PupilKalman(**kw)
                    self._kalman_right[tid] = PupilKalman(**kw)

    def _get_face_crop(self, frame, bbox, face_idx, aux_frames, pid_map):
        """Get the best available face/eye crop for measurement."""
        # Check for auxiliary stream (EYE_ONLY preferred, FACE_CLOSEUP fallback)
        if aux_frames and pid_map:
            tid = self._last_face_track_ids[face_idx] if face_idx < len(self._last_face_track_ids) else face_idx
            label = resolve_display_pid(tid, pid_map)
            aux_frame = self.get_aux_frame(aux_frames, label)
            if aux_frame is not None:
                return aux_frame

        # Fall back to cropping face from main frame
        if frame is None or bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return frame[y1:y2, x1:x2]

    def _measure(self, face_crop):
        """Dispatch to the appropriate measurement mode."""
        if self._mode == "ir":
            from Plugins.Phenomena.Pupillometry.iris_extraction import measure_ir
            return measure_ir(face_crop, threshold=self._ir_threshold)
        else:
            from ms.utils.mediapipe_face import extract_iris_data
            from Plugins.Phenomena.Pupillometry.iris_extraction import measure_rgb
            iris_data = extract_iris_data(face_crop)
            return measure_rgb(face_crop, iris_data, upscale=self._upscale)

    def _check_blink(self, face_crop, tid, frame_no):
        """Check for blink via EAR. Returns True if blinking."""
        from Plugins.Phenomena.Pupillometry.iris_extraction import compute_ear
        from ms.utils.mediapipe_face import extract_iris_data

        iris_data = extract_iris_data(face_crop)
        ear = compute_ear(iris_data)

        if ear is None:
            # Can't compute EAR -- treat as potential blink
            self._ear_counters[tid] += 1
        elif ear < self._ear_threshold:
            self._ear_counters[tid] += 1
        else:
            # Eye open -- finalize any in-progress blink
            if self._blink_in_progress[tid]:
                duration = frame_no - self._blink_start_frame[tid]
                self._blink_durations[tid].append(duration)
                self._blink_in_progress[tid] = False
            self._ear_counters[tid] = 0

        is_blink = self._ear_counters[tid] >= self._blink_consec
        self._blink_flags[tid] = is_blink

        # Track blink start
        if is_blink and not self._blink_in_progress[tid]:
            self._blink_in_progress[tid] = True
            self._blink_start_frame[tid] = frame_no
            self._blink_counts[tid] += 1
            self._blink_timestamps[tid].append(frame_no)

        return is_blink

    def _is_outlier(self, tid, ratio):
        """Hampel filter: reject if ratio > 3*MAD from window median."""
        window = self._outlier_deque[tid]
        window.append(ratio)

        # Need at least 3 samples for meaningful MAD
        if len(window) < 3:
            return False

        arr = np.array(window)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))

        if mad < 1e-6:
            return False

        return abs(ratio - med) > 3.0 * mad

    def _filter_ratio(self, tid, ratio):
        """Apply Kalman or EMA filtering to a ratio measurement."""
        if self._filter_mode == "kalman":
            return self._kalman[tid].update(ratio)
        else:
            if self._ema[tid] is None:
                self._ema[tid] = ratio
            else:
                self._ema[tid] = (self._ema_alpha * ratio +
                                  (1 - self._ema_alpha) * self._ema[tid])
            return self._ema[tid]

    def _filter_ratio_eye(self, tid, ratio, side):
        """Apply filtering to per-eye ratio."""
        if side == 'left':
            if self._filter_mode == "kalman":
                return self._kalman_left[tid].update(ratio)
            else:
                if self._ema_left[tid] is None:
                    self._ema_left[tid] = ratio
                else:
                    self._ema_left[tid] = (self._ema_alpha * ratio +
                                           (1 - self._ema_alpha) * self._ema_left[tid])
                return self._ema_left[tid]
        else:
            if self._filter_mode == "kalman":
                return self._kalman_right[tid].update(ratio)
            else:
                if self._ema_right[tid] is None:
                    self._ema_right[tid] = ratio
                else:
                    self._ema_right[tid] = (self._ema_alpha * ratio +
                                            (1 - self._ema_alpha) * self._ema_right[tid])
                return self._ema_right[tid]

    def _blink_rate(self, tid, frame_no):
        """Compute rolling blink rate (blinks per minute) over last 30s."""
        if not self._blink_timestamps[tid]:
            return 0.0
        window_frames = int(30 * self._fps)
        cutoff = frame_no - window_frames
        recent = [t for t in self._blink_timestamps[tid] if t >= cutoff]
        window_sec = min(frame_no / self._fps, 30.0)
        if window_sec < 1:
            return 0.0
        return len(recent) * 60.0 / window_sec

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, **kwargs) -> dict:
        frame_no = kwargs['frame_no']
        persons_gaze = kwargs.get('persons_gaze', [])
        face_bboxes = kwargs.get('face_bboxes', [])
        face_track_ids = kwargs.get('face_track_ids')
        frame = kwargs.get('frame')
        aux_frames = kwargs.get('aux_frames', {})
        pid_map = kwargs.get('pid_map')

        tids = face_track_ids if face_track_ids is not None \
               else list(range(len(persons_gaze)))

        self._pid_map = pid_map
        self._last_face_bboxes = face_bboxes
        self._last_face_track_ids = tids
        self._frame_no = frame_no
        self._current_dilation.clear()
        self._current_ratio.clear()
        self._current_iris_offset.clear()

        # Discover EyeMovement tracker on first frame (for dashboard widget)
        if self._eye_movement_tracker is None:
            for t in kwargs.get('_all_trackers', []):
                if getattr(t, 'name', '') == 'eye_movement':
                    self._eye_movement_tracker = t
                    break

        for fi in range(len(persons_gaze)):
            tid = tids[fi] if fi < len(tids) else fi
            self._init_track(tid)

            bbox = face_bboxes[fi] if fi < len(face_bboxes) else None
            crop = self._get_face_crop(frame, bbox, fi, aux_frames, pid_map)
            if crop is None:
                continue

            # Blink detection (before measurement)
            is_blink = self._check_blink(crop, tid, frame_no)
            if is_blink:
                # Record blink frame in time-series but skip measurement
                self._ts_frames[tid].append(frame_no)
                self._ts_ratios[tid].append(float('nan'))
                self._ts_dilation[tid].append(float('nan'))
                self._ts_valid[tid].append(0)
                self._ts_blink_rate[tid].append(
                    self._blink_rate(tid, frame_no))
                if self._per_eye:
                    self._ts_ratios_left[tid].append(float('nan'))
                    self._ts_ratios_right[tid].append(float('nan'))
                continue

            result = self._measure(crop)
            if result is None:
                continue

            ratio = result['ratio']

            # Outlier rejection (Hampel filter)
            if self._is_outlier(tid, ratio):
                # Clamp to window median instead of discarding
                window = self._outlier_deque[tid]
                ratio = float(np.median(list(window)))

            # Sanity bounds
            if not (0.1 <= ratio <= 0.8):
                continue

            self._raw_ratios[tid].append(ratio)
            self._valid_counts[tid] += 1

            # Compute or update baseline
            if self._baselines[tid] is None:
                if len(self._raw_ratios[tid]) >= self._baseline_frames:
                    self._baselines[tid] = float(
                        np.median(self._raw_ratios[tid][:self._baseline_frames])
                    )

            # Filter ratio (Kalman or EMA)
            filtered_ratio = self._filter_ratio(tid, ratio)
            self._current_ratio[tid] = filtered_ratio

            # Per-eye measurements
            if self._per_eye:
                left_r = result.get('left_ratio')
                right_r = result.get('right_ratio')
                if left_r is not None:
                    left_r = self._filter_ratio_eye(tid, left_r, 'left')
                    self._ts_ratios_left[tid].append(left_r)
                else:
                    self._ts_ratios_left[tid].append(float('nan'))
                if right_r is not None:
                    right_r = self._filter_ratio_eye(tid, right_r, 'right')
                    self._ts_ratios_right[tid].append(right_r)
                else:
                    self._ts_ratios_right[tid].append(float('nan'))

            # Store iris offset for dashboard eye widget
            iris_off = result.get('iris_offset')
            if iris_off is not None:
                self._current_iris_offset[tid] = (float(iris_off[0]),
                                                   float(iris_off[1]))

            # Dilation percentage
            baseline = self._baselines[tid]
            if baseline is not None and baseline > 0:
                dilation_pct = (filtered_ratio - baseline) / baseline * 100
                self._current_dilation[tid] = dilation_pct
                self._ts_dilation[tid].append(dilation_pct)
            else:
                self._ts_dilation[tid].append(0.0)

            self._ts_frames[tid].append(frame_no)
            self._ts_ratios[tid].append(filtered_ratio)
            self._ts_valid[tid].append(1)
            self._ts_blink_rate[tid].append(
                self._blink_rate(tid, frame_no))

        return {}

    # ── Frame overlay ─────────────────────────────────────────────────────────

    def draw_frame(self, frame) -> None:
        for fi, bbox in enumerate(self._last_face_bboxes):
            tid = (self._last_face_track_ids[fi]
                   if fi < len(self._last_face_track_ids) else fi)

            x1, y1, x2, y2 = bbox

            # Draw blink indicator
            if self._blink_flags.get(tid, False):
                blink_col = (0, 165, 255)  # orange
                lx = int(x1)
                ly = max(int(y1) - 24, 12)
                cv2.putText(frame, "BLINK", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, blink_col, 1,
                            cv2.LINE_AA)

            if tid not in self._current_dilation:
                continue

            dilation = self._current_dilation[tid]

            # Color by magnitude: green=contracted, yellow=normal, red=dilated
            if dilation > 15:
                col = (0, 0, 255)
            elif dilation > 5:
                col = (0, 180, 255)
            elif dilation < -10:
                col = (255, 180, 0)
            else:
                col = (0, 255, 0)

            label = f"Pupil: {dilation:+.1f}%"
            lx = int(x1)
            ly = max(int(y1) - 8, 12)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def dashboard_section(self, panel, y: int, line_h: int, *,
                          pid_map=None) -> int:
        draw_section, _, DASH_DIM = _dash()
        rows = []

        if self._current_dilation:
            for tid in sorted(self._current_dilation):
                dil = self._current_dilation[tid]
                ratio = self._current_ratio.get(tid, 0)
                plbl = resolve_display_pid(tid, pid_map)
                baseline = self._baselines.get(tid)
                bstr = f"  (bl:{baseline:.3f})" if baseline else "  (calibrating...)"
                blinks = self._blink_counts.get(tid, 0)
                col = self._COL if abs(dil) < 15 else (0, 0, 255)
                rows.append((f"{plbl}: {dil:+.1f}%  r={ratio:.3f}{bstr}  blinks={blinks}", col))
        else:
            rows = [("--", DASH_DIM)]

        return draw_section(panel, y, "PUPILLOMETRY", self._COL, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        if self._current_dilation:
            for tid in sorted(self._current_dilation):
                dil = self._current_dilation[tid]
                ratio = self._current_ratio.get(tid, 0)
                plbl = resolve_display_pid(tid, pid_map)
                blinks = self._blink_counts.get(tid, 0)
                brate = self._blink_rate(tid, self._frame_no)
                rows.append({
                    'label': plbl,
                    'value': f"{dil:+.1f}%  r={ratio:.3f}  blinks={blinks} ({brate:.0f}/min)",
                    'pct': min(1.0, max(0.0, (dil + 30) / 60)),
                })
        return {
            'title': 'PUPILLOMETRY',
            'colour': self._COL,
            'rows': rows,
            'empty_text': '--',
        }

    def latest_metrics(self) -> dict | None:
        if not self._current_dilation:
            return None
        metrics = {}
        for tid, dil in self._current_dilation.items():
            plbl = resolve_display_pid(tid, self._pid_map)
            metrics[f"pupil_{tid}"] = {
                'value': dil,
                'label': f"{plbl} dilation",
                'y_label': '% change',
            }
            metrics[f"blink_count_{tid}"] = {
                'value': self._blink_counts.get(tid, 0),
                'label': f"{plbl} blinks",
                'y_label': 'count',
            }
            metrics[f"blink_rate_{tid}"] = {
                'value': self._blink_rate(tid, self._frame_no),
                'label': f"{plbl} blink rate",
                'y_label': 'blinks/min',
            }
        return metrics

    # ── Custom Qt dashboard widget ──────────────────────────────────────────

    def dashboard_widget(self):
        try:
            from ms.GUI.eye_tracking_widget import EyeTrackingWidget
            self._eye_widget = EyeTrackingWidget()
            return self._eye_widget
        except ImportError:
            return None

    def dashboard_widget_update(self, data: dict) -> None:
        if self._eye_widget is None:
            return

        pid_map = self._pid_map

        # Get eye states from EyeMovement tracker if available
        eye_states = {}
        if self._eye_movement_tracker is not None:
            em_states = getattr(self._eye_movement_tracker, '_current_states', {})
            for tid, state in em_states.items():
                eye_states[tid] = state.value  # EyeState enum -> str

        # Push per-participant data to the widget
        for tid in sorted(set(list(self._current_ratio.keys()) +
                               list(self._current_dilation.keys()))):
            self._eye_widget.update_participant(
                tid,
                dilation_pct=self._current_dilation.get(tid),
                ratio=self._current_ratio.get(tid),
                baseline=self._baselines.get(tid),
                iris_offset=self._current_iris_offset.get(tid),
                eye_state=eye_states.get(tid),
            )

        self._eye_widget.refresh()

    # ── Time-series ───────────────────────────────────────────────────────────

    def time_series_data(self) -> dict:
        series = {}
        for tid in sorted(self._ts_frames):
            if not self._ts_dilation[tid]:
                continue
            plbl = resolve_display_pid(tid, self._pid_map)

            # Pupil dilation chart
            series[f"pupil_dilation_{plbl}"] = {
                'x': self._ts_frames[tid],
                'y': self._ts_dilation[tid],
                'label': f"{plbl} pupil dilation %",
                'chart_type': 'line',
                'color': self._COL,
            }

            # Blink rate chart
            if self._ts_blink_rate.get(tid):
                series[f"blink_rate_{plbl}"] = {
                    'x': self._ts_frames[tid],
                    'y': self._ts_blink_rate[tid],
                    'label': f"{plbl} blink rate (blinks/min)",
                    'chart_type': 'line',
                    'color': (0, 165, 255),
                }

            # Per-eye charts
            if self._per_eye and self._ts_ratios_left.get(tid):
                series[f"pupil_left_{plbl}"] = {
                    'x': self._ts_frames[tid],
                    'y': self._ts_ratios_left[tid],
                    'label': f"{plbl} left eye ratio",
                    'chart_type': 'line',
                    'color': (255, 100, 100),
                }
                series[f"pupil_right_{plbl}"] = {
                    'x': self._ts_frames[tid],
                    'y': self._ts_ratios_right[tid],
                    'label': f"{plbl} right eye ratio",
                    'chart_type': 'line',
                    'color': (100, 100, 255),
                }

        return series

    # ── CSV output ────────────────────────────────────────────────────────────

    def csv_rows(self, total_frames: int, *, pid_map=None) -> list:
        if not any(self._ts_frames.values()):
            return []

        # Time-series header
        header = ["frame_no", "participant", "pupil_iris_ratio",
                  "dilation_pct", "valid"]
        if self._per_eye:
            header.extend(["left_ratio", "right_ratio"])

        rows: list[list] = [
            [],
            ["pupillometry_timeseries"],
            header,
        ]
        for tid in sorted(self._ts_frames):
            plbl = resolve_display_pid(tid, pid_map)
            for i, fno in enumerate(self._ts_frames[tid]):
                row = [
                    fno, plbl,
                    f"{self._ts_ratios[tid][i]:.4f}",
                    f"{self._ts_dilation[tid][i]:.2f}",
                    self._ts_valid[tid][i] if i < len(self._ts_valid[tid]) else 1,
                ]
                if self._per_eye:
                    lr = self._ts_ratios_left.get(tid, [])
                    rr = self._ts_ratios_right.get(tid, [])
                    row.append(f"{lr[i]:.4f}" if i < len(lr) and not np.isnan(lr[i]) else "")
                    row.append(f"{rr[i]:.4f}" if i < len(rr) and not np.isnan(rr[i]) else "")
                rows.append(row)

        # Summary
        rows.append([])
        rows.append(["pupillometry_summary"])
        rows.append(["participant", "baseline_ratio", "mean_dilation_pct",
                      "std_dilation_pct", "max_dilation_pct", "n_valid_frames",
                      "total_blinks", "mean_blink_duration_frames",
                      "blinks_per_minute"])
        for tid in sorted(self._ts_frames):
            plbl = resolve_display_pid(tid, pid_map)
            baseline = self._baselines.get(tid)
            dils = [d for d in self._ts_dilation.get(tid, [])
                    if not np.isnan(d)]
            blinks = self._blink_counts.get(tid, 0)
            bdurs = self._blink_durations.get(tid, [])
            mean_bdur = np.mean(bdurs) if bdurs else 0
            total_sec = total_frames / self._fps if self._fps > 0 else 1
            bpm = blinks * 60.0 / total_sec if total_sec > 0 else 0

            if dils:
                rows.append([
                    plbl,
                    f"{baseline:.4f}" if baseline else "",
                    f"{np.mean(dils):.2f}",
                    f"{np.std(dils):.2f}",
                    f"{np.max(np.abs(dils)):.2f}",
                    self._valid_counts.get(tid, 0),
                    blinks,
                    f"{mean_bdur:.1f}",
                    f"{bpm:.1f}",
                ])

        # Blink events section
        rows.append([])
        rows.append(["blink_events"])
        rows.append(["participant", "blink_start_frame", "blink_duration_frames"])
        for tid in sorted(self._blink_timestamps):
            plbl = resolve_display_pid(tid, pid_map)
            timestamps = self._blink_timestamps[tid]
            durations = self._blink_durations[tid]
            for i, start in enumerate(timestamps):
                dur = durations[i] if i < len(durations) else ""
                rows.append([plbl, start, dur])

        return rows

    def console_summary(self, total_frames: int, *, pid_map=None) -> str | None:
        if not any(self._ts_frames.values()):
            return None

        lines = ["\n--- Pupillometry Summary ---"]
        for tid in sorted(self._ts_frames):
            plbl = resolve_display_pid(tid, pid_map)
            baseline = self._baselines.get(tid)
            dils = [d for d in self._ts_dilation.get(tid, [])
                    if not np.isnan(d)]
            n = self._valid_counts.get(tid, 0)
            blinks = self._blink_counts.get(tid, 0)
            total_sec = total_frames / self._fps if self._fps > 0 else 1
            bpm = blinks * 60.0 / total_sec if total_sec > 0 else 0
            if dils:
                lines.append(
                    f"  {plbl}: baseline={baseline:.4f if baseline else 'N/A'}"
                    f"  mean_dil={np.mean(dils):.1f}%"
                    f"  max_dil={np.max(np.abs(dils)):.1f}%"
                    f"  valid={n}/{total_frames}"
                    f"  blinks={blinks} ({bpm:.0f}/min)"
                )
        return "\n".join(lines)

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        g = parser.add_argument_group("Pupillometry plugin")
        g.add_argument("--pupillometry", action="store_true",
                        help="Enable pupillometry tracking.")
        g.add_argument("--pupil-mode", choices=["rgb", "ir"], default="rgb",
                        help="Measurement mode (default: rgb).")
        g.add_argument("--pupil-baseline", type=int, default=90, metavar="N",
                        help="Baseline frames for calibration (default: 90).")
        g.add_argument("--pupil-upscale", type=float, default=2.0, metavar="F",
                        help="Upscale factor for RGB mode (default: 2.0).")
        g.add_argument("--pupil-ir-thresh", type=int, default=40, metavar="N",
                        help="IR threshold for dark-pupil segmentation (default: 40).")
        g.add_argument("--pupil-filter", choices=["kalman", "ema"],
                        default="kalman",
                        help="Ratio smoothing filter (default: kalman).")
        g.add_argument("--pupil-ema-alpha", type=float, default=0.3,
                        metavar="F",
                        help="EMA smoothing alpha -- only used when "
                             "--pupil-filter is ema (default: 0.3).")
        g.add_argument("--pupil-kalman-process-noise", type=float,
                        default=0.0001, metavar="F",
                        help="Kalman process noise -- controls how fast the "
                             "filter adapts to ratio changes. Only used when "
                             "--pupil-filter is kalman (default: 0.0001).")
        g.add_argument("--pupil-kalman-meas-noise", type=float,
                        default=0.01, metavar="F",
                        help="Kalman measurement noise -- higher values smooth "
                             "more aggressively. Only used when "
                             "--pupil-filter is kalman (default: 0.01).")
        g.add_argument("--pupil-ear-thresh", type=float, default=0.21,
                        metavar="F",
                        help="EAR threshold for blink detection (default: 0.21).")
        g.add_argument("--pupil-blink-frames", type=int, default=2,
                        metavar="N",
                        help="Min consecutive low-EAR frames for blink (default: 2).")
        g.add_argument("--pupil-outlier-window", type=int, default=15,
                        metavar="N",
                        help="Hampel outlier filter window size (default: 15).")
        g.add_argument("--pupil-per-eye", action="store_true", default=False,
                        help="Enable per-eye (left/right) measurements.")

    @classmethod
    def from_args(cls, args):
        if not getattr(args, "pupillometry", False):
            return None
        inst = cls(
            mode=getattr(args, "pupil_mode", "rgb"),
            baseline_frames=getattr(args, "pupil_baseline", 90),
            ema_alpha=getattr(args, "pupil_ema_alpha", 0.3),
            upscale=getattr(args, "pupil_upscale", 2.0),
            ir_threshold=getattr(args, "pupil_ir_thresh", 40),
            filter_mode=getattr(args, "pupil_filter", "kalman"),
            kalman_process_noise=getattr(args, "pupil_kalman_process_noise", 1e-4),
            kalman_measurement_noise=getattr(args, "pupil_kalman_meas_noise", 1e-2),
            ear_threshold=getattr(args, "pupil_ear_thresh", 0.21),
            blink_consec=getattr(args, "pupil_blink_frames", 2),
            outlier_window=getattr(args, "pupil_outlier_window", 15),
            per_eye=getattr(args, "pupil_per_eye", False),
        )
        filter_info = inst._filter_mode
        if inst._filter_mode == "kalman":
            filter_info += (f" (Q={inst._kalman_process_noise}, "
                            f"R={inst._kalman_measurement_noise})")
        else:
            filter_info += f" (alpha={inst._ema_alpha})"
        print(
            f"Pupillometry: mode={inst._mode}  baseline={inst._baseline_frames}f"
            f"  filter={filter_info}  upscale={inst._upscale}x"
            f"  blink_ear={inst._ear_threshold}  per_eye={inst._per_eye}"
        )
        return inst


PLUGIN_CLASS = PupillometryTracker
