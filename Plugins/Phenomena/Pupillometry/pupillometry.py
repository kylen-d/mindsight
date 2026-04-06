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

from ms.pipeline_config import resolve_display_pid
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
    """

    name = "pupillometry"
    dashboard_panel = "right"

    _COL = (200, 120, 255)  # light purple

    def __init__(
        self,
        mode: str = "rgb",
        baseline_frames: int = 90,
        ema_alpha: float = 0.3,
        upscale: float = 2.0,
        ir_threshold: int = 40,
    ) -> None:
        self._mode = mode
        self._baseline_frames = baseline_frames
        self._ema_alpha = ema_alpha
        self._upscale = upscale
        self._ir_threshold = ir_threshold

        # Per-track state
        self._raw_ratios: dict[int, list[float]] = {}
        self._baselines: dict[int, float | None] = {}
        self._ema: dict[int, float | None] = {}
        self._valid_counts: dict[int, int] = {}
        self._median_buf: dict[int, collections.deque] = {}  # sliding window for median pre-filter

        # Time-series storage
        self._ts_frames: dict[int, list[int]] = {}
        self._ts_ratios: dict[int, list[float]] = {}
        self._ts_dilation: dict[int, list[float]] = {}

        # Current-frame cache for dashboard/draw
        self._current_dilation: dict[int, float] = {}
        self._current_ratio: dict[int, float] = {}
        self._current_iris_offset: dict[int, tuple] = {}
        self._last_face_bboxes: list = []
        self._last_face_track_ids: list = []

        # Cross-plugin references (discovered at runtime)
        self._eye_movement_tracker = None
        self._eye_widget = None

    def _init_track(self, tid: int) -> None:
        if tid not in self._raw_ratios:
            self._raw_ratios[tid] = []
            self._baselines[tid] = None
            self._ema[tid] = None
            self._valid_counts[tid] = 0
            self._median_buf[tid] = collections.deque(maxlen=5)
            self._ts_frames[tid] = []
            self._ts_ratios[tid] = []
            self._ts_dilation[tid] = []

    def _get_face_crop(self, frame, bbox, face_idx, aux_frames, pid_map):
        """Get the best available face/eye crop for measurement."""
        # Check for aux eye camera first
        if aux_frames and pid_map:
            # Reverse pid_map: track_id -> label
            tid = self._last_face_track_ids[face_idx] if face_idx < len(self._last_face_track_ids) else face_idx
            label = resolve_display_pid(tid, pid_map)
            aux_key = (label, 'eye_camera')
            if aux_key in aux_frames and aux_frames[aux_key] is not None:
                return aux_frames[aux_key]

        # Fall back to cropping face from main frame.
        # MediaPipe needs head/neck context beyond the tight face bbox,
        # so pad the crop by 50% on all sides.
        if frame is None or bbox is None:
            return None

        bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 10 or bh < 10:
            return None
        pad_x, pad_y = bw // 2, bh // 2
        h, w = frame.shape[:2]
        x1 = max(0, bx1 - pad_x)
        y1 = max(0, by1 - pad_y)
        x2 = min(w, bx2 + pad_x)
        y2 = min(h, by2 + pad_y)
        return frame[y1:y2, x1:x2]

    def _measure(self, face_crop):
        """Dispatch to the appropriate measurement mode."""
        if self._mode == "ir":
            from Plugins.Phenomena.Pupillometry.iris_extraction import measure_ir
            return measure_ir(face_crop, threshold=self._ir_threshold)
        else:
            import cv2
            from ms.utils.mediapipe_face import extract_iris_data
            from Plugins.Phenomena.Pupillometry.iris_extraction import measure_rgb
            # Upscale small face crops so mediapipe can detect landmarks
            h, w = face_crop.shape[:2]
            if max(h, w) < 200:
                s = 200 / max(h, w)
                face_crop = cv2.resize(face_crop,
                                       (int(w * s), int(h * s)),
                                       interpolation=cv2.INTER_CUBIC)
            iris_data = extract_iris_data(face_crop)
            return measure_rgb(face_crop, iris_data, upscale=self._upscale)

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

        self._last_face_bboxes = face_bboxes
        self._last_face_track_ids = tids
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

            result = self._measure(crop)
            if result is None:
                continue

            raw_ratio = result['ratio']

            # Outlier rejection: skip if too far from running median.
            # Only activate after the buffer is full (5 samples) and always
            # add to the buffer so the filter can recover from transients.
            buf = self._median_buf[tid]
            if len(buf) >= buf.maxlen:
                running_med = float(np.median(list(buf)))
                running_std = float(np.std(list(buf)))
                threshold = max(0.08, running_std * 4)
                if abs(raw_ratio - running_med) > threshold:
                    buf.append(raw_ratio)  # still update buffer so it adapts
                    continue  # but skip this frame for display/dilation

            buf.append(raw_ratio)
            self._raw_ratios[tid].append(raw_ratio)
            self._valid_counts[tid] += 1

            # Median pre-filter: use window median instead of raw value
            ratio = float(np.median(list(buf)))

            # Compute or update baseline (robust: trimmed mean of central 60%)
            if self._baselines[tid] is None:
                if len(self._raw_ratios[tid]) >= self._baseline_frames:
                    bl_data = sorted(self._raw_ratios[tid][:self._baseline_frames])
                    trim = max(1, len(bl_data) // 5)  # drop 20% from each tail
                    self._baselines[tid] = float(np.mean(bl_data[trim:-trim]))

            # EMA smoothing on the median-filtered ratio
            if self._ema[tid] is None:
                self._ema[tid] = ratio
            else:
                self._ema[tid] = (self._ema_alpha * ratio +
                                  (1 - self._ema_alpha) * self._ema[tid])

            # Use the EMA-smoothed value as the current ratio
            smoothed = self._ema[tid]
            self._current_ratio[tid] = smoothed

            # Store iris offset for dashboard eye widget
            iris_off = result.get('iris_offset')
            if iris_off is not None:
                self._current_iris_offset[tid] = (float(iris_off[0]),
                                                   float(iris_off[1]))

            # Dilation percentage (from smoothed ratio)
            baseline = self._baselines[tid]
            if baseline is not None and baseline > 0:
                dilation_pct = (smoothed - baseline) / baseline * 100
                self._current_dilation[tid] = dilation_pct
                self._ts_dilation[tid].append(dilation_pct)
            else:
                self._ts_dilation[tid].append(0.0)

            self._ts_frames[tid].append(frame_no)
            self._ts_ratios[tid].append(ratio)

        return {}

    # ── Frame overlay ─────────────────────────────────────────────────────────

    def draw_frame(self, frame) -> None:
        if not self._current_dilation:
            return

        for fi, bbox in enumerate(self._last_face_bboxes):
            tid = (self._last_face_track_ids[fi]
                   if fi < len(self._last_face_track_ids) else fi)
            if tid not in self._current_dilation:
                continue

            dilation = self._current_dilation[tid]
            x1, y1, x2, y2 = bbox

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
                col = self._COL if abs(dil) < 15 else (0, 0, 255)
                rows.append((f"{plbl}: {dil:+.1f}%  r={ratio:.3f}{bstr}", col))
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
                rows.append({
                    'label': plbl,
                    'value': f"{dil:+.1f}%  r={ratio:.3f}",
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
        return {
            f"pupil_{tid}": {
                'value': dil,
                'label': f"P{tid} dilation",
                'y_label': '% change',
            }
            for tid, dil in self._current_dilation.items()
        }

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

        pid_map = None  # not available in snapshot; labels set from track IDs

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
            series[f"pupil_dilation_{tid}"] = {
                'x': self._ts_frames[tid],
                'y': self._ts_dilation[tid],
                'label': f"P{tid} pupil dilation %",
                'chart_type': 'line',
                'color': self._COL,
            }
        return series

    # ── CSV output ────────────────────────────────────────────────────────────

    def csv_rows(self, total_frames: int, *, pid_map=None) -> list:
        if not any(self._ts_frames.values()):
            return []

        rows: list[list] = [
            [],
            ["pupillometry_timeseries"],
            ["frame_no", "participant", "pupil_iris_ratio", "dilation_pct", "valid"],
        ]
        for tid in sorted(self._ts_frames):
            plbl = resolve_display_pid(tid, pid_map)
            for i, fno in enumerate(self._ts_frames[tid]):
                rows.append([
                    fno, plbl,
                    f"{self._ts_ratios[tid][i]:.4f}",
                    f"{self._ts_dilation[tid][i]:.2f}",
                    1,
                ])

        # Summary
        rows.append([])
        rows.append(["pupillometry_summary"])
        rows.append(["participant", "baseline_ratio", "mean_dilation_pct",
                      "std_dilation_pct", "max_dilation_pct", "n_valid_frames"])
        for tid in sorted(self._ts_frames):
            plbl = resolve_display_pid(tid, pid_map)
            baseline = self._baselines.get(tid)
            dils = self._ts_dilation.get(tid, [])
            if dils:
                rows.append([
                    plbl,
                    f"{baseline:.4f}" if baseline else "",
                    f"{np.mean(dils):.2f}",
                    f"{np.std(dils):.2f}",
                    f"{np.max(np.abs(dils)):.2f}",
                    self._valid_counts.get(tid, 0),
                ])

        return rows

    def console_summary(self, total_frames: int, *, pid_map=None) -> str | None:
        if not any(self._ts_frames.values()):
            return None

        lines = ["\n--- Pupillometry Summary ---"]
        for tid in sorted(self._ts_frames):
            plbl = resolve_display_pid(tid, pid_map)
            baseline = self._baselines.get(tid)
            dils = self._ts_dilation.get(tid, [])
            n = self._valid_counts.get(tid, 0)
            if dils:
                lines.append(
                    f"  {plbl}: baseline={baseline:.4f if baseline else 'N/A'}"
                    f"  mean_dil={np.mean(dils):.1f}%"
                    f"  max_dil={np.max(np.abs(dils)):.1f}%"
                    f"  valid={n}/{total_frames}"
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
        g.add_argument("--pupil-ema", type=float, default=0.3, metavar="F",
                        help="EMA smoothing alpha (default: 0.3).")
        g.add_argument("--pupil-upscale", type=float, default=2.0, metavar="F",
                        help="Upscale factor for RGB mode (default: 2.0).")
        g.add_argument("--pupil-ir-thresh", type=int, default=40, metavar="N",
                        help="IR threshold for dark-pupil segmentation (default: 40).")

    @classmethod
    def from_args(cls, args):
        if not getattr(args, "pupillometry", False):
            return None
        inst = cls(
            mode=getattr(args, "pupil_mode", "rgb"),
            baseline_frames=getattr(args, "pupil_baseline", 90),
            ema_alpha=getattr(args, "pupil_ema", 0.3),
            upscale=getattr(args, "pupil_upscale", 2.0),
            ir_threshold=getattr(args, "pupil_ir_thresh", 40),
        )
        print(
            f"Pupillometry: mode={inst._mode}  baseline={inst._baseline_frames}f"
            f"  ema={inst._ema_alpha}  upscale={inst._upscale}x"
        )
        return inst


PLUGIN_CLASS = PupillometryTracker
