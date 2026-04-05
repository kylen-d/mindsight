"""Performance, tracking, and pipeline tuning settings."""

from __future__ import annotations

from argparse import Namespace

from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PerformanceSection(QWidget):
    """Skip-frames, detection scale, ReID, fast mode, profiling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._build_ui(lay)

    def _build_ui(self, lay):
        g = QGroupBox("Performance && Tracking")
        f = QFormLayout(g)

        self._skip_frames = QSpinBox()
        self._skip_frames.setRange(1, 10)
        self._skip_frames.setValue(1)
        self._skip_frames.setToolTip(
            "Run full detection every N frames (1 = every frame)")
        f.addRow("Skip frames:", self._skip_frames)

        self._detect_scale = QDoubleSpinBox()
        self._detect_scale.setRange(0.25, 1.0)
        self._detect_scale.setSingleStep(0.05)
        self._detect_scale.setValue(1.0)
        self._detect_scale.setDecimals(2)
        self._detect_scale.setToolTip(
            "Downscale factor for detection (lower = faster, less accurate)")
        f.addRow("Detect scale:", self._detect_scale)

        self._reid_grace = QDoubleSpinBox()
        self._reid_grace.setRange(0.0, 10.0)
        self._reid_grace.setSingleStep(0.5)
        self._reid_grace.setValue(1.0)
        self._reid_grace.setDecimals(1)
        self._reid_grace.setToolTip(
            "Seconds to keep a lost track before dropping it")
        f.addRow("ReID grace (s):", self._reid_grace)

        self._obj_persistence = QSpinBox()
        self._obj_persistence.setRange(0, 60)
        self._obj_persistence.setValue(0)
        self._obj_persistence.setToolTip(
            "Frames to persist objects after disappearance")
        f.addRow("Obj persistence:", self._obj_persistence)

        self._cb_gaze_debug = QCheckBox("Show pitch/yaw debug overlay")
        f.addRow(self._cb_gaze_debug)

        self._cb_fast = QCheckBox("Fast mode (bundle perf optimizations)")
        self._cb_fast.setToolTip(
            "Skip phenomena on non-detection frames, throttle dashboard, "
            "reduce GUI poll rate")
        f.addRow(self._cb_fast)

        self._skip_phenomena = QSpinBox()
        self._skip_phenomena.setRange(0, 30)
        self._skip_phenomena.setValue(0)
        self._skip_phenomena.setToolTip(
            "Run phenomena trackers every N frames (0 = every frame)")
        f.addRow("Skip phenomena:", self._skip_phenomena)

        self._cb_lite_overlay = QCheckBox("Lite overlay (minimal drawing)")
        self._cb_lite_overlay.setToolTip(
            "Disable cone rendering, convergence markers, and debug text")
        f.addRow(self._cb_lite_overlay)

        self._cb_no_dashboard = QCheckBox("No dashboard (raw frame only)")
        self._cb_no_dashboard.setToolTip(
            "Skip dashboard composition for maximum throughput")
        f.addRow(self._cb_no_dashboard)

        self._cb_profile = QCheckBox("Profile (per-stage timing)")
        self._cb_profile.setToolTip(
            "Print per-stage timing breakdown every 100 frames")
        f.addRow(self._cb_profile)

        lay.addWidget(g)

    # -- Namespace interface --------------------------------------------------

    def namespace_values(self) -> dict:
        return dict(
            skip_frames=self._skip_frames.value(),
            detect_scale=self._detect_scale.value(),
            reid_grace_seconds=self._reid_grace.value(),
            obj_persistence=self._obj_persistence.value(),
            gaze_debug=self._cb_gaze_debug.isChecked(),
            fast=self._cb_fast.isChecked(),
            skip_phenomena=self._skip_phenomena.value(),
            lite_overlay=self._cb_lite_overlay.isChecked(),
            no_dashboard=self._cb_no_dashboard.isChecked(),
            profile=self._cb_profile.isChecked(),
        )

    def apply_namespace(self, ns: Namespace):
        self._detect_scale.setValue(getattr(ns, 'detect_scale', 1.0))
        self._skip_frames.setValue(getattr(ns, 'skip_frames', 1))
        self._obj_persistence.setValue(getattr(ns, 'obj_persistence', 0))
        self._reid_grace.setValue(getattr(ns, 'reid_grace_seconds', 1.0))
        self._cb_gaze_debug.setChecked(
            bool(getattr(ns, 'gaze_debug', False)))
        self._cb_fast.setChecked(bool(getattr(ns, 'fast', False)))
        self._skip_phenomena.setValue(
            int(getattr(ns, 'skip_phenomena', 0)))
        self._cb_lite_overlay.setChecked(
            bool(getattr(ns, 'lite_overlay', False)))
        self._cb_no_dashboard.setChecked(
            bool(getattr(ns, 'no_dashboard', False)))
        self._cb_profile.setChecked(bool(getattr(ns, 'profile', False)))

    def reset_defaults(self):
        self._skip_frames.setValue(1)
        self._detect_scale.setValue(1.0)
        self._reid_grace.setValue(1.0)
        self._obj_persistence.setValue(0)
        self._cb_gaze_debug.setChecked(False)
        self._cb_fast.setChecked(False)
        self._skip_phenomena.setValue(0)
        self._cb_lite_overlay.setChecked(False)
        self._cb_no_dashboard.setChecked(False)
        self._cb_profile.setChecked(False)
