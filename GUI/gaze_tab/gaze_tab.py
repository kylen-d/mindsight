"""
gaze_tab.py
-----------
Gaze Tracker tab coordinator.  Composes section widgets and provides the
``_build_namespace`` / ``apply_namespace`` interface consumed by the rest
of the GUI.
"""

from __future__ import annotations

import queue
from argparse import Namespace
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..widgets import _bgr_to_pixmap, _hrow

from .detection_section import DetectionSection
from .gaze_backend_section import GazeBackendSection
from .ray_section import RaySection
from .performance_section import PerformanceSection
from .output_section import OutputSection


class GazeTab(QWidget):
    """Full-featured Gaze Tracker tab with CLI parity."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=4)
        self._log_q: queue.Queue = queue.Queue()
        self._dashboard_q: queue.Queue = queue.Queue(maxsize=30)
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll)

        self._phenomena_panel = None
        self._plugin_panel = None

        self._build_ui()

    # -- UI layout ------------------------------------------------------------

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # Left: scrollable settings panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_w = QWidget()
        settings_lay = QVBoxLayout(settings_w)
        settings_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        settings_lay.setSpacing(6)
        scroll.setWidget(settings_w)

        self._build_settings(settings_lay)

        # Right: vertical splitter (Video | Dashboard | Log)
        self._preview = QLabel()
        self._preview.setStyleSheet("background:#1a1a2e;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        from ..live_dashboard import LiveDashboardPanel
        self._dashboard_panel = LiveDashboardPanel(self._dashboard_q)

        log_group = QGroupBox("Log")
        log_group.setCheckable(True)
        log_group.setChecked(False)
        log_lay = QVBoxLayout(log_group)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(60)
        self._log_box.setFont(QFont("Courier", 10))
        log_lay.addWidget(self._log_box)
        self._log_box.setVisible(False)
        log_group.toggled.connect(self._log_box.setVisible)

        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.addWidget(self._preview)
        v_split.addWidget(self._dashboard_panel)
        v_split.addWidget(log_group)
        v_split.setStretchFactor(0, 3)
        v_split.setStretchFactor(1, 2)
        v_split.setStretchFactor(2, 0)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.addWidget(scroll)
        h_split.addWidget(v_split)
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)
        outer.addWidget(h_split)

    def _build_settings(self, lay):
        self._build_presets(lay)

        self._detection = DetectionSection()
        lay.addWidget(self._detection)

        self._gaze_backend = GazeBackendSection()
        lay.addWidget(self._gaze_backend)

        self._ray = RaySection()
        lay.addWidget(self._ray)

        self._performance = PerformanceSection()
        lay.addWidget(self._performance)

        self._build_phenomena(lay)
        self._build_plugins(lay)

        self._output = OutputSection()
        lay.addWidget(self._output)

        lay.addStretch(1)
        self._build_start_stop(lay)

    # -- Presets row ----------------------------------------------------------

    def _build_presets(self, lay):
        row = _hrow()
        self._load_preset_btn = QPushButton("Load Preset")
        self._save_preset_btn = QPushButton("Save Preset")
        self._import_pipeline_btn = QPushButton("Import Pipeline")
        self._export_pipeline_btn = QPushButton("Export Pipeline")
        self._reset_defaults_btn = QPushButton("Reset Defaults")
        self._reset_defaults_btn.setToolTip(
            "Reset all gaze settings to their default values")
        self._reset_defaults_btn.clicked.connect(self._reset_gaze_defaults)
        row.layout().addWidget(self._load_preset_btn)
        row.layout().addWidget(self._save_preset_btn)
        row.layout().addWidget(self._import_pipeline_btn)
        row.layout().addWidget(self._export_pipeline_btn)
        row.layout().addWidget(self._reset_defaults_btn)
        lay.addWidget(row)

    # -- Phenomena / Plugin containers ----------------------------------------

    def _build_phenomena(self, lay):
        g = QGroupBox("Phenomena Tracking")
        self._phenomena_container = QVBoxLayout(g)
        self._phenomena_placeholder = QLabel(
            "Phenomena panel will be loaded here.")
        self._phenomena_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        self._phenomena_placeholder.setStyleSheet("color: #888;")
        self._phenomena_container.addWidget(self._phenomena_placeholder)
        lay.addWidget(g)

    def _build_plugins(self, lay):
        g = QGroupBox("Plugin Settings")
        self._plugin_container = QVBoxLayout(g)
        self._plugin_placeholder = QLabel(
            "Plugin panel will be loaded here.")
        self._plugin_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        self._plugin_placeholder.setStyleSheet("color: #888;")
        self._plugin_container.addWidget(self._plugin_placeholder)
        lay.addWidget(g)

    # -- Start / Stop ---------------------------------------------------------

    def _build_start_stop(self, lay):
        btn_row = _hrow()
        self._start_btn = QPushButton("\u25b6  Start")
        self._start_btn.setStyleSheet(
            "QPushButton{background:#2a7a2a;color:white;"
            "font-weight:bold;padding:6px;}")
        self._stop_btn = QPushButton("\u25a0  Stop")
        self._stop_btn.setStyleSheet(
            "QPushButton{background:#7a2a2a;color:white;"
            "font-weight:bold;padding:6px;}")
        self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        btn_row.layout().addWidget(self._start_btn, 1)
        btn_row.layout().addWidget(self._stop_btn, 1)
        lay.addWidget(btn_row)

    # -- Panel injection (called by MainWindow) -------------------------------

    def set_phenomena_panel(self, panel):
        self._phenomena_placeholder.setVisible(False)
        self._phenomena_container.removeWidget(self._phenomena_placeholder)
        self._phenomena_placeholder.deleteLater()
        self._phenomena_container.addWidget(panel)
        self._phenomena_panel = panel

    def set_plugin_panel(self, panel):
        self._plugin_placeholder.setVisible(False)
        self._plugin_container.removeWidget(self._plugin_placeholder)
        self._plugin_placeholder.deleteLater()
        self._plugin_container.addWidget(panel)
        self._plugin_panel = panel

    # -- VP file (called by MainWindow from VP Builder tab) -------------------

    def set_vp_file(self, path: str):
        self._detection.set_vp_file(path)

    # -- Reset defaults -------------------------------------------------------

    def _reset_gaze_defaults(self):
        self._ray.reset_defaults()
        self._performance.reset_defaults()

    # -- Namespace construction -----------------------------------------------

    def _build_namespace(self) -> Namespace:
        vals = {}
        vals.update(self._detection.namespace_values())
        vals.update(self._gaze_backend.namespace_values())
        vals.update(self._ray.namespace_values())
        vals.update(self._performance.namespace_values())
        vals.update(self._output.namespace_values())

        # Static defaults for phenomena (overridden by panel below)
        vals.update(
            ja_quorum=1.0,
            joint_attention=False,
            mutual_gaze=False,
            social_ref=False,
            social_ref_window=60,
            gaze_follow=False,
            gaze_follow_lag=30,
            gaze_aversion=False,
            aversion_window=60,
            aversion_conf=0.5,
            scanpath=False,
            scanpath_dwell=8,
            gaze_leader=False,
            attn_span=False,
            all_phenomena=False,
            ja_window=0,
            ja_window_thresh=0.70,
            pipeline=None,
            project=None,
        )

        ns = Namespace(**vals)

        if self._phenomena_panel is not None:
            for key, val in self._phenomena_panel.get_values().items():
                setattr(ns, key, val)
        if self._plugin_panel is not None:
            for key, val in self._plugin_panel.get_values().items():
                setattr(ns, key, val)

        return ns

    # -- Namespace application (preset / pipeline loading) --------------------

    def apply_namespace(self, ns: Namespace):
        self._detection.apply_namespace(ns)
        self._gaze_backend.apply_namespace(ns)
        self._ray.apply_namespace(ns)
        self._performance.apply_namespace(ns)
        self._output.apply_namespace(ns)

        if self._phenomena_panel is not None:
            self._phenomena_panel.apply_values(vars(ns))
        if self._plugin_panel is not None:
            self._plugin_panel.apply_values(vars(ns))

    # -- Start / Stop / Poll --------------------------------------------------

    def _start(self):
        if self._worker and self._worker.is_alive():
            return
        ns = self._build_namespace()

        if not ns.source:
            QMessageBox.critical(self, "Error", "Source is required.")
            return
        has_gaze = (ns.mgaze_model or ns.gazelle_model
                    or ns.l2cs_model or ns.unigaze_model)
        if not has_gaze:
            QMessageBox.critical(
                self, "Error", "Gaze model path is required.")
            return
        if ns.vp_file and not Path(ns.vp_file).exists():
            QMessageBox.critical(
                self, "Error", f"VP file not found:\n{ns.vp_file}")
            return

        self._frame_q = queue.Queue(maxsize=4)
        self._log_q = queue.Queue()
        from ..workers import GazeWorker
        self._worker = GazeWorker(ns, self._frame_q, self._log_q,
                                   dashboard_q=self._dashboard_q)
        self._worker.start()
        self._dashboard_panel.reset()
        self._dashboard_panel.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._append_log("Starting...")
        _poll_ms = 50 if getattr(ns, 'fast', False) else 30
        self._poll_timer.start(_poll_ms)

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self._poll_timer.stop()
        self._dashboard_panel.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _poll(self):
        try:
            while True:
                self._append_log(self._log_q.get_nowait())
        except queue.Empty:
            pass
        try:
            frame = self._frame_q.get_nowait()
            if frame is None:
                self._poll_timer.stop()
                self._start_btn.setEnabled(True)
                self._stop_btn.setEnabled(False)
                self._append_log("Stopped.")
                return
            pw = self._preview.width() or 640
            ph = self._preview.height() or 480
            self._preview.setPixmap(_bgr_to_pixmap(frame, pw, ph))
        except queue.Empty:
            pass
        if self._worker and not self._worker.is_alive():
            self._poll_timer.stop()
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)

    def _append_log(self, msg: str):
        self._log_box.append(msg)
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum())
