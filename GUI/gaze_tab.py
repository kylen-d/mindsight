"""
gaze_tab.py
-----------
Complete rewrite of the Gaze Tracker tab with full CLI feature parity.

Builds an ``argparse.Namespace`` instead of a raw dict, includes ALL CLI
parameters, and provides placeholder containers for PhenomenaPanel and
PluginPanel widgets (to be embedded later by the main window).
"""

from __future__ import annotations

import queue
from argparse import Namespace
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QTextEdit,
    QVBoxLayout, QWidget,
)

from .widgets import _hrow, _browse_btn, _bgr_to_pixmap, VP_EXT, _HERE


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Gaze Tracker
# ══════════════════════════════════════════════════════════════════════════════

class GazeTab(QWidget):
    """Full-featured Gaze Tracker tab with CLI parity."""

    GAZE_ARCHS = [
        "resnet18", "resnet34", "resnet50", "mobilenetv2",
        "mobileone_s0", "mobileone_s1", "mobileone_s2",
        "mobileone_s3", "mobileone_s4",
    ]
    GAZE_DATASETS = ["gaze360", "mpiigaze"]
    GAZELLE_NAMES = sorted([
        "gazelle_dinov2_vitb14", "gazelle_dinov2_vitl14",
        "gazelle_dinov2_vitb14_inout", "gazelle_dinov2_vitl14_inout",
    ])

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=4)
        self._log_q: queue.Queue = queue.Queue()
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll)

        # Placeholder references for panels injected later
        self._phenomena_panel = None
        self._plugin_panel = None

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # Left: scrollable settings panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_w = QWidget()
        settings_lay = QVBoxLayout(settings_w)
        settings_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        settings_lay.setSpacing(6)
        scroll.setWidget(settings_w)
        outer.addWidget(scroll)

        self._build_settings(settings_lay)

        # Right: preview + log
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(right, stretch=1)

        self._preview = QLabel()
        self._preview.setStyleSheet("background:#1a1a2e;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_lay.addWidget(self._preview, stretch=3)

        log_group = QGroupBox("Log")
        log_lay = QVBoxLayout(log_group)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setFixedHeight(130)
        self._log_box.setFont(QFont("Courier", 10))
        log_lay.addWidget(self._log_box)
        right_lay.addWidget(log_group)

    # ── Settings sections ────────────────────────────────────────────────────

    def _build_settings(self, lay):
        self._build_source(lay)
        self._build_detection(lay)
        self._build_gaze_backend(lay)
        self._build_ray_intersection(lay)
        self._build_tracking(lay)
        self._build_gaze_tips(lay)
        self._build_phenomena(lay)
        self._build_plugins(lay)
        self._build_output(lay)
        self._build_presets(lay)
        self._build_start_stop(lay)
        lay.addStretch(1)

    # ·· 1. Source ·····························································

    def _build_source(self, lay):
        g = QGroupBox("Source")
        f = QFormLayout(g)
        self._src = QLineEdit("0")
        btn = _browse_btn()
        btn.clicked.connect(self._browse_source)
        f.addRow("Source:", _hrow(self._src, btn))
        lay.addWidget(g)

    # ·· 2. Object Detection ··················································

    def _build_detection(self, lay):
        g = QGroupBox("Object Detection")
        vl = QVBoxLayout(g)

        mode_row = _hrow()
        self._rb_det_yolo = QRadioButton("YOLO (text classes)")
        self._rb_det_vp = QRadioButton("YOLOE Visual Prompt")
        self._rb_det_yolo.setChecked(True)
        mode_row.layout().addWidget(self._rb_det_yolo)
        mode_row.layout().addWidget(self._rb_det_vp)
        mode_row.layout().addStretch(1)
        vl.addWidget(mode_row)

        # YOLO sub-panel
        self._yolo_det_panel = QWidget()
        fp = QFormLayout(self._yolo_det_panel)
        fp.setContentsMargins(0, 0, 0, 0)
        self._yolo_model = QLineEdit("yolov8n.pt")
        btn2 = _browse_btn()
        btn2.clicked.connect(lambda: self._browse_to(self._yolo_model, "*.pt"))
        fp.addRow("Model:", _hrow(self._yolo_model, btn2))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.35)
        self._conf_spin.setDecimals(2)
        fp.addRow("Conf:", self._conf_spin)
        self._classes = QLineEdit()
        self._classes.setPlaceholderText("all (comma-separated to filter)")
        fp.addRow("Classes:", self._classes)
        self._blacklist = QLineEdit()
        self._blacklist.setPlaceholderText("none (comma-separated to suppress)")
        fp.addRow("Blacklist:", self._blacklist)
        vl.addWidget(self._yolo_det_panel)

        # YOLOE VP sub-panel
        self._vp_det_panel = QWidget()
        fv = QFormLayout(self._vp_det_panel)
        fv.setContentsMargins(0, 0, 0, 0)
        self._vp_file = QLineEdit()
        self._vp_file.setPlaceholderText(
            "Select .vp.json (build in VP Builder tab)")
        vp_btn = _browse_btn()
        vp_btn.clicked.connect(self._browse_vp_file)
        fv.addRow("VP file:", _hrow(self._vp_file, vp_btn))
        self._yoloe_model = QLineEdit("yoloe-26l-seg.pt")
        yoloe_btn = _browse_btn()
        yoloe_btn.clicked.connect(
            lambda: self._browse_to(self._yoloe_model, "*.pt"))
        fv.addRow("YOLOE model:", _hrow(self._yoloe_model, yoloe_btn))
        self._vp_conf_spin = QDoubleSpinBox()
        self._vp_conf_spin.setRange(0.05, 0.95)
        self._vp_conf_spin.setSingleStep(0.05)
        self._vp_conf_spin.setValue(0.35)
        self._vp_conf_spin.setDecimals(2)
        fv.addRow("Conf:", self._vp_conf_spin)
        self._vp_det_panel.setVisible(False)
        vl.addWidget(self._vp_det_panel)

        self._rb_det_yolo.toggled.connect(self._refresh_det_mode)
        self._rb_det_vp.toggled.connect(self._refresh_det_mode)
        lay.addWidget(g)

    # ·· 3. Gaze Backend ······················································

    def _build_gaze_backend(self, lay):
        g = QGroupBox("Gaze Backend")
        vl = QVBoxLayout(g)
        rb_row = _hrow()
        self._rb_onnx = QRadioButton("ONNX")
        self._rb_pytorch = QRadioButton("PyTorch")
        self._rb_gazelle = QRadioButton("Gazelle")
        self._rb_onnx.setChecked(True)
        rb_row.layout().addWidget(self._rb_onnx)
        rb_row.layout().addWidget(self._rb_pytorch)
        rb_row.layout().addWidget(self._rb_gazelle)
        rb_row.layout().addStretch(1)
        vl.addWidget(rb_row)

        self._gaze_model = QLineEdit(
            str(_HERE / "GazeTracking" / "gaze-estimation"
                / "weights" / "mobileone_s0_gaze.onnx"))
        gm_btn = _browse_btn()
        gm_btn.clicked.connect(
            lambda: self._browse_to(self._gaze_model, "*.onnx *.pt"))
        model_row = QWidget()
        model_lay = QFormLayout(model_row)
        model_lay.setContentsMargins(0, 0, 0, 0)
        model_lay.addRow("Model:", _hrow(self._gaze_model, gm_btn))
        vl.addWidget(model_row)

        self._arch_widget = QWidget()
        awl = QFormLayout(self._arch_widget)
        awl.setContentsMargins(0, 0, 0, 0)
        self._gaze_arch = QComboBox()
        self._gaze_arch.addItems(self.GAZE_ARCHS)
        self._gaze_arch.setCurrentText("mobileone_s0")
        self._gaze_dataset = QComboBox()
        self._gaze_dataset.addItems(self.GAZE_DATASETS)
        awl.addRow("Arch:", self._gaze_arch)
        awl.addRow("Dataset:", self._gaze_dataset)
        self._arch_widget.setVisible(False)
        vl.addWidget(self._arch_widget)

        self._gazelle_widget = QWidget()
        gwl = QFormLayout(self._gazelle_widget)
        gwl.setContentsMargins(0, 0, 0, 0)
        self._gazelle_name = QComboBox()
        self._gazelle_name.addItems(self.GAZELLE_NAMES)
        self._gazelle_ckpt = QLineEdit()
        gc_btn = _browse_btn()
        gc_btn.clicked.connect(
            lambda: self._browse_to(self._gazelle_ckpt, "*.pt"))
        self._gazelle_inout = QDoubleSpinBox()
        self._gazelle_inout.setRange(0.0, 1.0)
        self._gazelle_inout.setSingleStep(0.05)
        self._gazelle_inout.setValue(0.5)
        self._gazelle_inout.setDecimals(2)
        gwl.addRow("Variant:", self._gazelle_name)
        gwl.addRow("Checkpoint:", _hrow(self._gazelle_ckpt, gc_btn))
        gwl.addRow("InOut thr:", self._gazelle_inout)
        self._gazelle_widget.setVisible(False)
        vl.addWidget(self._gazelle_widget)

        self._rb_onnx.toggled.connect(self._refresh_backend)
        self._rb_pytorch.toggled.connect(self._refresh_backend)
        self._rb_gazelle.toggled.connect(self._refresh_backend)
        lay.addWidget(g)

    # ·· 4. Gaze Ray & Intersection ···········································

    def _build_ray_intersection(self, lay):
        g = QGroupBox("Gaze Ray && Intersection")
        f = QFormLayout(g)

        self._ray_length = QDoubleSpinBox()
        self._ray_length.setRange(0.2, 5.0)
        self._ray_length.setSingleStep(0.1)
        self._ray_length.setValue(1.0)
        self._ray_length.setDecimals(1)
        f.addRow("Ray length:", self._ray_length)

        self._cb_conf_ray = QCheckBox("Scale ray length by confidence")
        f.addRow(self._cb_conf_ray)

        self._gaze_cone = QDoubleSpinBox()
        self._gaze_cone.setRange(0.0, 45.0)
        self._gaze_cone.setSingleStep(1.0)
        self._gaze_cone.setValue(0.0)
        self._gaze_cone.setDecimals(1)
        self._gaze_cone.setToolTip(
            "Vision cone angle in degrees (0 = standard ray)")
        f.addRow("Gaze cone:", self._gaze_cone)

        self._adaptive_ray_cb = QCheckBox("Extend ray to nearest object")
        f.addRow(self._adaptive_ray_cb)

        self._adaptive_snap_cb = QCheckBox(
            "Snap endpoint to object center (requires adaptive ray)")
        f.addRow(self._adaptive_snap_cb)

        self._snap_dist = QSpinBox()
        self._snap_dist.setRange(20, 500)
        self._snap_dist.setValue(150)
        f.addRow("Snap dist:", self._snap_dist)

        self._snap_switch = QSpinBox()
        self._snap_switch.setRange(1, 30)
        self._snap_switch.setValue(8)
        self._snap_switch.setToolTip("Frames before snap switches target")
        f.addRow("Snap switch frames:", self._snap_switch)

        lay.addWidget(g)

    # ·· 5. Tracking ··························································

    def _build_tracking(self, lay):
        g = QGroupBox("Tracking")
        f = QFormLayout(g)

        self._gaze_lock_cb = QCheckBox("Enable gaze lock-on")
        self._gaze_lock_cb.setChecked(False)
        f.addRow(self._gaze_lock_cb)

        self._dwell_frames = QSpinBox()
        self._dwell_frames.setRange(1, 120)
        self._dwell_frames.setValue(15)
        f.addRow("Dwell frames:", self._dwell_frames)

        self._lock_dist = QSpinBox()
        self._lock_dist.setRange(20, 400)
        self._lock_dist.setValue(100)
        f.addRow("Lock dist:", self._lock_dist)

        self._skip_frames = QSpinBox()
        self._skip_frames.setRange(1, 10)
        self._skip_frames.setValue(1)
        f.addRow("Skip frames:", self._skip_frames)

        self._detect_scale = QDoubleSpinBox()
        self._detect_scale.setRange(0.25, 1.0)
        self._detect_scale.setSingleStep(0.05)
        self._detect_scale.setValue(1.0)
        self._detect_scale.setDecimals(2)
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

        lay.addWidget(g)

    # ·· 6. Gaze Tips ·························································

    def _build_gaze_tips(self, lay):
        self._gaze_tips_group = QGroupBox("Gaze tips (virtual objects)")
        self._gaze_tips_group.setCheckable(True)
        self._gaze_tips_group.setChecked(False)
        ft = QFormLayout(self._gaze_tips_group)
        self._tip_radius = QSpinBox()
        self._tip_radius.setRange(20, 300)
        self._tip_radius.setValue(80)
        ft.addRow("Tip radius:", self._tip_radius)
        lay.addWidget(self._gaze_tips_group)

    # ·· 7. Phenomena ·························································

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

    # ·· 8. Plugins ···························································

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

    # ·· 9. Output ·····························································

    def _build_output(self, lay):
        g = QGroupBox("Output")
        f = QFormLayout(g)

        self._cb_save = QCheckBox("Save annotated video")
        f.addRow(self._cb_save)

        self._log_path = QLineEdit()
        self._log_path.setPlaceholderText("optional — click Browse")
        lb = _browse_btn()
        lb.clicked.connect(
            lambda: self._browse_save(self._log_path, "CSV (*.csv)"))
        f.addRow("Event log:", _hrow(self._log_path, lb))

        self._summary_path = QLineEdit()
        self._summary_path.setPlaceholderText("optional — click Browse")
        sb = _browse_btn()
        sb.clicked.connect(
            lambda: self._browse_save(self._summary_path, "CSV (*.csv)"))
        f.addRow("Summary CSV:", _hrow(self._summary_path, sb))

        heatmap_row = QWidget()
        hl = QHBoxLayout(heatmap_row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        self._cb_heatmap = QCheckBox("Heatmap")
        hl.addWidget(self._cb_heatmap)
        self._heatmap_path = QLineEdit()
        self._heatmap_path.setPlaceholderText("optional — heatmap output path")
        hl.addWidget(self._heatmap_path, 1)
        hb = _browse_btn()
        hb.clicked.connect(
            lambda: self._browse_save(
                self._heatmap_path, "Image (*.png *.jpg);;All (*)"))
        hl.addWidget(hb)
        f.addRow(heatmap_row)

        lay.addWidget(g)

    # ·· 10. Presets ···························································

    def _build_presets(self, lay):
        row = _hrow()
        self._load_preset_btn = QPushButton("Load Preset")
        self._save_preset_btn = QPushButton("Save Preset")
        self._import_pipeline_btn = QPushButton("Import Pipeline")
        self._export_pipeline_btn = QPushButton("Export Pipeline")
        row.layout().addWidget(self._load_preset_btn)
        row.layout().addWidget(self._save_preset_btn)
        row.layout().addWidget(self._import_pipeline_btn)
        row.layout().addWidget(self._export_pipeline_btn)
        lay.addWidget(row)

    # ·· 11. Start / Stop ·····················································

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

    # ── Visibility helpers ───────────────────────────────────────────────────

    def _refresh_det_mode(self):
        vp = self._rb_det_vp.isChecked()
        self._yolo_det_panel.setVisible(not vp)
        self._vp_det_panel.setVisible(vp)

    def _refresh_backend(self):
        self._arch_widget.setVisible(self._rb_pytorch.isChecked())
        self._gazelle_widget.setVisible(self._rb_gazelle.isChecked())

    # ── Browse helpers ───────────────────────────────────────────────────────

    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source", "",
            "Video/Image (*.mp4 *.mov *.avi *.jpg *.jpeg *.png *.bmp)"
            ";;All (*)")
        if path:
            self._src.setText(path)

    def _browse_vp_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select VP file", "",
            f"Visual Prompt (*{VP_EXT});;JSON (*.json);;All (*)")
        if path:
            self._vp_file.setText(path)

    def _browse_to(self, line_edit: QLineEdit, filt: str = "*"):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", f"Files ({filt});;All (*)")
        if path:
            line_edit.setText(path)

    def _browse_save(self, line_edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getSaveFileName(self, "Save as", "", filt)
        if path:
            line_edit.setText(path)

    # ── Panel injection ──────────────────────────────────────────────────────

    def set_phenomena_panel(self, panel):
        """Replace the phenomena placeholder with an actual PhenomenaPanel."""
        self._phenomena_placeholder.setVisible(False)
        self._phenomena_container.removeWidget(self._phenomena_placeholder)
        self._phenomena_placeholder.deleteLater()
        self._phenomena_container.addWidget(panel)
        self._phenomena_panel = panel

    def set_plugin_panel(self, panel):
        """Replace the plugin placeholder with an actual PluginPanel."""
        self._plugin_placeholder.setVisible(False)
        self._plugin_container.removeWidget(self._plugin_placeholder)
        self._plugin_placeholder.deleteLater()
        self._plugin_container.addWidget(panel)
        self._plugin_panel = panel

    # ── Namespace construction ───────────────────────────────────────────────

    def _build_namespace(self) -> Namespace:
        """Build an argparse.Namespace matching CLI attribute names."""
        use_vp = self._rb_det_vp.isChecked()
        backend = ("gazelle" if self._rb_gazelle.isChecked()
                   else "pytorch" if self._rb_pytorch.isChecked()
                   else "onnx")
        gaze_model = (self._gazelle_ckpt.text() if backend == "gazelle"
                      else self._gaze_model.text())
        cls_raw = self._classes.text().strip()
        bl_raw = self._blacklist.text().strip()

        ns = Namespace(
            source=self._src.text().strip(),
            # Detection
            model=self._yolo_model.text().strip(),
            conf=(self._vp_conf_spin.value() if use_vp
                  else self._conf_spin.value()),
            classes=([c.strip() for c in cls_raw.split(",") if c.strip()]
                     or []),
            blacklist=[c.strip() for c in bl_raw.split(",") if c.strip()],
            detect_scale=self._detect_scale.value(),
            vp_file=self._vp_file.text().strip() if use_vp else None,
            vp_model=self._yoloe_model.text().strip(),
            skip_frames=self._skip_frames.value(),
            obj_persistence=self._obj_persistence.value(),
            # Gaze
            gaze_model=gaze_model.strip() if gaze_model else "",
            gaze_arch=(self._gaze_arch.currentText()
                       if backend == "pytorch" else None),
            gaze_dataset=self._gaze_dataset.currentText(),
            ray_length=self._ray_length.value(),
            adaptive_ray=self._adaptive_ray_cb.isChecked(),
            adaptive_snap=self._adaptive_snap_cb.isChecked(),
            snap_dist=float(self._snap_dist.value()),
            conf_ray=self._cb_conf_ray.isChecked(),
            gaze_tips=self._gaze_tips_group.isChecked(),
            tip_radius=self._tip_radius.value(),
            gaze_cone=self._gaze_cone.value(),
            gaze_lock=self._gaze_lock_cb.isChecked(),
            dwell_frames=self._dwell_frames.value(),
            lock_dist=self._lock_dist.value(),
            gaze_debug=self._cb_gaze_debug.isChecked(),
            snap_switch_frames=self._snap_switch.value(),
            reid_grace_seconds=self._reid_grace.value(),
            # JA (defaults -- phenomena panel will override)
            ja_conf_gate=0.0,
            ja_quorum=1.0,
            # Output
            save=self._cb_save.isChecked() or None,
            log=self._log_path.text().strip() or None,
            summary=self._summary_path.text().strip() or None,
            heatmap=(self._heatmap_path.text().strip()
                     if self._cb_heatmap.isChecked() else None),
            # Pipeline/project (not used from this tab directly)
            pipeline=None,
            project=None,
            # Phenomena defaults (will be overridden by phenomena panel)
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
            # Gazelle-specific
            gazelle_model=(self._gazelle_ckpt.text().strip()
                           if backend == "gazelle" else None),
            gazelle_name=self._gazelle_name.currentText(),
            gazelle_inout_threshold=self._gazelle_inout.value(),
        )

        # Merge phenomena panel values if available
        if (hasattr(self, '_phenomena_panel')
                and self._phenomena_panel is not None):
            for key, val in self._phenomena_panel.get_values().items():
                setattr(ns, key, val)

        # Merge plugin panel values if available
        if (hasattr(self, '_plugin_panel')
                and self._plugin_panel is not None):
            for key, val in self._plugin_panel.get_values().items():
                setattr(ns, key, val)

        return ns

    # ── Namespace application (for preset / pipeline loading) ────────────────

    def apply_namespace(self, ns: Namespace):
        """Populate all widgets from a namespace."""
        self._src.setText(str(getattr(ns, 'source', '0')))

        # Detection
        self._yolo_model.setText(str(getattr(ns, 'model', 'yolov8n.pt')))
        conf = getattr(ns, 'conf', 0.35)
        self._conf_spin.setValue(conf)
        self._vp_conf_spin.setValue(conf)

        classes = getattr(ns, 'classes', [])
        self._classes.setText(
            ", ".join(classes) if isinstance(classes, list) else "")
        blacklist = getattr(ns, 'blacklist', [])
        self._blacklist.setText(
            ", ".join(blacklist) if isinstance(blacklist, list) else "")

        vp_file = getattr(ns, 'vp_file', None)
        if vp_file:
            self._vp_file.setText(str(vp_file))
            self._rb_det_vp.setChecked(True)
        else:
            self._rb_det_yolo.setChecked(True)

        self._yoloe_model.setText(
            str(getattr(ns, 'vp_model', 'yoloe-26l-seg.pt')))
        self._detect_scale.setValue(getattr(ns, 'detect_scale', 1.0))
        self._skip_frames.setValue(getattr(ns, 'skip_frames', 1))
        self._obj_persistence.setValue(getattr(ns, 'obj_persistence', 0))

        # Gaze backend
        gazelle_model = getattr(ns, 'gazelle_model', None)
        gaze_arch = getattr(ns, 'gaze_arch', None)
        if gazelle_model:
            self._rb_gazelle.setChecked(True)
            self._gazelle_ckpt.setText(str(gazelle_model))
        elif gaze_arch:
            self._rb_pytorch.setChecked(True)
            self._gaze_arch.setCurrentText(str(gaze_arch))
        else:
            self._rb_onnx.setChecked(True)

        gaze_model = getattr(ns, 'gaze_model', '')
        if gaze_model:
            self._gaze_model.setText(str(gaze_model))

        self._gaze_dataset.setCurrentText(
            str(getattr(ns, 'gaze_dataset', 'gaze360')))
        self._gazelle_name.setCurrentText(
            str(getattr(ns, 'gazelle_name', self.GAZELLE_NAMES[0])))
        self._gazelle_inout.setValue(
            getattr(ns, 'gazelle_inout_threshold', 0.5))

        # Ray & intersection
        self._ray_length.setValue(getattr(ns, 'ray_length', 1.0))
        self._cb_conf_ray.setChecked(bool(getattr(ns, 'conf_ray', False)))
        self._gaze_cone.setValue(getattr(ns, 'gaze_cone', 0.0))
        self._adaptive_ray_cb.setChecked(
            bool(getattr(ns, 'adaptive_ray', False)))
        self._adaptive_snap_cb.setChecked(
            bool(getattr(ns, 'adaptive_snap', False)))
        self._snap_dist.setValue(int(getattr(ns, 'snap_dist', 150)))
        self._snap_switch.setValue(
            int(getattr(ns, 'snap_switch_frames', 8)))

        # Tracking
        self._gaze_lock_cb.setChecked(
            bool(getattr(ns, 'gaze_lock', False)))
        self._dwell_frames.setValue(getattr(ns, 'dwell_frames', 15))
        self._lock_dist.setValue(getattr(ns, 'lock_dist', 100))
        self._reid_grace.setValue(
            getattr(ns, 'reid_grace_seconds', 1.0))
        self._cb_gaze_debug.setChecked(
            bool(getattr(ns, 'gaze_debug', False)))

        # Gaze tips
        self._gaze_tips_group.setChecked(
            bool(getattr(ns, 'gaze_tips', False)))
        self._tip_radius.setValue(getattr(ns, 'tip_radius', 80))

        # Output
        self._cb_save.setChecked(bool(getattr(ns, 'save', False)))
        self._log_path.setText(str(getattr(ns, 'log', '') or ''))
        self._summary_path.setText(str(getattr(ns, 'summary', '') or ''))
        heatmap = getattr(ns, 'heatmap', None)
        if heatmap:
            self._cb_heatmap.setChecked(True)
            self._heatmap_path.setText(str(heatmap))
        else:
            self._cb_heatmap.setChecked(False)
            self._heatmap_path.setText('')

        # Delegate to sub-panels
        if self._phenomena_panel is not None:
            self._phenomena_panel.apply_values(vars(ns))
        if self._plugin_panel is not None:
            self._plugin_panel.apply_values(vars(ns))

    # ── Start / Stop / Poll ──────────────────────────────────────────────────

    def _start(self):
        if self._worker and self._worker.is_alive():
            return
        ns = self._build_namespace()

        # Validation
        if not ns.source:
            QMessageBox.critical(self, "Error", "Source is required.")
            return
        if not ns.gaze_model and not ns.gazelle_model:
            QMessageBox.critical(
                self, "Error", "Gaze model path is required.")
            return
        if ns.vp_file and not Path(ns.vp_file).exists():
            QMessageBox.critical(
                self, "Error", f"VP file not found:\n{ns.vp_file}")
            return

        self._frame_q = queue.Queue(maxsize=4)
        self._log_q = queue.Queue()
        from .workers import GazeWorker
        self._worker = GazeWorker(ns, self._frame_q, self._log_q)
        self._worker.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._append_log("Starting...")
        self._poll_timer.start(30)

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self._poll_timer.stop()
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

    # ── Called externally to pre-fill a VP file (e.g. from VP Builder tab) ──

    def set_vp_file(self, path: str):
        """Set the VP file path and switch detection mode to YOLOE VP."""
        self._vp_file.setText(path)
        self._rb_det_vp.setChecked(True)
