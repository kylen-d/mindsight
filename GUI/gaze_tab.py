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
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from GazeTracking.Backends.L2CS.L2CS_Config import (
    ARCH_CHOICES as L2CS_ARCHS,
)
from GazeTracking.Backends.L2CS.L2CS_Config import (
    DEFAULT_MODEL as L2CS_DEFAULT_MODEL,
)
from GazeTracking.Backends.MGaze.MGaze_Config import DEFAULT_ONNX_MODEL

from .widgets import VP_EXT, _bgr_to_pixmap, _browse_btn, _hrow

# Check if UniGaze is available (optional dependency)
try:
    from GazeTracking.Backends.UniGaze.UniGaze_Config import (
        DEFAULT_VARIANT as UNIGAZE_DEFAULT,
    )
    from GazeTracking.Backends.UniGaze.UniGaze_Config import (
        MODEL_VARIANTS as UNIGAZE_VARIANTS,
    )
    _UNIGAZE_AVAILABLE = True
except ImportError:
    _UNIGAZE_AVAILABLE = False


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
        self._dashboard_q: queue.Queue = queue.Queue(maxsize=30)
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

        from PyQt6.QtWidgets import QSplitter

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

        # Live dashboard panel
        from .live_dashboard import LiveDashboardPanel
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
        v_split.setStretchFactor(0, 3)   # video gets most space
        v_split.setStretchFactor(1, 2)   # dashboard
        v_split.setStretchFactor(2, 0)   # log starts collapsed

        # Horizontal splitter: settings | video+dashboard+log
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.addWidget(scroll)
        h_split.addWidget(v_split)
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)
        outer.addWidget(h_split)

    # ── Settings sections ────────────────────────────────────────────────────

    def _build_settings(self, lay):
        self._build_presets(lay)
        self._build_source(lay)
        self._build_detection(lay)
        self._build_gaze_backend(lay)
        self._build_ray_geometry(lay)
        self._build_adaptive_snap(lay)
        self._build_fixation_lockon(lay)
        self._build_hit_detection(lay)
        self._build_performance(lay)
        self._build_phenomena(lay)
        self._build_plugins(lay)
        self._build_output(lay)
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

        # Global device selector (applies to all backends)
        common = QFormLayout()
        common.setContentsMargins(0, 8, 0, 0)
        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self._device_combo.setToolTip(
            "Compute device for all backends.  "
            "'auto' selects CUDA > MPS > CPU.")
        common.addRow("Device:", self._device_combo)
        vl.addLayout(common)

        self._rb_det_yolo.toggled.connect(self._refresh_det_mode)
        self._rb_det_vp.toggled.connect(self._refresh_det_mode)
        lay.addWidget(g)

    # ·· 3. Gaze Backend ······················································

    def _build_gaze_backend(self, lay):
        g = QGroupBox("Gaze Backend")
        vl = QVBoxLayout(g)

        # Row 1: MGaze, L2CS, UniGaze, Gazelle
        rb_row1 = _hrow()
        self._rb_mgaze = QRadioButton("MGaze")
        self._rb_mgaze.setChecked(True)
        rb_row1.layout().addWidget(self._rb_mgaze)
        self._rb_l2cs = QRadioButton("L2CS-Net")
        rb_row1.layout().addWidget(self._rb_l2cs)
        self._rb_unigaze = QRadioButton("UniGaze")
        if not _UNIGAZE_AVAILABLE:
            self._rb_unigaze.setEnabled(False)
            self._rb_unigaze.setToolTip(
                "Requires: pip install unigaze timm==0.3.2")
        rb_row1.layout().addWidget(self._rb_unigaze)
        self._rb_gazelle = QRadioButton("Gazelle")
        rb_row1.layout().addWidget(self._rb_gazelle)
        rb_row1.layout().addStretch(1)
        vl.addWidget(rb_row1)

        # -- MGaze model path + inference mode --
        self._mgaze_widget = QWidget()
        mgl = QVBoxLayout(self._mgaze_widget)
        mgl.setContentsMargins(0, 0, 0, 0)
        self._gaze_model = QLineEdit(DEFAULT_ONNX_MODEL)
        gm_btn = _browse_btn()
        gm_btn.clicked.connect(
            lambda: self._browse_to(self._gaze_model, "*.onnx *.pt"))
        model_row = QWidget()
        model_lay = QFormLayout(model_row)
        model_lay.setContentsMargins(0, 0, 0, 0)
        model_lay.addRow("Model:", _hrow(self._gaze_model, gm_btn))
        self._gaze_model.textChanged.connect(self._refresh_backend)
        mgl.addWidget(model_row)
        vl.addWidget(self._mgaze_widget)

        # -- MGaze arch/dataset (PyTorch inference mode only) --
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

        # -- L2CS-Net widget --
        self._l2cs_widget = QWidget()
        l2l = QFormLayout(self._l2cs_widget)
        l2l.setContentsMargins(0, 0, 0, 0)
        self._l2cs_model = QLineEdit(L2CS_DEFAULT_MODEL)
        l2m_btn = _browse_btn()
        l2m_btn.clicked.connect(
            lambda: self._browse_to(self._l2cs_model, "*.pkl *.pt *.onnx"))
        l2l.addRow("Model:", _hrow(self._l2cs_model, l2m_btn))
        self._l2cs_arch = QComboBox()
        self._l2cs_arch.addItems(L2CS_ARCHS)
        self._l2cs_arch.setCurrentText("ResNet50")
        self._l2cs_dataset = QComboBox()
        self._l2cs_dataset.addItems(self.GAZE_DATASETS)
        l2l.addRow("Arch:", self._l2cs_arch)
        l2l.addRow("Dataset:", self._l2cs_dataset)
        self._l2cs_widget.setVisible(False)
        vl.addWidget(self._l2cs_widget)

        # -- UniGaze widget --
        self._unigaze_widget = QWidget()
        ugl = QFormLayout(self._unigaze_widget)
        ugl.setContentsMargins(0, 0, 0, 0)
        self._unigaze_variant = QComboBox()
        if _UNIGAZE_AVAILABLE:
            self._unigaze_variant.addItems(list(UNIGAZE_VARIANTS.keys()))
            self._unigaze_variant.setCurrentText(UNIGAZE_DEFAULT)
        ugl.addRow("Variant:", self._unigaze_variant)
        self._unigaze_widget.setVisible(False)
        vl.addWidget(self._unigaze_widget)

        # -- Gazelle widget --
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
        self._gazelle_device = QComboBox()
        self._gazelle_device.addItems(["auto", "cpu", "cuda", "mps"])
        self._gazelle_skip = QSpinBox()
        self._gazelle_skip.setRange(0, 30)
        self._gazelle_skip.setValue(0)
        self._gazelle_skip.setToolTip(
            "Reuse the previous gaze result for N frames between "
            "inference runs. 0 = run every frame.")
        self._gazelle_fp16 = QCheckBox("FP16 (half-precision)")
        self._gazelle_fp16.setToolTip(
            "Use float16 inference on CUDA/MPS for ~1.5-2× speedup. "
            "Ignored on CPU.")
        self._gazelle_compile = QCheckBox("torch.compile()")
        self._gazelle_compile.setToolTip(
            "Use torch.compile() for 10-30% speedup after warmup. "
            "Requires PyTorch 2.0+.")
        gwl.addRow("Variant:", self._gazelle_name)
        gwl.addRow("Checkpoint:", _hrow(self._gazelle_ckpt, gc_btn))
        gwl.addRow("InOut thr:", self._gazelle_inout)
        gwl.addRow("Device:", self._gazelle_device)
        gwl.addRow("Skip frames:", self._gazelle_skip)
        gwl.addRow("", self._gazelle_fp16)
        gwl.addRow("", self._gazelle_compile)
        self._gazelle_widget.setVisible(False)
        vl.addWidget(self._gazelle_widget)

        # Enforce mutual exclusion across all backend radio buttons
        self._backend_group = QButtonGroup(self)
        self._backend_group.addButton(self._rb_mgaze)
        self._backend_group.addButton(self._rb_l2cs)
        self._backend_group.addButton(self._rb_unigaze)
        self._backend_group.addButton(self._rb_gazelle)

        self._rb_mgaze.toggled.connect(self._refresh_backend)
        self._rb_l2cs.toggled.connect(self._refresh_backend)
        self._rb_unigaze.toggled.connect(self._refresh_backend)
        self._rb_gazelle.toggled.connect(self._refresh_backend)
        lay.addWidget(g)

    # ·· 4. Gaze Ray Geometry ·················································

    def _build_ray_geometry(self, lay):
        g = QGroupBox("Gaze Ray Geometry")
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

        self._forward_gaze_thresh = QDoubleSpinBox()
        self._forward_gaze_thresh.setRange(0.0, 30.0)
        self._forward_gaze_thresh.setSingleStep(0.5)
        self._forward_gaze_thresh.setValue(5.0)
        self._forward_gaze_thresh.setDecimals(1)
        self._forward_gaze_thresh.setToolTip(
            "Pitch/yaw angles below this threshold (degrees) are treated as\n"
            "looking forward (at camera). Set to 0 to disable.")
        f.addRow("Forward threshold (\u00b0):", self._forward_gaze_thresh)

        # Gaze tips (nested checkable sub-group)
        self._gaze_tips_group = QGroupBox("Gaze tips (virtual objects)")
        self._gaze_tips_group.setCheckable(True)
        self._gaze_tips_group.setChecked(False)
        ft = QFormLayout(self._gaze_tips_group)
        self._tip_radius = QSpinBox()
        self._tip_radius.setRange(20, 300)
        self._tip_radius.setValue(80)
        ft.addRow("Tip radius:", self._tip_radius)
        f.addRow(self._gaze_tips_group)

        lay.addWidget(g)

    # ·· 5. Adaptive Snap ·····················································

    def _build_adaptive_snap(self, lay):
        self._adaptive_snap_group = QGroupBox("Adaptive Snap")
        self._adaptive_snap_group.setCheckable(True)
        self._adaptive_snap_group.setChecked(False)
        f = QFormLayout(self._adaptive_snap_group)

        self._adaptive_mode_combo = QComboBox()
        self._adaptive_mode_combo.addItems(["Extend", "Snap"])
        self._adaptive_mode_combo.setToolTip(
            "Extend: ray freely extends toward the nearest object.\n"
            "Snap: ray locks to the object centre.")
        f.addRow("Mode:", self._adaptive_mode_combo)

        self._snap_dist = QSpinBox()
        self._snap_dist.setRange(20, 500)
        self._snap_dist.setValue(150)
        f.addRow("Snap dist (px):", self._snap_dist)

        self._snap_bbox_scale = QDoubleSpinBox()
        self._snap_bbox_scale.setRange(0.0, 2.0)
        self._snap_bbox_scale.setSingleStep(0.1)
        self._snap_bbox_scale.setValue(0.0)
        self._snap_bbox_scale.setDecimals(2)
        self._snap_bbox_scale.setToolTip(
            "Fraction of bbox half-diagonal added to effective snap radius")
        f.addRow("Bbox scale:", self._snap_bbox_scale)

        # -- Scoring Weights sub-heading --
        lbl_scoring = QLabel("Scoring Weights")
        lbl_scoring.setStyleSheet("color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_scoring)

        self._snap_w_dist = QDoubleSpinBox()
        self._snap_w_dist.setRange(0.0, 3.0)
        self._snap_w_dist.setSingleStep(0.1)
        self._snap_w_dist.setValue(1.0)
        self._snap_w_dist.setDecimals(2)
        self._snap_w_dist.setToolTip("Scoring weight for normalized distance penalty")
        f.addRow("W distance:", self._snap_w_dist)

        self._snap_w_size = QDoubleSpinBox()
        self._snap_w_size.setRange(0.0, 3.0)
        self._snap_w_size.setSingleStep(0.1)
        self._snap_w_size.setValue(0.0)
        self._snap_w_size.setDecimals(2)
        self._snap_w_size.setToolTip("Scoring weight for angular size reward")
        f.addRow("W size:", self._snap_w_size)

        self._snap_w_intersect = QDoubleSpinBox()
        self._snap_w_intersect.setRange(0.0, 3.0)
        self._snap_w_intersect.setSingleStep(0.1)
        self._snap_w_intersect.setValue(0.5)
        self._snap_w_intersect.setDecimals(2)
        self._snap_w_intersect.setToolTip("Scoring bonus for ray-bbox intersection")
        f.addRow("W intersect:", self._snap_w_intersect)

        # -- Stabilization sub-heading --
        lbl_stab = QLabel("Stabilization")
        lbl_stab.setStyleSheet("color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_stab)

        self._snap_switch = QSpinBox()
        self._snap_switch.setRange(1, 30)
        self._snap_switch.setValue(8)
        self._snap_switch.setToolTip("Frames before snap switches target")
        f.addRow("Snap switch frames:", self._snap_switch)

        lay.addWidget(self._adaptive_snap_group)

    # ·· 6. Fixation Lock-On ··················································

    def _build_fixation_lockon(self, lay):
        self._fixation_group = QGroupBox("Fixation Lock-On")
        self._fixation_group.setCheckable(True)
        self._fixation_group.setChecked(False)
        f = QFormLayout(self._fixation_group)

        self._dwell_frames = QSpinBox()
        self._dwell_frames.setRange(1, 120)
        self._dwell_frames.setValue(15)
        self._dwell_frames.setToolTip(
            "Consecutive frames the gaze ray must stay near an object "
            "to trigger lock-on")
        f.addRow("Dwell frames:", self._dwell_frames)

        self._lock_dist = QSpinBox()
        self._lock_dist.setRange(20, 400)
        self._lock_dist.setValue(100)
        self._lock_dist.setToolTip(
            "Distance (px) from gaze ray to object centre for dwell counting")
        f.addRow("Lock dist (px):", self._lock_dist)

        lay.addWidget(self._fixation_group)

    # ·· 7. Hit Detection ·····················································

    def _build_hit_detection(self, lay):
        g = QGroupBox("Hit Detection")
        f = QFormLayout(g)

        self._hit_conf_gate = QDoubleSpinBox()
        self._hit_conf_gate.setRange(0.0, 1.0)
        self._hit_conf_gate.setSingleStep(0.05)
        self._hit_conf_gate.setValue(0.0)
        self._hit_conf_gate.setDecimals(2)
        self._hit_conf_gate.setToolTip(
            "Minimum per-face gaze confidence for ray-object hit detection.\n"
            "0 = disabled (all faces participate).")
        f.addRow("Hit conf gate:", self._hit_conf_gate)

        self._detect_extend = QSpinBox()
        self._detect_extend.setRange(0, 20000)
        self._detect_extend.setSingleStep(50)
        self._detect_extend.setValue(0)
        self._detect_extend.setToolTip(
            "Extend gaze-object detection N pixels past the visual\n"
            "ray/cone endpoint. 0 = detection matches visual exactly.")
        f.addRow("Extend detection (px):", self._detect_extend)

        self._detect_extend_scope = QComboBox()
        self._detect_extend_scope.addItems(["Objects", "Phenomena", "Both"])
        self._detect_extend_scope.setToolTip(
            "Scope for detection extension:\n"
            "Objects: extends ray-object hit detection only.\n"
            "Phenomena: extends phenomena tracking (mutual gaze, social ref) only.\n"
            "Both: extends both.")
        f.addRow("Extend scope:", self._detect_extend_scope)

        lay.addWidget(g)

    # ·· 8. Performance & Tracking ·············································

    def _build_performance(self, lay):
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

        self._cb_charts = QCheckBox("Generate post-run charts")
        self._cb_charts.setToolTip(
            "Save time-series charts for each active phenomena tracker "
            "alongside the summary CSV output.")
        f.addRow(self._cb_charts)

        anon_row = QWidget()
        al = QHBoxLayout(anon_row)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(4)
        self._cb_anonymize = QCheckBox("Anonymize faces")
        al.addWidget(self._cb_anonymize)
        self._anonymize_mode = QComboBox()
        self._anonymize_mode.addItems(["blur", "black"])
        self._anonymize_mode.setEnabled(False)
        al.addWidget(self._anonymize_mode)
        self._cb_anonymize.toggled.connect(self._anonymize_mode.setEnabled)
        f.addRow(anon_row)

        self._participant_ids = QLineEdit()
        self._participant_ids.setPlaceholderText("e.g. S70,S71,S72 (positional)")
        f.addRow("Participant IDs:", self._participant_ids)

        # Auxiliary Streams (collapsed by default)
        aux_grp = QGroupBox("Auxiliary Streams")
        aux_grp.setCheckable(True)
        aux_grp.setChecked(False)
        aux_lay = QVBoxLayout(aux_grp)

        self._aux_table = QTableWidget(0, 3)
        self._aux_table.setHorizontalHeaderLabels(["PID", "Type", "Source"])
        self._aux_table.horizontalHeader().setStretchLastSection(True)
        self._aux_table.setMinimumHeight(100)
        aux_lay.addWidget(self._aux_table)

        aux_btn_row = _hrow()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self._aux_add_row)
        rm_btn = QPushButton("Remove Row")
        rm_btn.clicked.connect(self._aux_remove_row)
        browse_btn = QPushButton("Browse Source…")
        browse_btn.clicked.connect(self._aux_browse_source)
        aux_btn_row.layout().addWidget(add_btn)
        aux_btn_row.layout().addWidget(rm_btn)
        aux_btn_row.layout().addWidget(browse_btn)
        aux_lay.addWidget(aux_btn_row)

        f.addRow(aux_grp)

        lay.addWidget(g)

    # ·· 10. Presets ···························································

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

    # ── Reset defaults ───────────────────────────────────────────────────────

    def _reset_gaze_defaults(self):
        """Reset all gaze-related settings to their default values."""
        # Ray geometry
        self._ray_length.setValue(1.0)
        self._cb_conf_ray.setChecked(False)
        self._gaze_cone.setValue(0.0)
        self._forward_gaze_thresh.setValue(5.0)
        self._gaze_tips_group.setChecked(False)
        self._tip_radius.setValue(80)

        # Adaptive snap
        self._adaptive_snap_group.setChecked(False)
        self._adaptive_mode_combo.setCurrentIndex(0)   # Extend
        self._snap_dist.setValue(150)
        self._snap_bbox_scale.setValue(0.0)
        self._snap_w_dist.setValue(1.0)
        self._snap_w_size.setValue(0.0)
        self._snap_w_intersect.setValue(0.5)
        self._snap_switch.setValue(8)

        # Fixation lock-on
        self._fixation_group.setChecked(False)
        self._dwell_frames.setValue(15)
        self._lock_dist.setValue(100)

        # Hit detection
        self._hit_conf_gate.setValue(0.0)
        self._detect_extend.setValue(0)
        self._detect_extend_scope.setCurrentIndex(0)

        # Performance & tracking
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

    # ── Visibility helpers ───────────────────────────────────────────────────

    def _refresh_det_mode(self):
        vp = self._rb_det_vp.isChecked()
        self._yolo_det_panel.setVisible(not vp)
        self._vp_det_panel.setVisible(vp)

    def _refresh_backend(self):
        is_mgaze = self._rb_mgaze.isChecked()
        self._mgaze_widget.setVisible(is_mgaze)
        # Show arch/dataset fields when an MGaze .pt model is selected
        mgaze_path = self._gaze_model.text().strip().lower()
        self._arch_widget.setVisible(is_mgaze and mgaze_path.endswith('.pt'))
        self._l2cs_widget.setVisible(self._rb_l2cs.isChecked())
        self._unigaze_widget.setVisible(self._rb_unigaze.isChecked())
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

    # ── Auxiliary stream table helpers ──────────────────────────────────────

    def _aux_add_row(self):
        row = self._aux_table.rowCount()
        self._aux_table.insertRow(row)
        self._aux_table.setItem(row, 0, QTableWidgetItem(""))
        self._aux_table.setItem(row, 1, QTableWidgetItem("eye_camera"))
        self._aux_table.setItem(row, 2, QTableWidgetItem(""))

    def _aux_remove_row(self):
        row = self._aux_table.currentRow()
        if row >= 0:
            self._aux_table.removeRow(row)

    def _aux_browse_source(self):
        row = self._aux_table.currentRow()
        if row < 0:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Auxiliary Video",
            filter="Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*)")
        if path:
            self._aux_table.setItem(row, 2, QTableWidgetItem(path))

    def _aux_stream_configs(self):
        """Read the aux table into a list of AuxStreamConfig or None."""
        from pipeline_config import AuxStreamConfig
        configs = []
        for r in range(self._aux_table.rowCount()):
            pid = (self._aux_table.item(r, 0).text().strip()
                   if self._aux_table.item(r, 0) else "")
            stype = (self._aux_table.item(r, 1).text().strip()
                     if self._aux_table.item(r, 1) else "")
            source = (self._aux_table.item(r, 2).text().strip()
                      if self._aux_table.item(r, 2) else "")
            if pid and stype and source:
                configs.append(AuxStreamConfig(pid=pid, stream_type=stype,
                                               source=source))
        return configs if configs else None

    # ── Namespace construction ───────────────────────────────────────────────

    def _build_namespace(self) -> Namespace:
        """Build an argparse.Namespace matching CLI attribute names."""
        use_vp = self._rb_det_vp.isChecked()
        if self._rb_gazelle.isChecked():
            backend = "gazelle"
        elif self._rb_l2cs.isChecked():
            backend = "l2cs"
        elif self._rb_unigaze.isChecked():
            backend = "unigaze"
        else:
            backend = "mgaze"

        cls_raw = self._classes.text().strip()
        bl_raw = self._blacklist.text().strip()

        # Build MGaze model path (auto-detects ONNX vs PyTorch from extension)
        mgaze_model = ""
        if backend == "mgaze":
            mgaze_model = self._gaze_model.text().strip()

        ns = Namespace(
            source=self._src.text().strip(),
            device=self._device_combo.currentText(),
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
            # Gaze -- MGaze
            mgaze_model=mgaze_model,
            mgaze_arch=(self._gaze_arch.currentText()
                        if backend == "mgaze" and mgaze_model.lower().endswith('.pt')
                        else None),
            mgaze_dataset=self._gaze_dataset.currentText(),
            # Gaze -- L2CS-Net
            l2cs_model=(self._l2cs_model.text().strip()
                        if backend == "l2cs" else None),
            l2cs_arch=self._l2cs_arch.currentText(),
            l2cs_dataset=self._l2cs_dataset.currentText(),
            # Gaze -- UniGaze
            unigaze_model=(self._unigaze_variant.currentText()
                           if backend == "unigaze" else None),
            ray_length=self._ray_length.value(),
            adaptive_ray=(self._adaptive_mode_combo.currentText().lower()
                         if self._adaptive_snap_group.isChecked()
                         else "off"),
            snap_dist=float(self._snap_dist.value()),
            snap_bbox_scale=self._snap_bbox_scale.value(),
            snap_w_dist=self._snap_w_dist.value(),
            snap_w_size=self._snap_w_size.value(),
            snap_w_intersect=self._snap_w_intersect.value(),
            conf_ray=self._cb_conf_ray.isChecked(),
            gaze_tips=self._gaze_tips_group.isChecked(),
            tip_radius=self._tip_radius.value(),
            gaze_cone=self._gaze_cone.value(),
            gaze_lock=self._fixation_group.isChecked(),
            dwell_frames=self._dwell_frames.value(),
            lock_dist=self._lock_dist.value(),
            gaze_debug=self._cb_gaze_debug.isChecked(),
            snap_switch_frames=self._snap_switch.value(),
            reid_grace_seconds=self._reid_grace.value(),
            forward_gaze_threshold=self._forward_gaze_thresh.value(),
            hit_conf_gate=self._hit_conf_gate.value(),
            detect_extend=float(self._detect_extend.value()),
            detect_extend_scope=self._detect_extend_scope.currentText().lower(),
            ja_quorum=1.0,
            # Output
            save=self._cb_save.isChecked() or None,
            log=self._log_path.text().strip() or None,
            summary=self._summary_path.text().strip() or None,
            heatmap=(self._heatmap_path.text().strip()
                     if self._cb_heatmap.isChecked() else None),
            charts=True if self._cb_charts.isChecked() else None,
            anonymize=(self._anonymize_mode.currentText()
                       if self._cb_anonymize.isChecked() else None),
            anonymize_padding=0.3,
            # Participant IDs
            participant_ids=(self._participant_ids.text().strip() or None),
            participant_csv=None,
            # Auxiliary streams
            aux_streams=self._aux_stream_configs(),
            aux_streams_raw=None,
            # Pipeline/project (not used from this tab directly)
            pipeline=None,
            project=None,
            # Phenomena defaults (will be overridden by phenomena panel)
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
            # Gazelle-specific
            gazelle_model=(self._gazelle_ckpt.text().strip()
                           if backend == "gazelle" else None),
            gazelle_name=self._gazelle_name.currentText(),
            gazelle_inout_threshold=self._gazelle_inout.value(),
            gazelle_device=self._gazelle_device.currentText(),
            gazelle_skip_frames=self._gazelle_skip.value(),
            gazelle_fp16=self._gazelle_fp16.isChecked(),
            gazelle_compile=self._gazelle_compile.isChecked(),
            # Performance
            fast=self._cb_fast.isChecked(),
            skip_phenomena=self._skip_phenomena.value(),
            lite_overlay=self._cb_lite_overlay.isChecked(),
            no_dashboard=self._cb_no_dashboard.isChecked(),
            profile=self._cb_profile.isChecked(),
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

        # Device
        self._device_combo.setCurrentText(
            str(getattr(ns, 'device', 'auto')))

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

        # Gaze backend selection
        gazelle_model = getattr(ns, 'gazelle_model', None)
        l2cs_model = getattr(ns, 'l2cs_model', None)
        unigaze_model = getattr(ns, 'unigaze_model', None)
        mgaze_arch = getattr(ns, 'mgaze_arch', None)
        if gazelle_model:
            self._rb_gazelle.setChecked(True)
            self._gazelle_ckpt.setText(str(gazelle_model))
        elif l2cs_model:
            self._rb_l2cs.setChecked(True)
            self._l2cs_model.setText(str(l2cs_model))
        elif unigaze_model:
            self._rb_unigaze.setChecked(True)
            self._unigaze_variant.setCurrentText(str(unigaze_model))
        elif mgaze_arch:
            self._rb_mgaze.setChecked(True)
            self._gaze_arch.setCurrentText(str(mgaze_arch))
        else:
            self._rb_mgaze.setChecked(True)

        # MGaze fields
        gaze_model = getattr(ns, 'mgaze_model', '')
        if gaze_model:
            self._gaze_model.setText(str(gaze_model))
        self._gaze_dataset.setCurrentText(
            str(getattr(ns, 'mgaze_dataset', 'gaze360')))

        # L2CS fields
        if l2cs_model:
            self._l2cs_model.setText(str(l2cs_model))
        l2cs_arch = getattr(ns, 'l2cs_arch', 'ResNet50')
        if l2cs_arch:
            self._l2cs_arch.setCurrentText(str(l2cs_arch))
        self._l2cs_dataset.setCurrentText(
            str(getattr(ns, 'l2cs_dataset', 'gaze360')))
        self._gazelle_name.setCurrentText(
            str(getattr(ns, 'gazelle_name', self.GAZELLE_NAMES[0])))
        self._gazelle_inout.setValue(
            getattr(ns, 'gazelle_inout_threshold', 0.5))
        self._gazelle_device.setCurrentText(
            str(getattr(ns, 'gazelle_device', 'auto')))
        self._gazelle_skip.setValue(
            int(getattr(ns, 'gazelle_skip_frames', 0)))
        self._gazelle_fp16.setChecked(
            bool(getattr(ns, 'gazelle_fp16', False)))
        self._gazelle_compile.setChecked(
            bool(getattr(ns, 'gazelle_compile', False)))

        # Ray & intersection
        self._ray_length.setValue(getattr(ns, 'ray_length', 1.0))
        self._cb_conf_ray.setChecked(bool(getattr(ns, 'conf_ray', False)))
        self._gaze_cone.setValue(getattr(ns, 'gaze_cone', 0.0))
        ar = getattr(ns, 'adaptive_ray', 'off')
        # Backward compat: old bool values
        if isinstance(ar, bool):
            ar = 'extend' if ar else 'off'
        ar = str(ar).lower()
        if ar == 'off':
            self._adaptive_snap_group.setChecked(False)
        else:
            self._adaptive_snap_group.setChecked(True)
            mode_idx = {"extend": 0, "snap": 1}.get(ar, 0)
            self._adaptive_mode_combo.setCurrentIndex(mode_idx)
        self._snap_dist.setValue(int(getattr(ns, 'snap_dist', 150)))
        self._snap_bbox_scale.setValue(
            float(getattr(ns, 'snap_bbox_scale', 0.5)))
        self._snap_w_dist.setValue(
            float(getattr(ns, 'snap_w_dist', 1.0)))
        self._snap_w_size.setValue(
            float(getattr(ns, 'snap_w_size', 0.3)))
        self._snap_w_intersect.setValue(
            float(getattr(ns, 'snap_w_intersect', 0.5)))
        self._snap_switch.setValue(
            int(getattr(ns, 'snap_switch_frames', 8)))
        self._forward_gaze_thresh.setValue(
            getattr(ns, 'forward_gaze_threshold', 5.0))
        self._hit_conf_gate.setValue(
            float(getattr(ns, 'hit_conf_gate', 0.0)))
        self._detect_extend.setValue(
            int(getattr(ns, 'detect_extend', 0)))
        scope = str(getattr(ns, 'detect_extend_scope', 'objects')).lower()
        scope_idx = {"objects": 0, "phenomena": 1, "both": 2}.get(scope, 0)
        self._detect_extend_scope.setCurrentIndex(scope_idx)

        # Fixation lock-on
        self._fixation_group.setChecked(
            bool(getattr(ns, 'gaze_lock', False)))
        self._dwell_frames.setValue(getattr(ns, 'dwell_frames', 15))
        self._lock_dist.setValue(getattr(ns, 'lock_dist', 100))
        self._reid_grace.setValue(
            getattr(ns, 'reid_grace_seconds', 1.0))
        self._cb_gaze_debug.setChecked(
            bool(getattr(ns, 'gaze_debug', False)))

        # Performance flags
        self._cb_fast.setChecked(bool(getattr(ns, 'fast', False)))
        self._skip_phenomena.setValue(int(getattr(ns, 'skip_phenomena', 0)))
        self._cb_lite_overlay.setChecked(
            bool(getattr(ns, 'lite_overlay', False)))
        self._cb_no_dashboard.setChecked(
            bool(getattr(ns, 'no_dashboard', False)))
        self._cb_profile.setChecked(bool(getattr(ns, 'profile', False)))

        # Gaze tips
        self._gaze_tips_group.setChecked(
            bool(getattr(ns, 'gaze_tips', False)))
        self._tip_radius.setValue(getattr(ns, 'tip_radius', 80))

        # Output
        self._cb_save.setChecked(bool(getattr(ns, 'save', False)))
        self._log_path.setText(str(getattr(ns, 'log', '') or ''))
        self._summary_path.setText(str(getattr(ns, 'summary', '') or ''))
        self._participant_ids.setText(
            str(getattr(ns, 'participant_ids', '') or ''))
        heatmap = getattr(ns, 'heatmap', None)
        if heatmap:
            self._cb_heatmap.setChecked(True)
            self._heatmap_path.setText(str(heatmap))
        else:
            self._cb_heatmap.setChecked(False)
            self._heatmap_path.setText('')

        self._cb_charts.setChecked(bool(getattr(ns, 'charts', False)))

        anon = getattr(ns, 'anonymize', None)
        self._cb_anonymize.setChecked(anon is not None)
        if anon:
            idx = self._anonymize_mode.findText(anon)
            if idx >= 0:
                self._anonymize_mode.setCurrentIndex(idx)

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
        # Reuse existing dashboard_q (don't recreate — panel holds the reference)
        from .workers import GazeWorker
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

    # ── Called externally to pre-fill a VP file (e.g. from VP Builder tab) ──

    def set_vp_file(self, path: str):
        """Set the VP file path and switch detection mode to YOLOE VP."""
        self._vp_file.setText(path)
        self._rb_det_vp.setChecked(True)
