"""Gaze backend selection and configuration (MGaze, L2CS, UniGaze, Gazelle)."""

from __future__ import annotations

from argparse import Namespace

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QRadioButton,
    QSpinBox,
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

from ..widgets import _browse_btn, _hrow

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


class GazeBackendSection(QWidget):
    """Backend radio buttons and per-backend configuration widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._build_ui(lay)

    def _build_ui(self, lay):
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
        self._gaze_arch.addItems(GAZE_ARCHS)
        self._gaze_arch.setCurrentText("mobileone_s0")
        self._gaze_dataset = QComboBox()
        self._gaze_dataset.addItems(GAZE_DATASETS)
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
        self._l2cs_dataset.addItems(GAZE_DATASETS)
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
        self._gazelle_name.addItems(GAZELLE_NAMES)
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
            "Use float16 inference on CUDA/MPS for ~1.5-2x speedup. "
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

        # Enforce mutual exclusion
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

    # -- Helpers --------------------------------------------------------------

    def _refresh_backend(self):
        is_mgaze = self._rb_mgaze.isChecked()
        self._mgaze_widget.setVisible(is_mgaze)
        mgaze_path = self._gaze_model.text().strip().lower()
        self._arch_widget.setVisible(is_mgaze and mgaze_path.endswith('.pt'))
        self._l2cs_widget.setVisible(self._rb_l2cs.isChecked())
        self._unigaze_widget.setVisible(self._rb_unigaze.isChecked())
        self._gazelle_widget.setVisible(self._rb_gazelle.isChecked())

    def _browse_to(self, line_edit: QLineEdit, filt: str = "*"):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", f"Files ({filt});;All (*)")
        if path:
            line_edit.setText(path)

    def selected_backend(self) -> str:
        if self._rb_gazelle.isChecked():
            return "gazelle"
        if self._rb_l2cs.isChecked():
            return "l2cs"
        if self._rb_unigaze.isChecked():
            return "unigaze"
        return "mgaze"

    # -- Namespace interface --------------------------------------------------

    def namespace_values(self) -> dict:
        backend = self.selected_backend()
        mgaze_model = ""
        if backend == "mgaze":
            mgaze_model = self._gaze_model.text().strip()
        return dict(
            mgaze_model=mgaze_model,
            mgaze_arch=(self._gaze_arch.currentText()
                        if backend == "mgaze"
                        and mgaze_model.lower().endswith('.pt')
                        else None),
            mgaze_dataset=self._gaze_dataset.currentText(),
            l2cs_model=(self._l2cs_model.text().strip()
                        if backend == "l2cs" else None),
            l2cs_arch=self._l2cs_arch.currentText(),
            l2cs_dataset=self._l2cs_dataset.currentText(),
            unigaze_model=(self._unigaze_variant.currentText()
                           if backend == "unigaze" else None),
            gazelle_model=(self._gazelle_ckpt.text().strip()
                           if backend == "gazelle" else None),
            gazelle_name=self._gazelle_name.currentText(),
            gazelle_inout_threshold=self._gazelle_inout.value(),
            gazelle_device=self._gazelle_device.currentText(),
            gazelle_skip_frames=self._gazelle_skip.value(),
            gazelle_fp16=self._gazelle_fp16.isChecked(),
            gazelle_compile=self._gazelle_compile.isChecked(),
        )

    def apply_namespace(self, ns: Namespace):
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

        gaze_model = getattr(ns, 'mgaze_model', '')
        if gaze_model:
            self._gaze_model.setText(str(gaze_model))
        self._gaze_dataset.setCurrentText(
            str(getattr(ns, 'mgaze_dataset', 'gaze360')))

        if l2cs_model:
            self._l2cs_model.setText(str(l2cs_model))
        l2cs_arch = getattr(ns, 'l2cs_arch', 'ResNet50')
        if l2cs_arch:
            self._l2cs_arch.setCurrentText(str(l2cs_arch))
        self._l2cs_dataset.setCurrentText(
            str(getattr(ns, 'l2cs_dataset', 'gaze360')))
        self._gazelle_name.setCurrentText(
            str(getattr(ns, 'gazelle_name', GAZELLE_NAMES[0])))
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
