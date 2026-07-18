"""Gaze backend selection and configuration (MGaze, Gazelle)."""

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
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mindsight.GazeTracking.Backends.MGaze.MGaze_Config import DEFAULT_ONNX_MODEL

from ..widgets import _browse_btn, _hrow

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

        # Row 1: MobileGaze, Gaze-LLE (paper terminology, D13; dests unchanged)
        rb_row1 = _hrow()
        self._rb_mgaze = QRadioButton("MobileGaze")
        self._rb_mgaze.setChecked(True)
        rb_row1.layout().addWidget(self._rb_mgaze)
        self._rb_gazelle = QRadioButton("Gaze-LLE")
        rb_row1.layout().addWidget(self._rb_gazelle)
        rb_row1.layout().addStretch(1)
        vl.addWidget(rb_row1)

        # -- MGaze model path + inference mode --
        self._mgaze_widget = QWidget()
        mgl = QVBoxLayout(self._mgaze_widget)
        mgl.setContentsMargins(0, 0, 0, 0)
        from mindsight.GUI.path_picker import KnownPathCombo, known_candidates
        self._gaze_model = KnownPathCombo(known_candidates("mgaze_model"))
        self._gaze_model.setText(DEFAULT_ONNX_MODEL)
        gm_btn = _browse_btn()
        gm_btn.clicked.connect(
            lambda: self._browse_to(self._gaze_model, "*.onnx *.pt",
                                    dest="mgaze_model"))
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

        # -- Gazelle widget --
        self._gazelle_widget = QWidget()
        gwl = QFormLayout(self._gazelle_widget)
        gwl.setContentsMargins(0, 0, 0, 0)
        self._gazelle_name = QComboBox()
        self._gazelle_name.addItems(GAZELLE_NAMES)
        self._gazelle_ckpt = KnownPathCombo(known_candidates("gazelle_model"))
        gc_btn = _browse_btn()
        gc_btn.clicked.connect(
            lambda: self._browse_to(self._gazelle_ckpt, "*.pt",
                                    dest="gazelle_model"))
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
        self._backend_group.addButton(self._rb_gazelle)

        self._rb_mgaze.toggled.connect(self._refresh_backend)
        self._rb_gazelle.toggled.connect(self._refresh_backend)
        lay.addWidget(g)

    # -- Helpers --------------------------------------------------------------

    def _refresh_backend(self):
        is_mgaze = self._rb_mgaze.isChecked()
        self._mgaze_widget.setVisible(is_mgaze)
        mgaze_path = self._gaze_model.text().strip().lower()
        self._arch_widget.setVisible(is_mgaze and mgaze_path.endswith('.pt'))
        self._gazelle_widget.setVisible(self._rb_gazelle.isChecked())

    def _browse_to(self, line_edit, filt: str = "*", dest: str = ""):
        from mindsight.GUI.path_picker import default_browse_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", default_browse_dir(dest),
            f"Files ({filt});;All (*)")
        if path:
            line_edit.setText(path)

    def selected_backend(self) -> str:
        if self._rb_gazelle.isChecked():
            return "gazelle"
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
        mgaze_arch = getattr(ns, 'mgaze_arch', None)
        if gazelle_model:
            self._rb_gazelle.setChecked(True)
            self._gazelle_ckpt.setText(str(gazelle_model))
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
