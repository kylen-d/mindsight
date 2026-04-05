"""Source selection and object detection settings."""

from __future__ import annotations

from argparse import Namespace

from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets import VP_EXT, _browse_btn, _hrow


class DetectionSection(QWidget):
    """Source path + YOLO / YOLOE detection settings + device selector."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._build_source(lay)
        self._build_detection(lay)

    # -- Source ---------------------------------------------------------------

    def _build_source(self, lay):
        g = QGroupBox("Source")
        f = QFormLayout(g)
        self._src = QLineEdit("0")
        btn = _browse_btn()
        btn.clicked.connect(self._browse_source)
        f.addRow("Source:", _hrow(self._src, btn))
        lay.addWidget(g)

    # -- Object Detection -----------------------------------------------------

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

        # Global device selector
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

    # -- Helpers --------------------------------------------------------------

    def _refresh_det_mode(self):
        vp = self._rb_det_vp.isChecked()
        self._yolo_det_panel.setVisible(not vp)
        self._vp_det_panel.setVisible(vp)

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

    def set_vp_file(self, path: str):
        """Set the VP file path and switch detection mode to YOLOE VP."""
        self._vp_file.setText(path)
        self._rb_det_vp.setChecked(True)

    # -- Namespace interface --------------------------------------------------

    def namespace_values(self) -> dict:
        use_vp = self._rb_det_vp.isChecked()
        cls_raw = self._classes.text().strip()
        bl_raw = self._blacklist.text().strip()
        return dict(
            source=self._src.text().strip(),
            device=self._device_combo.currentText(),
            model=self._yolo_model.text().strip(),
            conf=(self._vp_conf_spin.value() if use_vp
                  else self._conf_spin.value()),
            classes=([c.strip() for c in cls_raw.split(",") if c.strip()]
                     or []),
            blacklist=[c.strip() for c in bl_raw.split(",") if c.strip()],
            vp_file=self._vp_file.text().strip() if use_vp else None,
            vp_model=self._yoloe_model.text().strip(),
        )

    def apply_namespace(self, ns: Namespace):
        self._src.setText(str(getattr(ns, 'source', '0')))
        self._device_combo.setCurrentText(
            str(getattr(ns, 'device', 'auto')))
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
