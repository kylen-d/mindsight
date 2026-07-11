"""
record_session_dialog.py -- set up a live study session (UP5).

Collects the camera, the target run (a NEW session id or an existing PLANNED
session), and the session metadata (participants by on-screen position,
conditions, date/session/notes).  The tab owns the actual recording state;
this dialog only gathers the plan.
"""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

_GO_GREEN = ("QPushButton{background:#2a7a2a;color:white;"
             "font-weight:bold;padding:4px 26px;}")


class RecordSessionDialog(QDialog):
    """Camera + target run + metadata for a live session.

    Results after ``exec()``: ``camera_index`` (int), ``run_id`` (str),
    ``meta`` (Q2 dict), ``use_planned`` (bool).
    """

    def __init__(self, planned=None, parent=None, preselect: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Record Live Session")
        self._planned = list(planned or [])   # RunFolderInfo list
        self.camera_index = 0
        self.run_id = ""
        self.meta: dict = {}
        self.use_planned = False

        lay = QVBoxLayout(self)
        form = QFormLayout()

        cam_row = QHBoxLayout()
        self._camera = QComboBox()
        for i in range(4):
            self._camera.addItem(f"Camera {i}", i)
        cam_row.addWidget(self._camera, 1)
        refresh = QPushButton("Refresh")
        refresh.setToolTip("Detect connected cameras (may trigger a one-time "
                           "camera permission prompt)")
        refresh.clicked.connect(self._refresh_cameras)
        cam_row.addWidget(refresh)
        form.addRow("Camera:", self._wrap(cam_row))

        # Target: an existing planned session, or a new one.
        self._use_planned_radio = QRadioButton("A planned session:")
        self._planned_combo = QComboBox()
        for info in self._planned:
            self._planned_combo.addItem(info.run_id)
        self._new_radio = QRadioButton("A new session named:")
        self._run_id_edit = QLineEdit(
            datetime.now().strftime("session_%Y%m%d-%H%M"))
        if self._planned:
            self._use_planned_radio.setChecked(True)
            if preselect and self._planned_combo.findText(preselect) >= 0:
                self._planned_combo.setCurrentText(preselect)
        else:
            self._new_radio.setChecked(True)
            self._use_planned_radio.setEnabled(False)
            self._planned_combo.setEnabled(False)
            self._planned_combo.addItem("(none planned)")
        target1 = QHBoxLayout()
        target1.addWidget(self._use_planned_radio)
        target1.addWidget(self._planned_combo, 1)
        target2 = QHBoxLayout()
        target2.addWidget(self._new_radio)
        target2.addWidget(self._run_id_edit, 1)
        form.addRow("Record into:", self._wrap(target1))
        form.addRow("", self._wrap(target2))

        self._participants = QLineEdit()
        self._participants.setPlaceholderText(
            "left to right on screen, comma-separated -- e.g. S80, S81")
        form.addRow("Participants:", self._participants)
        self._conditions = QLineEdit()
        self._conditions.setPlaceholderText(
            "optional -- separate multiple with |")
        form.addRow("Conditions:", self._conditions)
        self._date = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        form.addRow("Date:", self._date)
        self._session = QLineEdit()
        form.addRow("Session:", self._session)
        self._notes = QLineEdit()
        form.addRow("Notes:", self._notes)
        lay.addLayout(form)

        hint = QLabel(
            "The raw camera feed records into the project as this run's "
            "video (re-analyzable later); analysis starts automatically when "
            "you end the session.")
        hint.setStyleSheet("color: #888;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        btns = QDialogButtonBox()
        start = btns.addButton("⏺  Start Recording",
                               QDialogButtonBox.ButtonRole.AcceptRole)
        start.setStyleSheet(_GO_GREEN)
        btns.addButton(QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self._use_planned_radio.toggled.connect(self._prefill_from_planned)
        self._planned_combo.currentIndexChanged.connect(
            self._prefill_from_planned)
        self._prefill_from_planned()

    @staticmethod
    def _wrap(layout):
        from PyQt6.QtWidgets import QWidget
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _refresh_cameras(self):
        # camera_enum matches cv2's device ordering; the combo stores each
        # device's cv2 index as item data (eyes-on A3).
        from .camera_enum import list_cameras
        current = self._camera.currentData()
        self._camera.clear()
        for idx, name in list_cameras():
            self._camera.addItem(name, idx)
        pos = self._camera.findData(current)
        self._camera.setCurrentIndex(max(pos, 0))

    def _prefill_from_planned(self, *_):
        if not (self._planned and self._use_planned_radio.isChecked()):
            return
        idx = self._planned_combo.currentIndex()
        if not (0 <= idx < len(self._planned)):
            return
        info = self._planned[idx]
        if info.meta.pid_map:
            self._participants.setText(", ".join(
                v for _, v in sorted(info.meta.pid_map.items())))
        if info.meta.conditions:
            self._conditions.setText("|".join(info.meta.conditions))
        mm = info.meta.manifest_meta
        if mm.get("date"):
            self._date.setText(str(mm["date"]))
        if mm.get("session"):
            self._session.setText(str(mm["session"]))
        if mm.get("notes"):
            self._notes.setText(str(mm["notes"]))

    def _build_meta(self) -> dict:
        meta: dict = {}
        labels = [p.strip() for p in self._participants.text().split(",")
                  if p.strip()]
        if labels:
            meta["participants"] = {i: lab for i, lab in enumerate(labels)}
        tags = [t.strip() for t in self._conditions.text().split("|")
                if t.strip()]
        if tags:
            meta["conditions"] = tags
        for key, edit in (("date", self._date), ("session", self._session),
                          ("notes", self._notes)):
            if edit.text().strip():
                meta[key] = edit.text().strip()
        return meta

    def _accept(self):
        from PyQt6.QtWidgets import QMessageBox
        self.use_planned = bool(self._planned
                                and self._use_planned_radio.isChecked())
        if self.use_planned:
            self.run_id = self._planned_combo.currentText()
        else:
            self.run_id = self._run_id_edit.text().strip()
        if not self.run_id:
            QMessageBox.warning(self, "Record Live Session",
                                "Name the session.")
            return
        data = self._camera.currentData()
        self.camera_index = (data if isinstance(data, int)
                             else max(self._camera.currentIndex(), 0))
        self.meta = self._build_meta()
        self.accept()
