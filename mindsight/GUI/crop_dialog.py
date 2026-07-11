"""
crop_dialog.py
--------------
The **Crop & Adjust Videos** tool (UP4 / HP4), launched from the Projects
tab overview: step through a project's videos, drag a crop rectangle on each
video's middle frame, optionally change its frame rate, then apply the whole
batch.

Non-destructive by default: each edited video is replaced IN PLACE (same
filename, so run.yaml and discovery never notice) and the untouched original
moves to an ``original/`` folder beside it -- invisible to run discovery,
which only reads top-level files.  A warned checkbox overwrites instead.
Edited videos hash differently, so the resume ledger re-runs them and
archives stale outputs on the next batch -- exactly the right behavior.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRubberBand,
    QSpinBox,
    QVBoxLayout,
)

from .widgets import _bgr_to_pixmap

_GO_GREEN = ("QPushButton{background:#2a7a2a;color:white;"
             "font-weight:bold;padding:4px 26px;}"
             "QPushButton:disabled{background:#33333f;color:#777;}")
_CANVAS_W, _CANVAS_H = 640, 360


class _CropCanvas(QLabel):
    """Middle-frame display with a drag-to-crop rubber band.

    Owns the display<->video coordinate mapping; emits ``rect_changed`` with
    an ``(x, y, w, h)`` tuple in VIDEO pixels (or None when cleared).
    """

    rect_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(_CANVAS_W, _CANVAS_H)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #1a1a2e; color: #556;")
        self._band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin = None
        self._orig_w = self._orig_h = 0
        self._have_frame = False

    def set_frame(self, frame_bgr, restore_rect=None):
        """Show *frame_bgr* (numpy BGR or None) and restore a prior crop."""
        self._band.hide()
        self._origin = None
        if frame_bgr is None:
            self._have_frame = False
            self._orig_w = self._orig_h = 0
            self.clear()
            self.setText("Preview unavailable for this video.")
            return
        self._orig_h, self._orig_w = frame_bgr.shape[:2]
        self._have_frame = True
        self.setPixmap(_bgr_to_pixmap(frame_bgr, _CANVAS_W, _CANVAS_H))
        if restore_rect:
            disp = self._video_to_display(restore_rect)
            if disp is not None:
                self._band.setGeometry(disp)
                self._band.show()

    # -- coordinate mapping ---------------------------------------------------

    def _display_box(self):
        """The pixmap's rectangle inside the (centered) label, or None."""
        pm = self.pixmap()
        if not self._have_frame or pm is None or pm.isNull():
            return None
        x = (self.width() - pm.width()) // 2
        y = (self.height() - pm.height()) // 2
        return QRect(x, y, pm.width(), pm.height())

    def _display_to_video(self, rect: QRect):
        box = self._display_box()
        if box is None:
            return None
        rect = rect.intersected(box)
        if rect.width() < 4 or rect.height() < 4:
            return None
        sx = self._orig_w / box.width()
        sy = self._orig_h / box.height()
        return (round((rect.x() - box.x()) * sx),
                round((rect.y() - box.y()) * sy),
                round(rect.width() * sx),
                round(rect.height() * sy))

    def _video_to_display(self, vrect):
        box = self._display_box()
        if box is None:
            return None
        x, y, w, h = vrect
        sx = box.width() / self._orig_w
        sy = box.height() / self._orig_h
        return QRect(box.x() + round(x * sx), box.y() + round(y * sy),
                     round(w * sx), round(h * sy))

    # -- mouse -> rubber band -------------------------------------------------

    def mousePressEvent(self, event):
        if not self._have_frame:
            return
        self._origin = event.position().toPoint()
        self._band.setGeometry(QRect(self._origin, self._origin))
        self._band.show()

    def mouseMoveEvent(self, event):
        if self._origin is not None:
            self._band.setGeometry(
                QRect(self._origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event):
        if self._origin is None:
            return
        rect = QRect(self._origin, event.position().toPoint()).normalized()
        self._origin = None
        vrect = self._display_to_video(rect)
        if vrect is None:
            self._band.hide()
        self.rect_changed.emit(vrect)

    def clear_crop(self):
        self._band.hide()
        self._origin = None
        self.rect_changed.emit(None)

    def show_rect(self, vrect):
        """Programmatically place the band (auto-crop pre-placement, LP1)."""
        disp = self._video_to_display(vrect) if vrect else None
        if disp is None:
            self._band.hide()
        else:
            self._band.setGeometry(disp)
            self._band.show()


class CropVideosDialog(QDialog):
    """Step through the project's videos; queue crops/fps edits; apply batch."""

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = Path(project)
        self.setWindowTitle("Crop & Adjust Videos")
        self._videos = self._discover()          # [(run_id, Path)]
        self._index = 0
        # run_id -> {"rect": tuple|None, "fps": float|None}
        self._edits: dict[str, dict] = {}
        self._frames: dict[str, object] = {}     # run_id -> BGR frame | None
        self._native_fps: dict[str, float] = {}
        self.applied = 0                         # for the caller / tests

        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel("Drag on the frame to crop. Changes apply to the "
                       "whole batch at the end.")
        title.setStyleSheet("color: #888;")
        head.addWidget(title)
        head.addStretch(1)
        self._progress_label = QLabel("")
        head.addWidget(self._progress_label)
        self._jump = QComboBox()
        self._jump.setMinimumWidth(140)
        self._jump.activated.connect(self._show_video)
        head.addWidget(self._jump)
        lay.addLayout(head)

        self._canvas = _CropCanvas()
        self._canvas.rect_changed.connect(self._rect_changed)
        canvas_row = QHBoxLayout()
        canvas_row.addStretch(1)
        canvas_row.addWidget(self._canvas)
        canvas_row.addStretch(1)
        lay.addLayout(canvas_row)

        info_row = QHBoxLayout()
        self._crop_label = QLabel("")
        info_row.addWidget(self._crop_label)
        clear_btn = QPushButton("Clear crop")
        clear_btn.clicked.connect(self._canvas.clear_crop)
        info_row.addWidget(clear_btn)
        info_row.addStretch(1)
        self._fps_check = QCheckBox("Change frame rate to:")
        self._fps_check.toggled.connect(self._fps_changed)
        info_row.addWidget(self._fps_check)
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 120.0)
        self._fps_spin.setDecimals(1)
        self._fps_spin.setValue(15.0)
        self._fps_spin.valueChanged.connect(self._fps_changed)
        info_row.addWidget(self._fps_spin)
        self._native_label = QLabel("")
        self._native_label.setStyleSheet("color: #888;")
        info_row.addWidget(self._native_label)
        lay.addLayout(info_row)

        # Auto-crop (LP1): pre-place the rectangle from detections; the user
        # reviews/adjusts in the same rubber band before anything re-encodes.
        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel("Auto-crop:"))
        self._auto_mode = QComboBox()
        self._auto_mode.addItems(["Objects by name", "Visual prompt file"])
        self._auto_mode.setToolTip(
            "Find the crop by naming objects (YOLOE text prompt) or by "
            "your study's visual prompt file")
        self._auto_mode.currentIndexChanged.connect(self._auto_mode_changed)
        auto_row.addWidget(self._auto_mode)
        self._auto_classes = QLineEdit("person, dining table")
        self._auto_classes.setToolTip(
            "Comma-separated object names the crop should contain")
        auto_row.addWidget(self._auto_classes, 1)
        self._auto_vp_btn = QPushButton("Choose VP...")
        self._auto_vp_btn.clicked.connect(self._choose_auto_vp)
        self._auto_vp_btn.setVisible(False)
        auto_row.addWidget(self._auto_vp_btn, 1)
        auto_row.addWidget(QLabel("Padding:"))
        self._auto_pad = QSpinBox()
        self._auto_pad.setRange(0, 1000)
        self._auto_pad.setValue(100)
        self._auto_pad.setSuffix(" px")
        auto_row.addWidget(self._auto_pad)
        auto_this = QPushButton("This video")
        auto_this.clicked.connect(self._auto_crop_current)
        auto_row.addWidget(auto_this)
        auto_all = QPushButton("All videos")
        auto_all.setToolTip(
            "Place a crop on every video, then step through to review "
            "before applying")
        auto_all.clicked.connect(self._auto_crop_all)
        auto_row.addWidget(auto_all)
        lay.addLayout(auto_row)
        self._auto_vp_file = None
        self._detector = None
        self._detector_key = None

        self._overwrite = QCheckBox(
            "Overwrite original files (no backup kept)")
        self._overwrite.setStyleSheet("color: #b8860b;")
        self._overwrite.setToolTip(
            "Unchecked (recommended): the untouched original is kept in an "
            "'original' folder beside the video.")
        lay.addWidget(self._overwrite)

        nav = QHBoxLayout()
        self._file_label = QLabel("")
        self._file_label.setStyleSheet("color: #888;")
        nav.addWidget(self._file_label)
        nav.addStretch(1)
        prev_b = QPushButton("‹ Prev video")
        prev_b.clicked.connect(lambda: self._show_video(self._index - 1))
        nav.addWidget(prev_b)
        next_b = QPushButton("Next video ›")
        next_b.clicked.connect(lambda: self._show_video(self._index + 1))
        nav.addWidget(next_b)
        self._apply_btn = QPushButton("Apply changes")
        self._apply_btn.setStyleSheet(_GO_GREEN)
        self._apply_btn.setMinimumHeight(32)
        self._apply_btn.clicked.connect(self._apply)
        nav.addWidget(self._apply_btn)
        lay.addLayout(nav)

        self._show_video(0)
        self._update_apply_btn()

    # -- data -----------------------------------------------------------------

    def _discover(self):
        from mindsight.project.project import Project
        try:
            proj = Project.open(str(self._project))
            return [(spec.run_id, Path(spec.source)) for spec in proj.runs()]
        except Exception:  # noqa: BLE001 -- unreadable project -> empty list
            return []

    def _load_frame(self, run_id: str, source: Path):
        if run_id in self._frames:
            return self._frames[run_id]
        import cv2
        frame = None
        cap = cv2.VideoCapture(str(source))
        if cap.isOpened():
            self._native_fps[run_id] = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                frame = frame if ok else None
        cap.release()
        self._frames[run_id] = frame
        return frame

    def _edit_for(self, run_id: str) -> dict:
        return self._edits.setdefault(run_id, {"rect": None, "fps": None})

    def pending(self) -> dict:
        """run_id -> edit dict, for edits that actually do something."""
        return {rid: e for rid, e in self._edits.items()
                if e["rect"] is not None or e["fps"] is not None}

    # -- per-video UI ----------------------------------------------------------

    def _show_video(self, index: int):
        if not self._videos:
            self._progress_label.setText("No videos in this project")
            self._canvas.setText("Add videos to the project first.")
            return
        self._index = max(0, min(index, len(self._videos) - 1))
        run_id, source = self._videos[self._index]
        self._progress_label.setText(
            f"Video {self._index + 1} of {len(self._videos)}")
        self._jump.blockSignals(True)
        self._jump.clear()
        self._jump.addItems([r for r, _ in self._videos])
        self._jump.setCurrentIndex(self._index)
        self._jump.blockSignals(False)
        self._file_label.setText(source.name)

        edit = self._edit_for(run_id)
        frame = self._load_frame(run_id, source)
        self._canvas.set_frame(frame, restore_rect=edit["rect"])
        native = self._native_fps.get(run_id, 0.0)
        self._native_label.setText(
            f"(currently {native:.1f} fps)" if native else "")
        self._fps_check.blockSignals(True)
        self._fps_spin.blockSignals(True)
        self._fps_check.setChecked(edit["fps"] is not None)
        if edit["fps"] is not None:
            self._fps_spin.setValue(edit["fps"])
        self._fps_spin.blockSignals(False)
        self._fps_check.blockSignals(False)
        self._refresh_crop_label()

    def _rect_changed(self, vrect):
        if not self._videos:
            return
        run_id, _ = self._videos[self._index]
        self._edit_for(run_id)["rect"] = vrect
        self._refresh_crop_label()
        self._update_apply_btn()

    def _fps_changed(self, *_):
        if not self._videos:
            return
        run_id, _ = self._videos[self._index]
        self._edit_for(run_id)["fps"] = (
            self._fps_spin.value() if self._fps_check.isChecked() else None)
        self._update_apply_btn()

    def _refresh_crop_label(self):
        if not self._videos:
            return
        run_id, _ = self._videos[self._index]
        rect = self._edit_for(run_id)["rect"]
        if rect:
            x, y, w, h = rect
            self._crop_label.setText(f"Crop: {w}×{h} at ({x}, {y})")
        else:
            self._crop_label.setText("No crop -- drag on the frame")

    def _update_apply_btn(self):
        n = len(self.pending())
        self._apply_btn.setText(f"Apply changes ({n})" if n
                                else "Apply changes")
        self._apply_btn.setEnabled(bool(n))

    # -- auto-crop (LP1) --------------------------------------------------------

    def _auto_mode_changed(self, idx: int):
        text_mode = idx == 0
        self._auto_classes.setVisible(text_mode)
        self._auto_vp_btn.setVisible(not text_mode)

    def _choose_auto_vp(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a visual prompt file", "",
            "Visual Prompt (*.vp.json);;All (*)")
        if path:
            self._auto_vp_file = path
            self._auto_vp_btn.setText(Path(path).name)

    def _ensure_detector(self):
        """Lazy-load (and cache) the landmark detector for the current mode."""
        from .auto_crop import load_landmark_detector
        if self._auto_mode.currentIndex() == 1:
            if not self._auto_vp_file:
                QMessageBox.warning(self, "Auto-crop",
                                    "Choose a visual prompt file first.")
                return None
            key = ("vp", self._auto_vp_file)
        else:
            classes = [c.strip() for c in self._auto_classes.text().split(",")
                       if c.strip()]
            if not classes:
                QMessageBox.warning(
                    self, "Auto-crop",
                    "Name at least one object (e.g. person, dining table).")
                return None
            key = ("text", tuple(classes))
        if self._detector_key == key and self._detector is not None:
            return self._detector
        self._crop_label.setText(
            "Loading detector (first use takes a moment)...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        try:
            if key[0] == "vp":
                self._detector = load_landmark_detector(
                    "vp", vp_file=self._auto_vp_file)
            else:
                self._detector = load_landmark_detector(
                    "text", classes=list(key[1]))
        except Exception as exc:  # noqa: BLE001 -- plain-English, not a crash
            self._detector = None
            self._refresh_crop_label()
            QMessageBox.critical(
                self, "Auto-crop",
                f"Could not load the detector:\n{exc}\n\nCheck that the "
                "YOLOE weights (yoloe-26l-seg.pt) are installed in the "
                "Models tab.")
            return None
        self._detector_key = key
        self._refresh_crop_label()
        return self._detector

    def _auto_rect_for(self, run_id: str, source: Path):
        from .auto_crop import detect_boxes, union_rect
        frame = self._load_frame(run_id, source)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        boxes = detect_boxes(self._detector, frame)
        return union_rect(boxes, self._auto_pad.value(), w, h)

    def _auto_crop_current(self):
        if not self._videos or self._ensure_detector() is None:
            return
        run_id, source = self._videos[self._index]
        rect = self._auto_rect_for(run_id, source)
        if rect is None:
            QMessageBox.information(
                self, "Auto-crop",
                "Nothing matching was found in this video's frame -- adjust "
                "the object names or padding, or crop by hand.")
            return
        self._edit_for(run_id)["rect"] = rect
        self._canvas.show_rect(rect)
        self._refresh_crop_label()
        self._update_apply_btn()

    def _auto_crop_all(self):
        """LP1's batch flow: soft-crop every video, review, then Apply."""
        if not self._videos or self._ensure_detector() is None:
            return
        progress = QProgressDialog("Auto-cropping...", None, 0,
                                   len(self._videos), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        found = 0
        for i, (run_id, source) in enumerate(self._videos):
            progress.setValue(i)
            progress.setLabelText(
                f"Detecting in {source.name} ({i + 1}/{len(self._videos)})")
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            rect = self._auto_rect_for(run_id, source)
            if rect is not None:
                self._edit_for(run_id)["rect"] = rect
                found += 1
        progress.setValue(len(self._videos))
        self._show_video(self._index)      # re-render the current band
        self._update_apply_btn()
        QMessageBox.information(
            self, "Auto-crop",
            f"Placed crop rectangles on {found} of {len(self._videos)} "
            "video(s). Step through to review and adjust, then Apply.")

    # -- apply -----------------------------------------------------------------

    def _apply(self):
        self._fps_changed()      # capture the current video's fps state
        pending = self.pending()
        if not pending:
            return
        overwrite = self._overwrite.isChecked()
        mode = ("overwriting the originals (no backup)" if overwrite
                else "originals kept in an 'original' folder")
        reply = QMessageBox.question(
            self, "Apply changes",
            f"Re-encode {len(pending)} video(s) -- {mode}.\n\n"
            "Videos that already have results will re-run (old outputs are "
            "archived) the next time the project runs. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        from mindsight.io.video_edit import apply_edit
        by_id = dict(self._videos)
        progress = QProgressDialog("Re-encoding...", None, 0, len(pending),
                                   self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        done, failed = [], []
        for i, (run_id, edit) in enumerate(pending.items()):
            progress.setValue(i)
            progress.setLabelText(
                f"Re-encoding {by_id[run_id].name} ({i + 1}/{len(pending)})")
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            try:
                apply_edit(by_id[run_id], rect=edit["rect"], fps=edit["fps"],
                           overwrite=overwrite)
                done.append(run_id)
            except (ValueError, RuntimeError, OSError) as exc:
                failed.append(f"{run_id}: {exc}")
        progress.setValue(len(pending))

        for run_id in done:                       # edits consumed
            self._edits.pop(run_id, None)
            self._frames.pop(run_id, None)        # re-decode the new file
        self.applied += len(done)
        self._show_video(self._index)
        self._update_apply_btn()
        if failed:
            QMessageBox.warning(
                self, "Some videos failed",
                f"Re-encoded {len(done)} of {len(pending)}.\nProblems:\n"
                + "\n".join(failed))
        else:
            QMessageBox.information(
                self, "Done", f"Re-encoded {len(done)} video(s).")
