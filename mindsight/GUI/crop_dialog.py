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
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRubberBand,
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
