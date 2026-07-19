"""
GUI/validation_annotator.py — Ground-truth annotator dialog (validation suite).

Opened per validation set from the Inference Tuning tab's validation
workbench.  Reuses the VP Builder's ImageCanvas: three explicit tools
because the canvas's suggest mode and drag-drawing are mutually
exclusive —

* **Target**: a click places the current participant's gaze target
  (canvas suggest mode with no proposals, so clicks arrive as
  ``point_clicked``); Offscreen / Uncertain / Skip buttons record the
  eval-harness label states instead of a point.
* **Draw box**: drag draws a labeled object box (for the IoU metric).
* **Suggest box**: FastSAM point-prompt proposals, exactly the VP
  Builder flow; an accepted proposal becomes an object box.

Every mutation autosaves through the ValidationStore (atomic writes) —
a long labeling session never loses work.  Frames decode on demand from
the set's source video; a ``frame_provider`` seam keeps tests
video-free.
"""
from __future__ import annotations

import queue
import threading

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from mindsight.GUI.widgets import ImageCanvas

_TARGET_COLOURS_BGR = [(80, 200, 255), (100, 255, 130), (255, 150, 50),
                       (200, 149, 255)]


class VideoFrameProvider:
    """Default frame provider: decodes frames from the set's video."""

    def __init__(self, video_path: str):
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(video_path))
        self.count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self, frame_no: int):
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        self._cap.release()


class ValidationAnnotatorDialog(QDialog):
    """Frame strip left · canvas center · participants/objects right."""

    def __init__(self, vset, store, parent=None, frame_provider=None,
                 suggester_factory=None):
        super().__init__(parent)
        self.setWindowTitle(f"Annotate validation set — {vset.name}")
        self.resize(1100, 640)
        self._vset = vset
        self._store = store
        self._provider = frame_provider or VideoFrameProvider(vset.video)
        self._suggester_factory = suggester_factory
        self._suggester = None
        self._suggest_q: queue.Queue = queue.Queue()
        self._suggest_busy = False
        self._suggest_timer = QTimer(self)
        self._suggest_timer.timeout.connect(self._poll_suggest)
        self._current_frame_no: int | None = None
        self._current_frame = None

        root = QHBoxLayout(self)

        # ── Left: frame strip ────────────────────────────────────────────
        left = QVBoxLayout()
        left.addWidget(QLabel("Frames"))
        self._frame_list = QListWidget()
        self._frame_list.currentRowChanged.connect(self._on_frame_row)
        left.addWidget(self._frame_list, 1)

        sample_row = QHBoxLayout()
        self._every_spin = QSpinBox()
        self._every_spin.setRange(1, 10000)
        self._every_spin.setValue(int(self._vset.every or 30))
        self._every_spin.setPrefix("every ")
        sample_row.addWidget(self._every_spin)
        sample_btn = QPushButton("Sample")
        sample_btn.setToolTip("Add every Nth frame of the video to the set.")
        sample_btn.clicked.connect(self._on_sample)
        sample_row.addWidget(sample_btn)
        left.addLayout(sample_row)

        add_row = QHBoxLayout()
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 10_000_000)
        add_row.addWidget(self._frame_spin)
        add_btn = QPushButton("Add frame")
        add_btn.clicked.connect(self._on_add_frame)
        add_row.addWidget(add_btn)
        left.addLayout(add_row)

        rm_btn = QPushButton("Remove frame")
        rm_btn.clicked.connect(self._on_remove_frame)
        left.addWidget(rm_btn)
        root.addLayout(left, 0)

        # ── Center: canvas + tools ───────────────────────────────────────
        center = QVBoxLayout()
        self._canvas = ImageCanvas()
        self._canvas.point_clicked.connect(self._on_point)
        self._canvas.crop_drawn.connect(self._on_box_drawn)
        self._canvas.suggestion_accepted.connect(self._on_suggestion_accepted)
        center.addWidget(self._canvas, 1)

        tools = QHBoxLayout()
        self._tool_target = QRadioButton("Target")
        self._tool_target.setToolTip(
            "Click where the selected participant is looking.")
        self._tool_draw = QRadioButton("Draw box")
        self._tool_draw.setToolTip("Drag to draw a labeled object box.")
        self._tool_suggest = QRadioButton("Suggest box")
        self._tool_suggest.setToolTip(
            "Click an object to get FastSAM box proposals (needs the "
            "FastSAM-s weight from the Models tab).")
        self._tool_target.setChecked(True)
        for rb in (self._tool_target, self._tool_draw, self._tool_suggest):
            rb.toggled.connect(self._on_tool_changed)
            tools.addWidget(rb)
        tools.addStretch(1)
        self._status = QLabel("")
        tools.addWidget(self._status)
        center.addLayout(tools)

        nav = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(lambda: self._step(-1))
        nav.addWidget(prev_btn)
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(lambda: self._step(+1))
        nav.addWidget(next_btn)
        nav.addStretch(1)
        center.addLayout(nav)
        root.addLayout(center, 1)

        # ── Right: participant + objects ─────────────────────────────────
        right = QVBoxLayout()
        right.addWidget(QLabel("Participant"))
        self._pid_combo = QComboBox()
        self._pid_combo.setEditable(True)
        self._pid_combo.addItems(["P0", "P1", "P2", "P3"])
        right.addWidget(self._pid_combo)

        for text, state in (("Off-screen", "offscreen"),
                            ("Uncertain", "uncertain"),
                            ("Skip", "skip")):
            btn = QPushButton(text)
            btn.clicked.connect(
                lambda _=False, s=state: self._set_state_label(s))
            right.addWidget(btn)
        clear_btn = QPushButton("Clear label")
        clear_btn.clicked.connect(self._clear_label)
        right.addWidget(clear_btn)

        right.addWidget(QLabel("Labels on this frame"))
        self._label_list = QListWidget()
        right.addWidget(self._label_list, 1)

        right.addWidget(QLabel("Object name"))
        self._obj_name = QLineEdit("object")
        right.addWidget(self._obj_name)
        right.addWidget(QLabel("Objects on this frame"))
        self._obj_list = QListWidget()
        right.addWidget(self._obj_list, 1)
        rm_obj = QPushButton("Remove object")
        rm_obj.clicked.connect(self._on_remove_object)
        right.addWidget(rm_obj)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        right.addWidget(buttons)
        root.addLayout(right, 0)

        self._reload_frame_list()
        if self._frame_list.count():
            self._frame_list.setCurrentRow(0)
        self._on_tool_changed()

    # ── Frame strip ──────────────────────────────────────────────────────────

    def _frame_numbers(self) -> list[int]:
        return self._vset.frames()

    def _reload_frame_list(self, keep_frame=None):
        self._frame_list.blockSignals(True)
        self._frame_list.clear()
        row_for = 0
        for i, fno in enumerate(self._frame_numbers()):
            pts = sum(1 for v in self._vset.labels.get(fno, {}).values())
            objs = len(self._vset.objects.get(fno, []))
            self._frame_list.addItem(f"{fno}   ({pts} label, {objs} obj)")
            if keep_frame is not None and fno == keep_frame:
                row_for = i
        self._frame_list.blockSignals(False)
        if self._frame_list.count():
            self._frame_list.setCurrentRow(row_for)
        else:
            self._current_frame_no = None
            self._current_frame = None
            self._render()

    def _on_frame_row(self, row: int):
        frames = self._frame_numbers()
        if not (0 <= row < len(frames)):
            return
        self._current_frame_no = frames[row]
        self._current_frame = self._provider.read(self._current_frame_no)
        self._canvas.set_suggestions([])
        if self._current_frame is None:
            self._status.setText(
                f"Frame {self._current_frame_no}: cannot decode")
        else:
            self._status.setText("")
        self._render()

    def _on_sample(self):
        step = self._every_spin.value()
        total = getattr(self._provider, "count", 0)
        if total <= 0:
            QMessageBox.warning(self, "No video",
                                "The set's video has no readable frames.")
            return
        for fno in range(0, total, step):
            self._vset.add_frame(fno)
        self._vset.every = step
        self._save()
        self._reload_frame_list(keep_frame=self._current_frame_no)

    def _on_add_frame(self):
        self._vset.add_frame(self._frame_spin.value())
        self._save()
        self._reload_frame_list(keep_frame=self._frame_spin.value())

    def _on_remove_frame(self):
        if self._current_frame_no is None:
            return
        self._vset.remove_frame(self._current_frame_no)
        self._save()
        self._reload_frame_list()

    def _step(self, delta: int):
        row = self._frame_list.currentRow() + delta
        if 0 <= row < self._frame_list.count():
            self._frame_list.setCurrentRow(row)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_N, Qt.Key.Key_Space):
            self._step(+1)
        elif event.key() == Qt.Key.Key_B:
            self._step(-1)
        else:
            super().keyPressEvent(event)

    # ── Tools ────────────────────────────────────────────────────────────────

    def _on_tool_changed(self, *_):
        # Target and Suggest both need the canvas's suggest mode (clicks
        # arrive as point_clicked); Draw box needs it OFF so drags work.
        if self._tool_suggest.isChecked():
            from mindsight.GUI.region_suggest import fastsam_path
            if self._suggester_factory is None and fastsam_path() is None:
                QMessageBox.information(
                    self, "Weight needed",
                    "Suggest mode needs the FastSAM-s segmentation weight "
                    "(24 MB).\n\nDownload it on the Models tab (SAM row).")
                self._tool_target.setChecked(True)
                return
        self._canvas.set_suggest_mode(not self._tool_draw.isChecked())

    def _pid(self) -> str:
        """Participant key for the store.  Standard P<N> labels store as
        digit keys — the eval-harness labels convention, matching the
        gaze stream's face_idx — so scripts/eval_gaze.py scores the set
        file directly.  Custom labels (e.g. S70) store as typed and are
        matched against participant_label at scoring time."""
        text = self._pid_combo.currentText().strip() or "P0"
        if len(text) >= 2 and text[0] in "Pp" and text[1:].isdigit():
            return text[1:]
        return text

    # ── Labeling ─────────────────────────────────────────────────────────────

    def _on_point(self, ix: int, iy: int):
        if self._current_frame_no is None:
            return
        if self._tool_suggest.isChecked():
            self._request_suggestions(ix, iy)
            return
        self._vset.set_label(self._current_frame_no, self._pid(),
                             {"x": ix, "y": iy})
        self._save()
        self._advance_pid()
        self._refresh_current()

    def _set_state_label(self, state: str):
        if self._current_frame_no is None:
            return
        self._vset.set_label(self._current_frame_no, self._pid(), state)
        self._save()
        self._advance_pid()
        self._refresh_current()

    def _clear_label(self):
        if self._current_frame_no is None:
            return
        self._vset.clear_label(self._current_frame_no, self._pid())
        self._save()
        self._refresh_current()

    def _advance_pid(self):
        nxt = self._pid_combo.currentIndex() + 1
        if 0 < nxt < self._pid_combo.count():
            self._pid_combo.setCurrentIndex(nxt)

    def _on_box_drawn(self, x1: int, y1: int, x2: int, y2: int):
        if self._current_frame_no is None or x2 - x1 < 2 or y2 - y1 < 2:
            return
        self._vset.add_object(self._current_frame_no,
                              self._obj_name.text().strip() or "object",
                              (x1, y1, x2, y2))
        self._save()
        self._refresh_current()

    def _on_remove_object(self):
        if self._current_frame_no is None:
            return
        row = self._obj_list.currentRow()
        if row < 0:
            return
        self._vset.remove_object(self._current_frame_no, row)
        self._save()
        self._refresh_current()

    # ── Suggest flow (VP Builder pattern) ────────────────────────────────────

    def _request_suggestions(self, ix: int, iy: int):
        if self._suggest_busy or self._current_frame is None:
            return
        if self._suggester is None:
            if self._suggester_factory is not None:
                self._suggester = self._suggester_factory()
            else:
                from mindsight.GUI.region_suggest import RegionSuggester
                self._suggester = RegionSuggester()
        frame, suggester = self._current_frame, self._suggester
        self._suggest_busy = True
        self._status.setText("Suggesting…")

        def work():
            try:
                self._suggest_q.put(("ok", suggester.suggest(frame, ix, iy)))
            except Exception as exc:                       # pragma: no cover
                self._suggest_q.put(("err", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        self._suggest_timer.start(100)

    def _poll_suggest(self):
        try:
            kind, payload = self._suggest_q.get_nowait()
        except queue.Empty:
            return
        self._suggest_timer.stop()
        self._suggest_busy = False
        if kind == "err":
            self._status.setText(f"Suggest failed: {payload}")
            return
        self._canvas.set_suggestions(payload)
        self._status.setText(
            f"{len(payload)} proposal(s) — click one to accept."
            if payload else "No proposals here — try another point.")

    def _on_suggestion_accepted(self, index: int):
        boxes = self._canvas._suggestions
        if not (0 <= index < len(boxes)) or self._current_frame_no is None:
            return
        x1, y1, x2, y2 = boxes[index]
        self._vset.add_object(self._current_frame_no,
                              self._obj_name.text().strip() or "object",
                              (x1, y1, x2, y2))
        self._canvas.set_suggestions([])
        self._save()
        self._refresh_current()

    # ── Rendering / persistence ──────────────────────────────────────────────

    def _refresh_current(self):
        self._reload_frame_list(keep_frame=self._current_frame_no)
        self._render()

    def _render(self):
        self._label_list.clear()
        self._obj_list.clear()
        if self._current_frame_no is None or self._current_frame is None:
            self._canvas.set_image_data(None, [], [])
            return
        fno = self._current_frame_no
        import cv2
        disp = self._current_frame.copy()
        for i, (pid, v) in enumerate(
                sorted(self._vset.labels.get(fno, {}).items())):
            colour = _TARGET_COLOURS_BGR[i % len(_TARGET_COLOURS_BGR)]
            shown = f"P{pid}" if pid.isdigit() else pid
            if isinstance(v, dict):
                cv2.drawMarker(disp, (v["x"], v["y"]), colour,
                               cv2.MARKER_CROSS, 18, 2)
                cv2.putText(disp, shown, (v["x"] + 8, v["y"] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
                            cv2.LINE_AA)
                self._label_list.addItem(f"{shown}: ({v['x']}, {v['y']})")
            else:
                self._label_list.addItem(f"{shown}: {v}")
        crops = [{"x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"],
                  "label": b["name"]}
                 for b in self._vset.objects.get(fno, [])]
        for b in self._vset.objects.get(fno, []):
            self._obj_list.addItem(
                f"{b['name']}  ({b['x1']},{b['y1']})-({b['x2']},{b['y2']})")
        self._canvas.set_image_data(disp, [], crops)

    def _save(self):
        self._store.save(self._vset)

    def done(self, result):
        if hasattr(self._provider, "close"):
            self._provider.close()
        super().done(result)
