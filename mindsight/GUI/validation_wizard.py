"""
GUI/validation_wizard.py — Guided validation-set wizard (W4B rework).

Replaces the single-screen annotator dialog (user feedback: unclear
units, mystery controls, no visible path from "new set" to "labeled
frames").  Three self-explaining steps on a stacked layout:

1. **Set** (new sets only): name the set and pick its source video.
2. **Frames**: sample every N *frames* — the spinner is labeled, and a
   live readout translates it ("= every 1.0 s at 30 fps → adds ~29
   frames").  You cannot reach labeling with zero frames.
3. **Label**: click where each participant is looking on every frame
   (or mark off-screen / uncertain / skip).  Gaze targets only — the
   object-box tools are shelved (user ruling: gaze-only v1).

Every mutation autosaves through the ValidationStore.  A
``frame_provider_factory`` seam keeps tests video-free.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mindsight.GUI.widgets import ImageCanvas
from mindsight.validation import ValidationSet, ValidationSetError

_TARGET_COLOURS_BGR = [(80, 200, 255), (100, 255, 130), (255, 150, 50),
                       (200, 149, 255)]

PAGE_SET, PAGE_FRAMES, PAGE_LABEL = 0, 1, 2


class VideoFrameProvider:
    """Default frame provider: decodes frames from the set's video."""

    def __init__(self, video_path: str):
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(video_path))
        self.count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self, frame_no: int):
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        self._cap.release()


class ValidationSetWizard(QDialog):
    """Set -> Frames -> Label, with per-step gating and explanations."""

    def __init__(self, store, vset: ValidationSet | None = None, parent=None,
                 frame_provider_factory=VideoFrameProvider):
        super().__init__(parent)
        self._store = store
        self._provider_factory = frame_provider_factory
        self._provider = None
        self._vset = vset
        self._current_frame_no: int | None = None
        self._current_frame = None
        new_set = vset is None
        self.setWindowTitle("New validation set" if new_set
                            else f"Annotate — {vset.name}")
        self.resize(1100, 640)

        lay = QVBoxLayout(self)
        self._stack = QStackedWidget()
        lay.addWidget(self._stack, 1)
        self._stack.addWidget(self._build_set_page())
        self._stack.addWidget(self._build_frames_page())
        self._stack.addWidget(self._build_label_page())

        if new_set:
            self._go(PAGE_SET)
        else:
            self._open_provider()
            self._go(PAGE_LABEL if vset.frames() else PAGE_FRAMES)

    # ── Page 1: set name + video ─────────────────────────────────────────────

    def _build_set_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        intro = QLabel(
            "A validation set is a handful of labeled frames from one "
            "video: you mark where each participant is actually looking, "
            "and Validate scores the current settings against those marks.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Set name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. office-a")
        self._name_edit.textChanged.connect(self._update_set_gate)
        row.addWidget(self._name_edit, 1)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Source video:"))
        self._video_edit = QLineEdit()
        self._video_edit.setPlaceholderText("choose the clip to label…")
        self._video_edit.textChanged.connect(self._update_set_gate)
        row2.addWidget(self._video_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_video)
        row2.addWidget(browse)
        lay.addLayout(row2)

        self._set_gate_msg = QLabel("")
        self._set_gate_msg.setWordWrap(True)
        lay.addWidget(self._set_gate_msg)
        lay.addStretch(1)

        nav = QHBoxLayout()
        nav.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        nav.addWidget(cancel)
        self._set_next_btn = QPushButton("Next: choose frames ▶")
        self._set_next_btn.setEnabled(False)
        self._set_next_btn.clicked.connect(self._on_set_next)
        nav.addWidget(self._set_next_btn)
        lay.addLayout(nav)
        return page

    def _on_browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the set's source video", "",
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*)")
        if path:
            self._video_edit.setText(path)

    def _update_set_gate(self, *_):
        name = self._name_edit.text().strip()
        video = self._video_edit.text().strip()
        ok = bool(name) and Path(video).is_file()
        msg = ""
        if name and video and not Path(video).is_file():
            msg = "That video file does not exist."
        self._set_gate_msg.setText(msg)
        self._set_next_btn.setEnabled(ok)

    def _on_set_next(self):
        try:
            self._vset = ValidationSet(name=self._name_edit.text().strip(),
                                       video=self._video_edit.text().strip())
            self._store.save(self._vset)
        except ValidationSetError as exc:
            self._set_gate_msg.setText(str(exc))
            return
        self._open_provider()
        self._go(PAGE_FRAMES)

    # ── Page 2: frame sampling ───────────────────────────────────────────────

    def _build_frames_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        self._video_info = QLabel("")
        self._video_info.setWordWrap(True)
        lay.addWidget(self._video_info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Sample every"))
        self._every_spin = QSpinBox()
        self._every_spin.setRange(1, 100000)
        self._every_spin.setValue(30)
        self._every_spin.setSuffix(" frames")
        self._every_spin.valueChanged.connect(self._update_sample_readout)
        row.addWidget(self._every_spin)
        self._sample_readout = QLabel("")
        row.addWidget(self._sample_readout, 1)
        lay.addLayout(row)

        sample_btn = QPushButton("Add sampled frames")
        sample_btn.clicked.connect(self._on_sample)
        lay.addWidget(sample_btn)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Or add a single frame, #"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 10_000_000)
        self._frame_spin.setToolTip(
            "A specific frame number to add to the set (0 = first frame).")
        row2.addWidget(self._frame_spin)
        add_btn = QPushButton("Add frame")
        add_btn.clicked.connect(self._on_add_frame)
        row2.addWidget(add_btn)
        row2.addStretch(1)
        lay.addLayout(row2)

        self._frames_count_label = QLabel("")
        lay.addWidget(self._frames_count_label)
        lay.addStretch(1)

        nav = QHBoxLayout()
        nav.addStretch(1)
        cancel = QPushButton("Close")
        cancel.clicked.connect(self.reject)
        nav.addWidget(cancel)
        self._frames_next_btn = QPushButton("Next: label frames ▶")
        self._frames_next_btn.setEnabled(False)
        self._frames_next_btn.clicked.connect(lambda: self._go(PAGE_LABEL))
        nav.addWidget(self._frames_next_btn)
        lay.addLayout(nav)
        return page

    def _open_provider(self):
        if self._provider is None:
            self._provider = self._provider_factory(self._vset.video)

    def _update_sample_readout(self, *_):
        step = self._every_spin.value()
        fps = float(getattr(self._provider, "fps", 0.0) or 0.0)
        total = int(getattr(self._provider, "count", 0) or 0)
        parts = []
        if fps > 0:
            parts.append(f"= every {step / fps:.1f} s at {fps:.0f} fps")
        if total > 0:
            parts.append(f"→ adds ~{len(range(0, total, step))} frames")
        self._sample_readout.setText("   ".join(parts))

    def _refresh_frames_page(self):
        fps = float(getattr(self._provider, "fps", 0.0) or 0.0)
        total = int(getattr(self._provider, "count", 0) or 0)
        info = f"Clip: {Path(self._vset.video).name} — {total} frames"
        if fps > 0 and total:
            info += f" · {fps:.2f} fps · {total / fps:.1f} s"
        self._video_info.setText(info)
        self._update_sample_readout()
        n = len(self._vset.frames())
        self._frames_count_label.setText(
            f"{n} frame(s) in the set." if n else
            "No frames yet — add sampled frames above to start labeling.")
        self._frames_next_btn.setEnabled(n > 0)

    def _on_sample(self):
        total = int(getattr(self._provider, "count", 0) or 0)
        step = self._every_spin.value()
        for fno in range(0, total, step):
            self._vset.add_frame(fno)
        self._vset.every = step
        self._save()
        self._refresh_frames_page()

    def _on_add_frame(self):
        self._vset.add_frame(self._frame_spin.value())
        self._save()
        self._refresh_frames_page()

    # ── Page 3: gaze-target labeling ─────────────────────────────────────────

    def _build_label_page(self) -> QWidget:
        page = QWidget()
        root = QHBoxLayout(page)

        left = QVBoxLayout()
        left.addWidget(QLabel("Frames"))
        self._frame_list = QListWidget()
        self._frame_list.currentRowChanged.connect(self._on_frame_row)
        left.addWidget(self._frame_list, 1)
        rm_btn = QPushButton("Remove frame")
        rm_btn.clicked.connect(self._on_remove_frame)
        left.addWidget(rm_btn)
        more_btn = QPushButton("◀ Add more frames…")
        more_btn.clicked.connect(lambda: self._go(PAGE_FRAMES))
        left.addWidget(more_btn)
        root.addLayout(left, 0)

        center = QVBoxLayout()
        hint = QLabel(
            "Click where the selected participant is looking; the "
            "selection then advances to the next participant.  "
            "n / space = next frame, b = back.")
        hint.setWordWrap(True)
        center.addWidget(hint)
        self._canvas = ImageCanvas()
        self._canvas.set_suggest_mode(True)          # clicks -> point_clicked
        self._canvas.point_clicked.connect(self._on_point)
        center.addWidget(self._canvas, 1)
        nav = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(lambda: self._step(-1))
        nav.addWidget(prev_btn)
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(lambda: self._step(+1))
        nav.addWidget(next_btn)
        nav.addStretch(1)
        self._label_status = QLabel("")
        nav.addWidget(self._label_status)
        center.addLayout(nav)
        root.addLayout(center, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Participant"))
        self._pid_combo = QComboBox()
        self._pid_combo.setEditable(True)
        self._pid_combo.addItems(["P0", "P1", "P2", "P3"])
        self._pid_combo.setToolTip(
            "Whose gaze you are marking.  P0 is the left-most face; "
            "type a custom label to match custom participant IDs.")
        right.addWidget(self._pid_combo)

        for text, state, tip in (
                ("Looking off-screen", "offscreen",
                 "This participant is looking somewhere outside the frame."),
                ("Uncertain", "uncertain",
                 "You cannot tell where they look — excluded from scoring."),
                ("Skip participant", "skip",
                 "Do not score this participant on this frame.")):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, s=state: self._set_state(s))
            right.addWidget(btn)
        clear_btn = QPushButton("Clear label")
        clear_btn.clicked.connect(self._clear_label)
        right.addWidget(clear_btn)

        right.addWidget(QLabel("Labels on this frame"))
        self._label_list = QListWidget()
        right.addWidget(self._label_list, 1)

        finish = QPushButton("Finish")
        finish.clicked.connect(self.accept)
        right.addWidget(finish)
        root.addLayout(right, 0)
        return page

    # ── Navigation / shared state ────────────────────────────────────────────

    def _go(self, page: int):
        self._stack.setCurrentIndex(page)
        if page == PAGE_FRAMES:
            self._refresh_frames_page()
        elif page == PAGE_LABEL:
            self._reload_frame_list(keep_frame=self._current_frame_no)

    def _frame_numbers(self) -> list[int]:
        return self._vset.frames()

    def _reload_frame_list(self, keep_frame=None):
        self._frame_list.blockSignals(True)
        self._frame_list.clear()
        row_for = 0
        for i, fno in enumerate(self._frame_numbers()):
            n = len(self._vset.labels.get(fno, {}))
            self._frame_list.addItem(
                f"frame {fno}   ({n} label{'s' if n != 1 else ''})")
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
        self._label_status.setText(
            "" if self._current_frame is not None
            else f"Frame {self._current_frame_no}: cannot decode")
        self._render()

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
        if (self._stack.currentIndex() == PAGE_LABEL
                and event.key() in (Qt.Key.Key_N, Qt.Key.Key_Space)):
            self._step(+1)
        elif (self._stack.currentIndex() == PAGE_LABEL
                and event.key() == Qt.Key.Key_B):
            self._step(-1)
        else:
            super().keyPressEvent(event)

    # ── Labeling ─────────────────────────────────────────────────────────────

    def _pid(self) -> str:
        """P<N> labels store as digit keys (the eval-harness convention,
        matching the gaze stream's face_idx); custom labels as typed."""
        text = self._pid_combo.currentText().strip() or "P0"
        if len(text) >= 2 and text[0] in "Pp" and text[1:].isdigit():
            return text[1:]
        return text

    def _advance_pid(self):
        nxt = self._pid_combo.currentIndex() + 1
        if 0 < nxt < self._pid_combo.count():
            self._pid_combo.setCurrentIndex(nxt)

    def _on_point(self, ix: int, iy: int):
        if self._current_frame_no is None:
            return
        self._vset.set_label(self._current_frame_no, self._pid(),
                             {"x": ix, "y": iy})
        self._save()
        self._advance_pid()
        self._refresh_current()

    def _set_state(self, state: str):
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

    # ── Rendering / persistence ──────────────────────────────────────────────

    def _refresh_current(self):
        self._reload_frame_list(keep_frame=self._current_frame_no)
        self._render()

    def _render(self):
        self._label_list.clear()
        if self._current_frame_no is None or self._current_frame is None:
            self._canvas.set_image_data(None, [], [])
            return
        import cv2
        disp = self._current_frame.copy()
        for i, (pid, v) in enumerate(
                sorted(self._vset.labels.get(self._current_frame_no,
                                             {}).items())):
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
        self._canvas.set_image_data(disp, [], [])

    def _save(self):
        self._store.save(self._vset)

    def done(self, result):
        if self._provider is not None and hasattr(self._provider, "close"):
            self._provider.close()
        super().done(result)
