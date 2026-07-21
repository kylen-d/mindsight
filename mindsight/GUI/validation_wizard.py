"""
GUI/validation_wizard.py — Guided validation-set wizard (W4C redesign).

Matches the Build-Project wizard's house style (user feedback: the W4B
three-page stack didn't look or feel like the app's other wizards): a
left step list with visited-step navigation, a green Continue button
with per-step validation, and plain-English titles.

Four steps:

1. **Set** — name, how many people are on screen, participant labels
   (seeded from project metadata in project mode), and WHAT is being
   validated: one video (picked right here; the Videos step is
   skipped), several videos, or a whole project — the wizard reshapes
   itself around that choice (user clarification, W4C round 2).
2. **Videos** — one or more clips per set (W4C multi-video sets); add
   individual files or every staged video of a project.  Hidden in
   single-video mode.
3. **Frames** — per-video sampling ("sample every N frames" with a
   seconds-equivalent readout) or single-frame adds; "Sample ALL
   videos" covers the whole set at once.
4. **Label** — click where each participant looks (or mark off-screen /
   uncertain / skip).  The participant selector RESETS to the first
   participant on every frame change, advances on each label, hops to
   the next frame after the last participant, and every mutation is
   undoable (Ctrl+Z / Undo button) and autosaved.

A ``frame_provider_factory`` seam keeps tests video-free.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mindsight.GUI.widgets import ImageCanvas
from mindsight.validation import (
    ValidationSet,
    ValidationSetError,
    clips_from_project,
)

_TARGET_COLOURS_BGR = [(80, 200, 255), (100, 255, 130), (255, 150, 50),
                       (200, 149, 255)]

_GO_GREEN = ("QPushButton{background:#2a7a2a;color:white;"
             "font-weight:bold;padding:4px 26px;}"
             "QPushButton:disabled{background:#33333f;color:#777;}")

PAGE_SET, PAGE_VIDEOS, PAGE_FRAMES, PAGE_LABEL = 0, 1, 2, 3
_STEPS = ("Set", "Videos", "Frames", "Label")

_UNDO_LIMIT = 200


class VideoFrameProvider:
    """Default frame provider: decodes frames from one clip's video."""

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
    """Step list + stacked pages, Build-Project-wizard style."""

    def __init__(self, store, vset: ValidationSet | None = None, parent=None,
                 frame_provider_factory=VideoFrameProvider):
        super().__init__(parent)
        self._store = store
        self._provider_factory = frame_provider_factory
        self._providers: dict = {}
        self._vset = vset
        self._editing = vset is not None
        self._clip_idx = 0
        self._current_frame_no: int | None = None
        self._current_frame = None
        self._undo_stack: list = []
        self.setWindowTitle("New validation set" if not self._editing
                            else f"Annotate — {vset.name}")
        self.resize(1100, 640)

        outer = QHBoxLayout(self)
        self._steps = QListWidget()
        self._steps.setFixedWidth(150)
        for i, name in enumerate(_STEPS):
            self._steps.addItem(f"{i + 1}.  {name}")
        self._steps.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._steps.currentRowChanged.connect(self._step_clicked)
        outer.addWidget(self._steps)

        right = QVBoxLayout()
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_set_page())
        self._stack.addWidget(self._build_videos_page())
        self._stack.addWidget(self._build_frames_page())
        self._stack.addWidget(self._build_label_page())
        right.addWidget(self._stack, 1)

        nav = QHBoxLayout()
        self._gate_msg = QLabel("")
        self._gate_msg.setWordWrap(True)
        self._gate_msg.setStyleSheet("color: #b58900;")
        nav.addWidget(self._gate_msg, 1)
        self._back_btn = QPushButton("‹  Back")
        self._back_btn.clicked.connect(self._go_back)
        nav.addWidget(self._back_btn)
        self._next_btn = QPushButton("Continue  ›")
        self._next_btn.setStyleSheet(_GO_GREEN)
        self._next_btn.setMinimumHeight(30)
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)
        right.addLayout(nav)
        outer.addLayout(right, 1)

        QShortcut(QKeySequence.StandardKey.Undo, self, self._undo)

        self._visited = 0
        if self._editing:
            self._load_existing_fields()
            self._visited = len(_STEPS) - 1        # all steps clickable
            self._set_step(PAGE_LABEL if vset.total_frames()
                           else (PAGE_FRAMES if vset.clips else PAGE_VIDEOS))
        else:
            self._set_step(PAGE_SET)

    # ── Step navigation ──────────────────────────────────────────────────────

    def _set_step(self, i: int):
        self._stack.setCurrentIndex(i)
        self._gate_msg.setText("")
        self._steps.blockSignals(True)
        self._steps.setCurrentRow(i)
        self._visited = max(self._visited, i)
        for row in range(self._steps.count()):
            item = self._steps.item(row)
            flag = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            item.setFlags(flag if row <= self._visited
                          else Qt.ItemFlag.NoItemFlags)
        self._steps.blockSignals(False)
        self._back_btn.setEnabled(i > 0)
        self._next_btn.setText("Finish" if i == PAGE_LABEL
                               else "Continue  ›")
        if i == PAGE_VIDEOS:
            self._refresh_videos_page()
        elif i == PAGE_FRAMES:
            self._refresh_frames_page()
        elif i == PAGE_LABEL:
            self._refresh_label_page()

    def _step_clicked(self, row: int):
        if 0 <= row <= self._visited and row != self._stack.currentIndex():
            if self._stack.currentIndex() == PAGE_SET \
                    and not self._commit_set_page():
                self._steps.blockSignals(True)
                self._steps.setCurrentRow(PAGE_SET)
                self._steps.blockSignals(False)
                return
            self._set_step(row)

    def _go_back(self):
        i = self._stack.currentIndex()
        prev = i - 1
        if prev == PAGE_VIDEOS and self._mode() == "single":
            prev = PAGE_SET                  # Videos step skipped
        self._set_step(max(0, prev))

    def _go_next(self):
        i = self._stack.currentIndex()
        if i == PAGE_SET and not self._commit_set_page():
            return
        if i == PAGE_VIDEOS and not self._validate_videos():
            return
        if i == PAGE_FRAMES and self._vset.total_frames() == 0:
            self._gate_msg.setText(
                "Add at least one frame before labeling.")
            return
        if i == PAGE_LABEL:
            self.accept()
            return
        nxt = i + 1
        if nxt == PAGE_VIDEOS and self._mode() == "single":
            nxt = PAGE_FRAMES                # Videos step skipped
        self._set_step(nxt)

    # ── Step 1: Set ──────────────────────────────────────────────────────────

    def _build_set_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        title = QLabel("About your validation set")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)
        intro = QLabel(
            "A validation set is ground truth for your own footage: for a "
            "sample of frames you mark where each participant is actually "
            "looking, and Validate scores the current settings against "
            "those marks.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #888;")
        lay.addWidget(intro)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. office-a")
        form.addRow("Set name:", self._name_edit)

        self._people_spin = QSpinBox()
        self._people_spin.setRange(1, 8)
        self._people_spin.setValue(2)
        self._people_spin.setToolTip(
            "How many people are usually on screen in each video -- "
            "drives the participant selector on the Label step.")
        self._people_spin.valueChanged.connect(self._sync_default_labels)
        form.addRow("People per video:", self._people_spin)

        self._labels_edit = QLineEdit()
        self._labels_edit.setPlaceholderText("P0, P1")
        self._labels_edit.setToolTip(
            "The labels offered while labeling, comma-separated.  Keep the "
            "P0/P1 defaults (P0 = left-most face) or use your study's own "
            "labels (e.g. S70, S71) to match custom participant IDs.")
        form.addRow("Participant labels:", self._labels_edit)
        lay.addLayout(form)
        self._sync_default_labels()

        # What is being validated (user clarification): one video /
        # several videos / a whole project.  The wizard reshapes around
        # the choice -- single-video mode picks the file right here and
        # skips the Videos step.
        mode_title = QLabel("What are you validating?")
        mode_title.setStyleSheet("font-weight: bold;")
        lay.addWidget(mode_title)
        self._mode_single = QRadioButton("One video")
        self._mode_multi = QRadioButton("Several videos")
        self._mode_project = QRadioButton("A whole project")
        self._mode_single.setChecked(True)
        mode_row = QHBoxLayout()
        for rb in (self._mode_single, self._mode_multi, self._mode_project):
            rb.toggled.connect(self._on_mode_changed)
            mode_row.addWidget(rb)
        mode_row.addStretch(1)
        lay.addLayout(mode_row)

        # Single-video mode: the clip is picked right here.
        self._video_row = QWidget()
        vrow = QHBoxLayout(self._video_row)
        vrow.setContentsMargins(0, 0, 0, 0)
        vrow.addWidget(QLabel("Source video:"))
        self._video_edit = QLineEdit()
        self._video_edit.setPlaceholderText("choose the clip to label…")
        vrow.addWidget(self._video_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_video)
        vrow.addWidget(browse)
        lay.addWidget(self._video_row)

        # Project mode: the project is picked right here.
        self._proj_row = QWidget()
        prow = QHBoxLayout(self._proj_row)
        prow.setContentsMargins(0, 0, 0, 0)
        proj_btn = QPushButton("Choose project…")
        proj_btn.setToolTip(
            "Pick a MindSight project folder: every staged video becomes a "
            "clip of this set, and participant labels come from the "
            "project's run metadata.")
        proj_btn.clicked.connect(self._on_import_project)
        prow.addWidget(proj_btn)
        self._proj_note = QLabel("")
        self._proj_note.setStyleSheet("color: #888;")
        prow.addWidget(self._proj_note, 1)
        lay.addWidget(self._proj_row)

        self._multi_note = QLabel(
            "You will add the videos on the next step.")
        self._multi_note.setStyleSheet("color: #888;")
        lay.addWidget(self._multi_note)

        lay.addStretch(1)
        self._on_mode_changed()
        return page

    def _mode(self) -> str:
        if self._mode_project.isChecked():
            return "project"
        if self._mode_multi.isChecked():
            return "multi"
        return "single"

    def _on_mode_changed(self, *_):
        mode = self._mode()
        self._video_row.setVisible(mode == "single")
        self._proj_row.setVisible(mode == "project")
        self._multi_note.setVisible(mode == "multi")
        if hasattr(self, "_gate_msg"):       # nav row builds after pages
            self._gate_msg.setText("")
        # The Videos step only exists for multi/project sets.
        item = self._steps.item(PAGE_VIDEOS)
        if item is not None:
            item.setHidden(mode == "single")

    def _on_browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the set's source video", "",
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*)")
        if path:
            self._video_edit.setText(path)

    def _default_labels_text(self) -> str:
        return ", ".join(f"P{i}" for i in range(self._people_spin.value()))

    def _sync_default_labels(self, *_):
        """Keep the labels line following the people count until the user
        types their own labels."""
        current = self._labels_edit.text().strip()
        if not current or current == getattr(self, "_last_default", ""):
            self._labels_edit.setText(self._default_labels_text())
        self._last_default = self._default_labels_text()

    def _participants(self) -> list[str]:
        parts = [p.strip() for p in self._labels_edit.text().split(",")
                 if p.strip()]
        return parts or [f"P{i}" for i in range(self._people_spin.value())]

    def _on_import_project(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose a MindSight project folder")
        if not path:
            return
        try:
            clips, participants = clips_from_project(path)
        except ValidationSetError as exc:
            self._gate_msg.setText(str(exc))
            return
        self._pending_clips = clips
        if participants:
            self._labels_edit.setText(", ".join(participants))
            self._people_spin.setValue(
                min(max(len(participants), 1), self._people_spin.maximum()))
        if not self._name_edit.text().strip():
            self._name_edit.setText(f"{Path(path).name}-validation")
        self._proj_note.setText(
            f"{len(clips)} video(s) staged from {Path(path).name} -- "
            "they appear on the Videos step.")

    def _load_existing_fields(self):
        self._name_edit.setText(self._vset.name)
        self._name_edit.setReadOnly(True)
        self._name_edit.setToolTip("Existing sets keep their name.")
        if self._vset.participants:
            self._labels_edit.setText(", ".join(self._vset.participants))
            self._people_spin.setValue(min(
                max(len(self._vset.participants), 1),
                self._people_spin.maximum()))
        # Reflect the set's shape in the mode choice (a one-clip set reads
        # as single-video; switching to 'Several videos' re-reveals the
        # Videos step to grow it).
        if len(self._vset.clips) > 1:
            self._mode_multi.setChecked(True)
        else:
            self._mode_single.setChecked(True)
            if self._vset.clips:
                self._video_edit.setText(self._vset.video)
        self._on_mode_changed()

    def _commit_set_page(self) -> bool:
        name = self._name_edit.text().strip()
        if not name:
            self._gate_msg.setText("Give the set a name first.")
            return False
        mode = self._mode()
        if mode == "single":
            video = self._video_edit.text().strip()
            if not video:
                self._gate_msg.setText("Choose the video to label.")
                return False
            if not Path(video).is_file():
                self._gate_msg.setText("That video file does not exist.")
                return False
        if mode == "project" and not (
                getattr(self, "_pending_clips", None)
                or (self._vset is not None and self._vset.clips)):
            self._gate_msg.setText(
                "Choose a project first -- its staged videos become the "
                "set's clips.")
            return False
        try:
            if self._vset is None:
                self._vset = ValidationSet(name=name)
            self._vset.participants = self._participants()
            if mode == "single":
                if not self._vset.clips:
                    self._vset.add_clip(video)
                elif self._vset.clips[0].video != video:
                    self._vset.clips[0].video = video
            for info in getattr(self, "_pending_clips", []) or []:
                if self._vset.clip_for_video(info["video"]) is None:
                    self._vset.add_clip(info["video"])
            self._pending_clips = []
            self._save()
        except ValidationSetError as exc:
            self._gate_msg.setText(str(exc))
            return False
        return True

    # ── Step 2: Videos ───────────────────────────────────────────────────────

    def _build_videos_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        title = QLabel("Videos in this set")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)
        hint = QLabel(
            "A set can hold one clip or a whole study.  Validate runs every "
            "video here and pools the score (with a per-video breakdown).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        lay.addWidget(hint)

        self._video_list = QListWidget()
        lay.addWidget(self._video_list, 1)

        row = QHBoxLayout()
        add_btn = QPushButton("Add video…")
        add_btn.clicked.connect(self._on_add_video)
        row.addWidget(add_btn)
        proj_btn = QPushButton("Add all from a project…")
        proj_btn.clicked.connect(self._on_add_project_videos)
        row.addWidget(proj_btn)
        rm_btn = QPushButton("Remove selected")
        rm_btn.clicked.connect(self._on_remove_video)
        row.addWidget(rm_btn)
        row.addStretch(1)
        lay.addLayout(row)
        return page

    def _refresh_videos_page(self):
        self._video_list.clear()
        for clip in self._vset.clips:
            p = Path(clip.video)
            n = len(clip.frames())
            missing = "" if p.is_file() else "   (missing!)"
            item_text = (f"{p.name}   ({n} frame{'s' if n != 1 else ''} "
                         f"chosen){missing}")
            self._video_list.addItem(item_text)
            self._video_list.item(
                self._video_list.count() - 1).setToolTip(clip.video)

    def _on_add_video(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add video(s) to the set", "",
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*)")
        for path in paths:
            if path and self._vset.clip_for_video(path) is None:
                self._vset.add_clip(path)
        if paths:
            self._save()
            self._refresh_videos_page()

    def _on_add_project_videos(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose a MindSight project folder")
        if not path:
            return
        try:
            clips, participants = clips_from_project(path)
        except ValidationSetError as exc:
            self._gate_msg.setText(str(exc))
            return
        for info in clips:
            if self._vset.clip_for_video(info["video"]) is None:
                self._vset.add_clip(info["video"])
        for label in participants:
            if label not in self._vset.participants:
                self._vset.participants.append(label)
        self._save()
        self._refresh_videos_page()

    def _on_remove_video(self):
        row = self._video_list.currentRow()
        if not (0 <= row < len(self._vset.clips)):
            return
        clip = self._vset.clips[row]
        if clip.frames():
            from PyQt6.QtWidgets import QMessageBox
            if QMessageBox.question(
                    self, "Remove video",
                    f"Remove {Path(clip.video).name} and its "
                    f"{len(clip.frames())} labeled frame(s)?") \
                    != QMessageBox.StandardButton.Yes:
                return
        self._vset.remove_clip(row)
        self._clip_idx = 0
        self._save()
        self._refresh_videos_page()

    def _validate_videos(self) -> bool:
        if not self._vset.clips:
            self._gate_msg.setText("Add at least one video.")
            return False
        missing = [Path(c.video).name for c in self._vset.clips
                   if not Path(c.video).is_file()]
        if missing:
            self._gate_msg.setText(
                "Missing video file(s): " + ", ".join(missing))
            return False
        return True

    # ── Step 3: Frames ───────────────────────────────────────────────────────

    def _build_frames_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        title = QLabel("Choose frames to label")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)

        form = QFormLayout()
        self._frames_video_combo = QComboBox()
        self._frames_video_combo.currentIndexChanged.connect(
            self._on_frames_video_changed)
        form.addRow("Video:", self._frames_video_combo)
        lay.addLayout(form)

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

        btn_row = QHBoxLayout()
        sample_btn = QPushButton("Add sampled frames")
        sample_btn.setToolTip("Sample the video selected above.")
        sample_btn.clicked.connect(self._on_sample)
        btn_row.addWidget(sample_btn)
        self._sample_all_btn = QPushButton("Sample ALL videos")
        self._sample_all_btn.setToolTip(
            "Apply the same sampling to every video in the set.")
        self._sample_all_btn.clicked.connect(self._on_sample_all)
        btn_row.addWidget(self._sample_all_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Or add a single frame, #"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 10_000_000)
        self._frame_spin.setToolTip(
            "A specific frame number to add (0 = first frame).")
        row2.addWidget(self._frame_spin)
        add_btn = QPushButton("Add frame")
        add_btn.clicked.connect(self._on_add_frame)
        row2.addWidget(add_btn)
        row2.addStretch(1)
        lay.addLayout(row2)

        self._frames_count_label = QLabel("")
        self._frames_count_label.setWordWrap(True)
        lay.addWidget(self._frames_count_label)
        lay.addStretch(1)
        return page

    def _current_clip(self):
        if not self._vset or not self._vset.clips:
            return None
        self._clip_idx = min(self._clip_idx, len(self._vset.clips) - 1)
        return self._vset.clips[self._clip_idx]

    def _provider_for(self, video: str):
        if video not in self._providers:
            self._providers[video] = self._provider_factory(video)
        return self._providers[video]

    def _sync_video_combo(self, combo: QComboBox):
        combo.blockSignals(True)
        combo.clear()
        for clip in self._vset.clips:
            combo.addItem(Path(clip.video).name, clip.video)
        combo.setCurrentIndex(self._clip_idx)
        combo.blockSignals(False)
        combo.setVisible(len(self._vset.clips) > 1)

    def _on_frames_video_changed(self, idx: int):
        if 0 <= idx < len(self._vset.clips):
            self._clip_idx = idx
            self._refresh_frames_page(resync_combo=False)

    def _refresh_frames_page(self, resync_combo: bool = True):
        clip = self._current_clip()
        if clip is None:
            return
        if resync_combo:
            self._sync_video_combo(self._frames_video_combo)
        provider = self._provider_for(clip.video)
        fps = float(getattr(provider, "fps", 0.0) or 0.0)
        total = int(getattr(provider, "count", 0) or 0)
        info = f"Clip: {Path(clip.video).name} — {total} frames"
        if fps > 0 and total:
            info += f" · {fps:.2f} fps · {total / fps:.1f} s"
        self._video_info.setText(info)
        self._update_sample_readout()
        self._sample_all_btn.setVisible(len(self._vset.clips) > 1)
        parts = []
        for c in self._vset.clips:
            n = len(c.frames())
            parts.append(f"{Path(c.video).name}: {n}")
        total_n = self._vset.total_frames()
        self._frames_count_label.setText(
            (f"{total_n} frame(s) in the set"
             + (f"  ({', '.join(parts)})" if len(parts) > 1 else "."))
            if total_n else
            "No frames yet — add sampled frames above to start labeling.")

    def _update_sample_readout(self, *_):
        clip = self._current_clip()
        if clip is None:
            return
        provider = self._provider_for(clip.video)
        step = self._every_spin.value()
        fps = float(getattr(provider, "fps", 0.0) or 0.0)
        total = int(getattr(provider, "count", 0) or 0)
        parts = []
        if fps > 0:
            parts.append(f"= every {step / fps:.1f} s at {fps:.0f} fps")
        if total > 0:
            parts.append(f"→ adds ~{len(range(0, total, step))} frames")
        self._sample_readout.setText("   ".join(parts))

    def _sample_clip(self, clip) -> list[int]:
        provider = self._provider_for(clip.video)
        total = int(getattr(provider, "count", 0) or 0)
        step = self._every_spin.value()
        added = []
        for fno in range(0, total, step):
            if fno not in clip.labels:
                clip.add_frame(fno)
                added.append(fno)
        clip.every = step
        return added

    def _on_sample(self):
        clip = self._current_clip()
        if clip is None:
            return
        added = self._sample_clip(clip)
        if added:
            self._push_undo(("add_frames", self._clip_idx, added))
        self._save()
        self._refresh_frames_page()

    def _on_sample_all(self):
        for i, clip in enumerate(self._vset.clips):
            added = self._sample_clip(clip)
            if added:
                self._push_undo(("add_frames", i, added))
        self._save()
        self._refresh_frames_page()

    def _on_add_frame(self):
        clip = self._current_clip()
        if clip is None:
            return
        fno = self._frame_spin.value()
        if fno not in clip.labels:
            clip.add_frame(fno)
            self._push_undo(("add_frames", self._clip_idx, [fno]))
        self._save()
        self._refresh_frames_page()

    # ── Step 4: Label ────────────────────────────────────────────────────────

    def _build_label_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        top = QHBoxLayout()
        self._label_video_combo = QComboBox()
        self._label_video_combo.currentIndexChanged.connect(
            self._on_label_video_changed)
        top.addWidget(self._label_video_combo)
        self._progress_label = QLabel("")
        top.addWidget(self._progress_label, 1)
        root.addLayout(top)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Frames"))
        self._frame_list = QListWidget()
        self._frame_list.currentRowChanged.connect(self._on_frame_row)
        left.addWidget(self._frame_list, 1)
        rm_btn = QPushButton("Remove frame")
        rm_btn.clicked.connect(self._on_remove_frame)
        left.addWidget(rm_btn)
        body.addLayout(left, 0)

        center = QVBoxLayout()
        hint = QLabel(
            "Click where the selected participant is looking — the "
            "selection advances by itself, and hops to the next frame "
            "after the last participant.  n / space = next frame, "
            "b = back, Ctrl+Z = undo.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
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
        body.addLayout(center, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Participant"))
        self._pid_combo = QComboBox()
        self._pid_combo.setEditable(True)
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
        self._undo_btn = QPushButton("↶ Undo")
        self._undo_btn.setToolTip("Undo the last change (Ctrl+Z).")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._undo)
        right.addWidget(self._undo_btn)

        right.addWidget(QLabel("Labels on this frame"))
        self._label_list = QListWidget()
        right.addWidget(self._label_list, 1)
        body.addLayout(right, 0)
        root.addLayout(body, 1)
        return page

    def _on_label_video_changed(self, idx: int):
        if 0 <= idx < len(self._vset.clips):
            self._clip_idx = idx
            self._current_frame_no = None
            self._reload_frame_list()

    def _refresh_label_page(self):
        self._sync_video_combo(self._label_video_combo)
        self._sync_pid_combo()
        self._reload_frame_list(keep_frame=self._current_frame_no)

    def _sync_pid_combo(self):
        labels = (self._vset.participants
                  or [f"P{i}" for i in range(self._people_spin.value())])
        current = self._pid_combo.currentText()
        self._pid_combo.blockSignals(True)
        self._pid_combo.clear()
        self._pid_combo.addItems(labels)
        if current in labels:
            self._pid_combo.setCurrentText(current)
        self._pid_combo.blockSignals(False)

    def _n_participants(self) -> int:
        return max(self._pid_combo.count(), 1)

    def _frame_numbers(self) -> list[int]:
        clip = self._current_clip()
        return clip.frames() if clip is not None else []

    def _reload_frame_list(self, keep_frame=None):
        clip = self._current_clip()
        self._frame_list.blockSignals(True)
        self._frame_list.clear()
        row_for = 0
        n_pids = self._n_participants()
        for i, fno in enumerate(self._frame_numbers()):
            n = len(clip.labels.get(fno, {})) if clip else 0
            check = " ✓" if n >= n_pids else ""
            self._frame_list.addItem(f"frame {fno}   ({n}/{n_pids}){check}")
            if keep_frame is not None and fno == keep_frame:
                row_for = i
        self._frame_list.blockSignals(False)
        if self._frame_list.count():
            self._frame_list.setCurrentRow(row_for)
            # currentRowChanged does not fire when the row is unchanged --
            # render explicitly so the canvas always matches.
            self._on_frame_row(row_for)
        else:
            self._current_frame_no = None
            self._current_frame = None
            self._render()
        self._update_progress_label()

    def _update_progress_label(self):
        clip = self._current_clip()
        if clip is None:
            self._progress_label.setText("")
            return
        frames = self._frame_numbers()
        n_pids = self._n_participants()
        done = sum(1 for f in frames
                   if len(clip.labels.get(f, {})) >= n_pids)
        pos = ""
        if self._current_frame_no is not None \
                and self._current_frame_no in frames:
            pos = (f"frame {frames.index(self._current_frame_no) + 1}"
                   f"/{len(frames)} · ")
        self._progress_label.setText(
            f"{pos}{done}/{len(frames)} frames fully labeled · "
            f"{self._vset.point_label_count()} points in the set")

    def _on_frame_row(self, row: int):
        frames = self._frame_numbers()
        if not (0 <= row < len(frames)):
            return
        changed = frames[row] != self._current_frame_no
        self._current_frame_no = frames[row]
        clip = self._current_clip()
        provider = self._provider_for(clip.video)
        self._current_frame = provider.read(self._current_frame_no)
        self._label_status.setText(
            "" if self._current_frame is not None
            else f"Frame {self._current_frame_no}: cannot decode")
        if changed:
            # Creature comfort (user request): a fresh frame always starts
            # at the first participant.
            self._pid_combo.setCurrentIndex(0)
        self._render()
        self._update_progress_label()

    def _on_remove_frame(self):
        if self._current_frame_no is None:
            return
        clip = self._current_clip()
        fno = self._current_frame_no
        self._push_undo(("remove_frame", self._clip_idx, fno,
                         dict(clip.labels.get(fno, {})),
                         list(clip.objects.get(fno, []))))
        clip.remove_frame(fno)
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
        """Next participant; past the last one, hop to the next frame
        (which resets the selector to the first participant)."""
        nxt = self._pid_combo.currentIndex() + 1
        if 0 < nxt < self._pid_combo.count():
            self._pid_combo.setCurrentIndex(nxt)
        elif nxt >= self._pid_combo.count():
            self._step(+1)

    def _record_label(self, value):
        if self._current_frame_no is None:
            return
        clip = self._current_clip()
        pid = self._pid()
        self._push_undo(("label", self._clip_idx, self._current_frame_no,
                         pid, clip.get_label(self._current_frame_no, pid)))
        clip.set_label(self._current_frame_no, pid, value)
        self._save()
        self._advance_pid()
        self._refresh_current()

    def _on_point(self, ix: int, iy: int):
        self._record_label({"x": ix, "y": iy})

    def _set_state(self, state: str):
        self._record_label(state)

    def _clear_label(self):
        if self._current_frame_no is None:
            return
        clip = self._current_clip()
        pid = self._pid()
        prev = clip.get_label(self._current_frame_no, pid)
        if prev is None:
            return
        self._push_undo(("label", self._clip_idx, self._current_frame_no,
                         pid, prev))
        clip.clear_label(self._current_frame_no, pid)
        self._save()
        self._refresh_current()

    # ── Undo ─────────────────────────────────────────────────────────────────

    def _push_undo(self, entry):
        self._undo_stack.append(entry)
        del self._undo_stack[:-_UNDO_LIMIT]
        self._undo_btn.setEnabled(True)

    def _undo(self):
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        kind, clip_idx = entry[0], entry[1]
        if not (0 <= clip_idx < len(self._vset.clips)):
            return
        clip = self._vset.clips[clip_idx]
        if kind == "label":
            _k, _c, frame, pid, prev = entry
            if prev is None:
                clip.clear_label(frame, pid)
            else:
                clip.set_label(frame, pid, prev)
            self._clip_idx = clip_idx
            self._current_frame_no = frame
        elif kind == "add_frames":
            _k, _c, frames = entry
            for fno in frames:
                clip.remove_frame(fno)
        elif kind == "remove_frame":
            _k, _c, frame, labels, objects = entry
            clip.labels[int(frame)] = labels
            if objects:
                clip.objects[int(frame)] = objects
            self._clip_idx = clip_idx
            self._current_frame_no = frame
        self._undo_btn.setEnabled(bool(self._undo_stack))
        self._save()
        page = self._stack.currentIndex()
        if page == PAGE_LABEL:
            self._refresh_label_page()
        elif page == PAGE_FRAMES:
            self._refresh_frames_page()
        elif page == PAGE_VIDEOS:
            self._refresh_videos_page()

    # ── Rendering / persistence ──────────────────────────────────────────────

    def _refresh_current(self):
        self._reload_frame_list(keep_frame=self._current_frame_no)

    def _render(self):
        self._label_list.clear()
        if self._current_frame_no is None or self._current_frame is None:
            self._canvas.set_image_data(None, [], [])
            return
        import cv2
        clip = self._current_clip()
        disp = self._current_frame.copy()
        for i, (pid, v) in enumerate(
                sorted(clip.labels.get(self._current_frame_no, {}).items())):
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
        for provider in self._providers.values():
            if hasattr(provider, "close"):
                provider.close()
        self._providers = {}
        super().done(result)
