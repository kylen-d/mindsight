"""
project_wizard.py
-----------------
The **Build New Project** wizard (UP3 / HP3): a guided, plain-English path
from "I have a folder of study videos" to a ready-to-run MindSight project,
with zero knowledge of the project layout required.

Five steps (left step list, right pane): Study -> Videos -> Tag each video ->
Pipeline -> Review & create.  The tagging step shows each video's MIDDLE
frame, one video at a time, with participant labels ordered by on-screen
position ("Leftmost person", "2nd from left", ...) -- the word "track" never
appears.  Creation drives the existing project layer verbatim:
``create_project`` + per-video ``stage_run`` (run-folder layout + run.yaml)
+ a pipeline preset copied/written to ``Pipeline/pipeline.yaml``.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .widgets import middle_frame_pixmap

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
_GO_GREEN = ("QPushButton{background:#2a7a2a;color:white;"
             "font-weight:bold;padding:4px 26px;}"
             "QPushButton:disabled{background:#33333f;color:#777;}")

_STEPS = ("Study", "Videos", "Tag each video", "Pipeline", "Review & create")


def _ordinal_label(i: int, total: int) -> str:
    """Position-based participant field labels, left to right."""
    if total == 1:
        return "Person"
    if i == 0:
        return "Leftmost person"
    if i == 1:
        return "2nd from left"
    if i == 2:
        return "3rd from left"
    return f"{i + 1}th from left"


def low_power_preset_path() -> Path | None:
    """The UNVALIDATED throughput preset, resolved next to the known-good one."""
    from mindsight.config_compat import known_good_preset_path
    kg = known_good_preset_path()
    if kg is None:
        return None
    lp = kg.parent / "pipeline_low_power.yaml"
    return lp if lp.is_file() else None


class BuildProjectWizard(QDialog):
    """Left step column + right pane; Back/Continue; Create on the last step."""

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build New Project")
        self.resize(860, 560)
        self._settings = settings          # RunSettingsStore (may be None)
        self.created_path: Path | None = None

        # Per-video wizard state: list of dicts
        # {source: Path, run_id: str, meta: dict, pixmap: QPixmap|None}
        self._videos: list[dict] = []
        self._tag_index = 0
        self._import_yaml: Path | None = None

        outer = QHBoxLayout(self)
        self._steps = QListWidget()
        self._steps.setFixedWidth(170)
        for i, name in enumerate(_STEPS):
            self._steps.addItem(f"{i + 1}.  {name}")
        self._steps.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._steps.currentRowChanged.connect(self._step_clicked)
        outer.addWidget(self._steps)

        right = QVBoxLayout()
        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_study_page())
        self._pages.addWidget(self._build_videos_page())
        self._pages.addWidget(self._build_tag_page())
        self._pages.addWidget(self._build_pipeline_page())
        self._pages.addWidget(self._build_review_page())
        right.addWidget(self._pages, 1)

        nav = QHBoxLayout()
        nav.addStretch(1)
        self._back_btn = QPushButton("‹  Back")
        self._back_btn.clicked.connect(self._go_back)
        nav.addWidget(self._back_btn)
        self._next_btn = QPushButton("Continue  ›")
        self._next_btn.setStyleSheet(_GO_GREEN)
        self._next_btn.setMinimumHeight(32)
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)
        right.addLayout(nav)
        outer.addLayout(right, 1)

        self._visited = 0                  # highest step reached
        self._set_step(0)

    # ── Step navigation ──────────────────────────────────────────────────────

    def _set_step(self, i: int):
        self._pages.setCurrentIndex(i)
        self._steps.blockSignals(True)
        self._steps.setCurrentRow(i)
        # Steps beyond the furthest visited are not clickable destinations.
        self._visited = max(self._visited, i)
        for row in range(self._steps.count()):
            item = self._steps.item(row)
            flag = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            item.setFlags(flag if row <= self._visited
                          else Qt.ItemFlag.NoItemFlags)
        self._steps.blockSignals(False)
        self._back_btn.setEnabled(i > 0)
        last = i == len(_STEPS) - 1
        self._next_btn.setText("Create Project" if last else "Continue  ›")
        if i == 2:
            self._show_tag_video(self._tag_index)
        if last:
            self._refresh_review()

    def _step_clicked(self, row: int):
        if 0 <= row <= self._visited and row != self._pages.currentIndex():
            self._set_step(row)

    def _go_back(self):
        self._set_step(max(0, self._pages.currentIndex() - 1))

    def _go_next(self):
        i = self._pages.currentIndex()
        if i == 0 and not self._validate_study():
            return
        if i == 1 and not self._validate_videos():
            return
        if i == 2:
            self._save_tag_fields()
            untagged = [v for v in self._videos
                        if not v["meta"].get("participants")]
            if untagged:
                reply = QMessageBox.question(
                    self, "Untagged videos",
                    f"{len(untagged)} video(s) have no participant labels "
                    "yet. You can add them later in Analyze Footage.\n\n"
                    "Continue anyway?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No)
                if reply != QMessageBox.StandardButton.Yes:
                    return
        if i == 3 and not self._validate_pipeline():
            return
        if i == len(_STEPS) - 1:
            self._create()
            return
        self._set_step(i + 1)

    # ── Step 1: Study ────────────────────────────────────────────────────────

    def _build_study_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        title = QLabel("About your study")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)

        form = QFormLayout()
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. CookingStudy2026")
        form.addRow("Project name:", self._name)

        loc_row = QHBoxLayout()
        self._location = QLineEdit()
        self._location.setPlaceholderText(
            "Folder the project will be created inside...")
        loc_browse = QPushButton("Browse...")
        loc_browse.clicked.connect(self._browse_location)
        loc_row.addWidget(self._location, 1)
        loc_row.addWidget(loc_browse)
        form.addRow("Create inside:", loc_row)

        self._people = QSpinBox()
        self._people.setRange(1, 8)
        self._people.setValue(2)
        self._people.setToolTip(
            "How many people are usually on screen in each video")
        form.addRow("People per video:", self._people)
        lay.addLayout(form)

        cond_title = QLabel("Conditions (optional)")
        cond_title.setStyleSheet("font-weight: bold;")
        lay.addWidget(cond_title)
        cond_hint = QLabel(
            "List your study's conditions (e.g. warm, cold). Each video can "
            "be tagged with them in step 3.")
        cond_hint.setStyleSheet("color: #888;")
        cond_hint.setWordWrap(True)
        lay.addWidget(cond_hint)
        cond_row = QHBoxLayout()
        self._cond_edit = QLineEdit()
        self._cond_edit.setPlaceholderText("Add a condition...")
        self._cond_edit.returnPressed.connect(self._add_condition)
        add_cond = QPushButton("Add")
        add_cond.clicked.connect(self._add_condition)
        rm_cond = QPushButton("Remove selected")
        rm_cond.clicked.connect(self._remove_condition)
        cond_row.addWidget(self._cond_edit, 1)
        cond_row.addWidget(add_cond)
        cond_row.addWidget(rm_cond)
        lay.addLayout(cond_row)
        self._cond_list = QListWidget()
        self._cond_list.setMaximumHeight(90)
        lay.addWidget(self._cond_list)

        notes_title = QLabel("Study notes (optional)")
        notes_title.setStyleSheet("font-weight: bold;")
        lay.addWidget(notes_title)
        self._notes = QTextEdit()
        self._notes.setPlaceholderText(
            "Anything a colleague should know about this project -- saved "
            "as notes.md in the project folder.")
        self._notes.setMaximumHeight(80)
        lay.addWidget(self._notes)
        lay.addStretch(1)
        return page

    def _browse_location(self):
        path = QFileDialog.getExistingDirectory(
            self, "Folder to create the project inside")
        if path:
            self._location.setText(path)

    def _add_condition(self):
        text = self._cond_edit.text().strip()
        if text and text not in self.conditions():
            self._cond_list.addItem(text)
        self._cond_edit.clear()

    def _remove_condition(self):
        for item in self._cond_list.selectedItems():
            self._cond_list.takeItem(self._cond_list.row(item))

    def conditions(self) -> list[str]:
        return [self._cond_list.item(i).text()
                for i in range(self._cond_list.count())]

    def _validate_study(self) -> bool:
        name = self._name.text().strip()
        if not name or any(sep in name for sep in ("/", "\\")):
            QMessageBox.warning(self, "Project name",
                                "Give the project a name (no / or \\).")
            return False
        loc_text = self._location.text().strip()
        # NOTE: Path("") is Path(".") -- an empty field must not silently
        # resolve to the current working directory.
        if not loc_text or not Path(loc_text).is_dir():
            QMessageBox.warning(self, "Location",
                                "Choose an existing folder to create the "
                                "project inside.")
            return False
        loc = Path(loc_text)
        if (loc / name).exists() and any((loc / name).iterdir()):
            QMessageBox.warning(self, "Location",
                                f"'{name}' already exists there and is not "
                                "empty.")
            return False
        return True

    # ── Step 2: Videos ───────────────────────────────────────────────────────

    def _build_videos_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        title = QLabel("Add your videos")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)
        hint = QLabel(
            "Each video becomes one run in the project. You can rename runs "
            "here -- the name labels the video's outputs.")
        hint.setStyleSheet("color: #888;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        btn_row = QHBoxLayout()
        add_files = QPushButton("Add Videos...")
        add_files.clicked.connect(self._add_videos)
        add_dir = QPushButton("Add Folder...")
        add_dir.clicked.connect(self._add_folder)
        rm = QPushButton("Remove selected")
        rm.clicked.connect(self._remove_videos)
        btn_row.addWidget(add_files)
        btn_row.addWidget(add_dir)
        btn_row.addWidget(rm)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self._video_table = QTableWidget(0, 3)
        self._video_table.setHorizontalHeaderLabels(
            ["Run name", "Video file", "Size"])
        self._video_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._video_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        lay.addWidget(self._video_table, 1)

        self._copy_radio = QRadioButton(
            "Copy videos into the project (originals untouched)")
        self._move_radio = QRadioButton(
            "Move videos into the project (frees the original location)")
        self._copy_radio.setChecked(True)
        lay.addWidget(self._copy_radio)
        lay.addWidget(self._move_radio)
        return page

    def _add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add videos", "",
            "Video (*.mp4 *.mov *.avi *.mkv);;All (*)")
        self._append_videos(paths)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add a video folder")
        if folder:
            paths = sorted(
                str(p) for p in Path(folder).iterdir()
                if p.suffix.lower() in _VIDEO_EXTS and p.is_file())
            self._append_videos(paths)

    def _append_videos(self, paths):
        from mindsight.project.staging import _sanitize_run_id
        existing = {v["source"] for v in self._videos}
        for p in paths or []:
            src = Path(p)
            if not src.is_file() or src in existing:
                continue
            self._videos.append({
                "source": src,
                "run_id": _sanitize_run_id(src.stem),
                "meta": {},
                "pixmap": None,
            })
        self._refresh_video_table()

    def _remove_videos(self):
        rows = sorted({i.row() for i in self._video_table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            del self._videos[r]
        self._tag_index = 0
        self._refresh_video_table()

    def _refresh_video_table(self):
        self._video_table.blockSignals(True)
        self._video_table.setRowCount(len(self._videos))
        for r, v in enumerate(self._videos):
            name_item = QTableWidgetItem(v["run_id"])
            self._video_table.setItem(r, 0, name_item)
            src_item = QTableWidgetItem(str(v["source"]))
            src_item.setFlags(src_item.flags()
                              & ~Qt.ItemFlag.ItemIsEditable)
            self._video_table.setItem(r, 1, src_item)
            try:
                size = v["source"].stat().st_size / 1e6
                size_text = f"{size:,.0f} MB"
            except OSError:
                size_text = "?"
            size_item = QTableWidgetItem(size_text)
            size_item.setFlags(size_item.flags()
                               & ~Qt.ItemFlag.ItemIsEditable)
            self._video_table.setItem(r, 2, size_item)
        self._video_table.blockSignals(False)

    def _validate_videos(self) -> bool:
        from mindsight.project.staging import _sanitize_run_id
        if not self._videos:
            QMessageBox.warning(self, "Videos", "Add at least one video.")
            return False
        # Pull (sanitized) edited run names back from the table.
        seen = set()
        for r, v in enumerate(self._videos):
            item = self._video_table.item(r, 0)
            raw = (item.text().strip() if item else "") or v["source"].stem
            run_id = _sanitize_run_id(raw)
            if run_id.lower() in seen:
                QMessageBox.warning(
                    self, "Run names",
                    f"Two runs would be named '{run_id}' -- run names must "
                    "be unique.")
                return False
            seen.add(run_id.lower())
            v["run_id"] = run_id
        self._refresh_video_table()
        return True

    # ── Step 3: Tag each video ───────────────────────────────────────────────

    def _build_tag_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        head = QHBoxLayout()
        title = QLabel("Tag each video")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        head.addWidget(title)
        head.addStretch(1)
        self._tag_progress = QLabel("")
        head.addWidget(self._tag_progress)
        self._tag_jump = QComboBox()
        self._tag_jump.setMinimumWidth(140)
        self._tag_jump.activated.connect(self._jump_to_video)
        head.addWidget(self._tag_jump)
        lay.addLayout(head)

        self._tag_frame = QLabel()
        self._tag_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tag_frame.setMinimumHeight(220)
        self._tag_frame.setStyleSheet("background: #1a1a2e; color: #556;")
        lay.addWidget(self._tag_frame, 1)

        guide = QLabel(
            "MindSight numbers people by where they appear on screen, left "
            "to right. Type each person's ID under their position.")
        guide.setStyleSheet("color: #888;")
        guide.setWordWrap(True)
        lay.addWidget(guide)

        # Participant fields + conditions + manifest fields live in a scroll
        # area so many-participant studies don't squash the frame.
        form_host = QWidget()
        self._tag_form = QFormLayout(form_host)
        self._tag_form.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_host)
        scroll.setMaximumHeight(190)
        lay.addWidget(scroll)

        nav = QHBoxLayout()
        self._tag_file_label = QLabel("")
        self._tag_file_label.setStyleSheet("color: #888;")
        nav.addWidget(self._tag_file_label)
        nav.addStretch(1)
        prev_b = QPushButton("‹ Prev video")
        prev_b.clicked.connect(lambda: self._step_video(-1))
        next_b = QPushButton("Next video ›")
        next_b.clicked.connect(lambda: self._step_video(+1))
        nav.addWidget(prev_b)
        nav.addWidget(next_b)
        lay.addLayout(nav)

        self._participant_edits: list[QLineEdit] = []
        self._condition_checks: list = []
        self._cond_free = None
        self._date_edit = self._session_edit = self._run_notes_edit = None
        return page

    def _rebuild_tag_form(self):
        while self._tag_form.rowCount():
            self._tag_form.removeRow(0)
        self._participant_edits = []
        self._condition_checks = []
        n = self._people.value()
        for i in range(n):
            edit = QLineEdit()
            edit.setPlaceholderText("e.g. S70")
            self._tag_form.addRow(_ordinal_label(i, n) + ":", edit)
            self._participant_edits.append(edit)
        conds = self.conditions()
        if conds:
            from PyQt6.QtWidgets import QCheckBox
            row = QHBoxLayout()
            for c in conds:
                cb = QCheckBox(c)
                self._condition_checks.append(cb)
                row.addWidget(cb)
            row.addStretch(1)
            host = QWidget()
            host.setLayout(row)
            self._tag_form.addRow("Conditions:", host)
            self._cond_free = None
        else:
            self._cond_free = QLineEdit()
            self._cond_free.setPlaceholderText(
                "optional -- separate multiple with |")
            self._tag_form.addRow("Conditions:", self._cond_free)
        self._date_edit = QLineEdit()
        self._session_edit = QLineEdit()
        self._run_notes_edit = QLineEdit()
        self._tag_form.addRow("Date:", self._date_edit)
        self._tag_form.addRow("Session:", self._session_edit)
        self._tag_form.addRow("Notes:", self._run_notes_edit)

    def _show_tag_video(self, index: int):
        if not self._videos:
            self._tag_progress.setText("No videos added yet")
            return
        self._save_tag_fields()      # persist edits from the previous video
        self._tag_index = max(0, min(index, len(self._videos) - 1))
        v = self._videos[self._tag_index]
        self._rebuild_tag_form()
        self._tag_progress.setText(
            f"Video {self._tag_index + 1} of {len(self._videos)}")
        self._tag_jump.blockSignals(True)
        self._tag_jump.clear()
        self._tag_jump.addItems([x["run_id"] for x in self._videos])
        self._tag_jump.setCurrentIndex(self._tag_index)
        self._tag_jump.blockSignals(False)
        self._tag_file_label.setText(v["source"].name)
        if v["pixmap"] is None:
            v["pixmap"] = middle_frame_pixmap(v["source"]) or False
        if v["pixmap"]:
            self._tag_frame.setPixmap(v["pixmap"])
        else:
            self._tag_frame.setText(
                "Preview unavailable -- you can still tag this video.")
        # Restore any previously entered meta.
        meta = v["meta"]
        for i, edit in enumerate(self._participant_edits):
            edit.setText((meta.get("participants") or {}).get(i, ""))
        tags = meta.get("conditions") or []
        for cb in self._condition_checks:
            cb.setChecked(cb.text() in tags)
        if self._cond_free is not None:
            self._cond_free.setText("|".join(tags))
        default_date = ""
        try:
            default_date = datetime.fromtimestamp(
                v["source"].stat().st_mtime).strftime("%Y-%m-%d")
        except OSError:
            pass
        self._date_edit.setText(meta.get("date", default_date))
        self._session_edit.setText(str(meta.get("session", "") or ""))
        self._run_notes_edit.setText(meta.get("notes", "") or "")

    def _save_tag_fields(self):
        if not self._videos or not self._participant_edits:
            return
        v = self._videos[self._tag_index]
        participants = {i: e.text().strip()
                        for i, e in enumerate(self._participant_edits)
                        if e.text().strip()}
        if self._condition_checks:
            tags = [cb.text() for cb in self._condition_checks
                    if cb.isChecked()]
        elif self._cond_free is not None:
            tags = [t.strip() for t in self._cond_free.text().split("|")
                    if t.strip()]
        else:
            tags = []
        meta = {}
        if participants:
            meta["participants"] = participants
        if tags:
            meta["conditions"] = tags
        if self._date_edit and self._date_edit.text().strip():
            meta["date"] = self._date_edit.text().strip()
        if self._session_edit and self._session_edit.text().strip():
            meta["session"] = self._session_edit.text().strip()
        if self._run_notes_edit and self._run_notes_edit.text().strip():
            meta["notes"] = self._run_notes_edit.text().strip()
        v["meta"] = meta

    def _step_video(self, delta: int):
        self._show_tag_video(self._tag_index + delta)

    def _jump_to_video(self, row: int):
        self._show_tag_video(row)

    # ── Step 4: Pipeline ─────────────────────────────────────────────────────

    def _build_pipeline_page(self):
        from mindsight.config_compat import known_good_preset_path
        page = QWidget()
        lay = QVBoxLayout(page)
        title = QLabel("Choose the analysis settings")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)
        hint = QLabel(
            "The project gets its own copy of these settings "
            "(Pipeline/pipeline.yaml). You can change them any time from "
            "Analyze Footage.")
        hint.setStyleSheet("color: #888;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self._pipe_kg = QRadioButton(
            "KG_Standard -- the validated preset (recommended)")
        self._pipe_kg.setChecked(True)
        lay.addWidget(self._pipe_kg)
        if known_good_preset_path() is None:
            self._pipe_kg.setEnabled(False)
            self._pipe_kg.setToolTip("preset file not found in this install")

        self._pipe_lp = QRadioButton(
            "Low Power -- faster on weak hardware (UNVALIDATED for research "
            "conclusions)")
        lay.addWidget(self._pipe_lp)
        if low_power_preset_path() is None:
            self._pipe_lp.setEnabled(False)
            self._pipe_lp.setToolTip("preset file not found in this install")

        self._pipe_current = QRadioButton(
            "Current Inference Settings -- copy what Analyze Footage uses "
            "right now")
        lay.addWidget(self._pipe_current)
        if self._settings is None:
            self._pipe_current.setEnabled(False)

        imp_row = QHBoxLayout()
        self._pipe_import = QRadioButton("A pipeline YAML file:")
        imp_row.addWidget(self._pipe_import)
        self._import_label = QLabel("none chosen")
        self._import_label.setStyleSheet("color: #888;")
        imp_row.addWidget(self._import_label, 1)
        imp_btn = QPushButton("Choose...")
        imp_btn.clicked.connect(self._choose_import_yaml)
        imp_row.addWidget(imp_btn)
        lay.addLayout(imp_row)
        lay.addStretch(1)
        return page

    def _choose_import_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a pipeline YAML", "", "YAML (*.yaml *.yml)")
        if path:
            self._import_yaml = Path(path)
            self._import_label.setText(path)
            self._pipe_import.setChecked(True)

    def _validate_pipeline(self) -> bool:
        if self._pipe_import.isChecked() and (
                self._import_yaml is None
                or not self._import_yaml.is_file()):
            QMessageBox.warning(self, "Pipeline",
                                "Choose the YAML file to import.")
            return False
        return True

    def _pipeline_choice_text(self) -> str:
        if self._pipe_kg.isChecked():
            return "KG_Standard preset"
        if self._pipe_lp.isChecked():
            return "Low Power preset (UNVALIDATED)"
        if self._pipe_current.isChecked():
            return "current Inference Settings"
        return f"imported: {self._import_yaml}"

    def _write_pipeline(self, project: Path):
        from mindsight.config_compat import known_good_preset_path
        target = project / "Pipeline" / "pipeline.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._pipe_kg.isChecked():
            shutil.copy2(known_good_preset_path(), target)
        elif self._pipe_lp.isChecked():
            shutil.copy2(low_power_preset_path(), target)
        elif self._pipe_current.isChecked():
            from .pipeline_dialog import _namespace_to_yaml_dict
            cfg = _namespace_to_yaml_dict(self._settings.ns(), full=True)
            target.write_text(yaml.dump(cfg, default_flow_style=False,
                                        sort_keys=False))
        else:
            shutil.copy2(self._import_yaml, target)

    # ── Step 5: Review & create ──────────────────────────────────────────────

    def _build_review_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        title = QLabel("Review")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)
        self._review = QLabel("")
        self._review.setWordWrap(True)
        self._review.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self._review)
        lay.addStretch(1)
        return page

    def _refresh_review(self):
        tagged = sum(1 for v in self._videos
                     if v["meta"].get("participants"))
        mode = "copied" if self._copy_radio.isChecked() else "moved"
        conds = ", ".join(self.conditions()) or "none defined"
        lines = [
            f"<b>{self._name.text().strip()}</b> in "
            f"{self._location.text().strip()}",
            f"{len(self._videos)} video(s), {mode} into the project "
            f"({tagged} tagged, {len(self._videos) - tagged} to tag later)",
            f"Conditions: {conds}",
            f"Analysis settings: {self._pipeline_choice_text()}",
        ]
        if self._notes.toPlainText().strip():
            lines.append("Study notes: saved to notes.md")
        self._review.setText("<br>".join(lines))

    def _create(self):
        from mindsight.project.runner import create_project
        from mindsight.project.staging import stage_run
        name = self._name.text().strip()
        location = Path(self._location.text().strip())
        mode = "copy" if self._copy_radio.isChecked() else "move"
        try:
            project = create_project(location, name)
        except ValueError as exc:
            QMessageBox.warning(self, "Create project", str(exc))
            return
        self._write_pipeline(project)
        notes = self._notes.toPlainText().strip()
        if notes:
            (project / "notes.md").write_text(notes + "\n")

        progress = QProgressDialog(
            "Adding videos...", None, 0, len(self._videos), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        staged, failed = [], []
        for i, v in enumerate(self._videos):
            progress.setValue(i)
            progress.setLabelText(
                f"Adding {v['source'].name} ({i + 1}/{len(self._videos)})")
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            try:
                stage_run(project, v["source"], v["meta"] or None,
                          run_id=v["run_id"], mode=mode)
                staged.append(v["run_id"])
            except (ValueError, OSError) as exc:
                failed.append(f"{v['run_id']}: {exc}")
        progress.setValue(len(self._videos))

        if failed:
            QMessageBox.warning(
                self, "Some videos could not be added",
                f"Added {len(staged)} of {len(self._videos)} videos.\n"
                "Problems:\n" + "\n".join(failed)
                + "\n\nThe project was still created -- you can add the "
                "remaining videos from Analyze Footage.")
        self.created_path = project
        self.accept()
