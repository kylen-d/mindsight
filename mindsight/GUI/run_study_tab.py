"""
run_study_tab.py
----------------
The **Analyze Footage** tab (SP3.1 Batch G, Step 15) -- the RA home screen.

It CONSUMES the project layer, it never computes (D11): open a project via
:class:`~mindsight.project.project.Project`, render its
:class:`~mindsight.project.preflight.PreflightReport` as a checklist, list its
staged runs (:class:`~mindsight.project.staging.RunSpec`) with ledger status
(``Project.status()``) and a resume-plan preview (``Project.decide`` via
``Project.decisions``), and drive the batch through the SAME
:class:`~mindsight.GUI.workers.ProjectWorker` the whole GUI shares.

Resume UX (Q6): resume is ON by default with NO checkbox -- the runs table
previews skip / process / re-run+archive; "Re-run all" (confirm) maps to
``resume=False``; a per-row right-click "Re-run this run" invalidates that run's
ledger record (``Project.invalidate``).

Study-setup panels (participants / conditions / pipeline / output / Save
project.yaml) relocate here from the retired Project Mode tab into a collapsible
"Study setup" area, reusing the participants / conditions section widgets as-is.

The manual path (Q7): "Add single run..." picks a video + participants /
conditions / date, then either runs it now (single-source) or saves it into the
project as a staged run folder.

Every user-visible string says "Analyze Footage" (Q9 amendment); the internal
identifiers keep the ``run_study`` naming.
"""

from __future__ import annotations

import queue
from argparse import Namespace
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .conditions_section import ConditionsSection
from .participants_section import ParticipantsSection
from .widgets import CollapsibleGroupBox, _bgr_to_pixmap

# Severity presentation for the preflight checklist.
_SEV_ICON = {"ok": "✓", "warn": "▲", "fail": "✗"}
_SEV_COLOUR = {"ok": "#2a7a2a", "warn": "#b8860b", "fail": "#b22222"}


# ══════════════════════════════════════════════════════════════════════════════
# Preflight checklist widget
# ══════════════════════════════════════════════════════════════════════════════

class PreflightChecklist(QWidget):
    """Renders a :class:`PreflightReport` as icon + message + fix-hint rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(4, 4, 4, 4)
        self._lay.setSpacing(2)
        self._placeholder = QLabel("Open a project to run preflight.")
        self._placeholder.setStyleSheet("color: #888; font-style: italic;")
        self._lay.addWidget(self._placeholder)

    def clear(self):
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def render(self, report):
        """Render a PreflightReport (or clear to a placeholder when None)."""
        self.clear()
        if report is None:
            ph = QLabel("Open a project to run preflight.")
            ph.setStyleSheet("color: #888; font-style: italic;")
            self._lay.addWidget(ph)
            return
        for check in report.checks:
            icon = _SEV_ICON.get(check.severity, "?")
            colour = _SEV_COLOUR.get(check.severity, "#888")
            row = QLabel(f"{icon}  <b>{check.label}:</b> {check.message}")
            row.setTextFormat(Qt.TextFormat.RichText)
            row.setWordWrap(True)
            row.setStyleSheet(f"color: {colour};")
            self._lay.addWidget(row)
            if check.fix_hint and check.severity != "ok":
                hint = QLabel(f"↳ {check.fix_hint}")
                hint.setWordWrap(True)
                hint.setStyleSheet("color: #999; margin-left: 18px;")
                self._lay.addWidget(hint)
        summary = QLabel(
            f"{'PASSED' if report.ok else 'FAILED'} "
            f"({report.n_fail} failure(s), {report.n_warn} warning(s))")
        summary.setStyleSheet(
            f"font-weight: bold; color: "
            f"{'#2a7a2a' if report.ok else '#b22222'};")
        self._lay.addWidget(summary)


# ══════════════════════════════════════════════════════════════════════════════
# Manual "Add single run" dialog (Q7)
# ══════════════════════════════════════════════════════════════════════════════

class ManualRunDialog(QDialog):
    """Pick a video + participants / conditions / date, then Run now or Save.

    Exposes ``result_action`` ("run" / "save" / None), the chosen ``video`` path,
    the ``meta`` dict (Q2 keys), the ``move`` flag, and (for run-now) the chosen
    ``output_dir``.  Building the RunSpec is the caller's job (staging helpers).
    """

    def __init__(self, project_path: Path | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add single run")
        self._project_path = project_path
        self.result_action: str | None = None
        self.video: str | None = None
        self.meta: dict = {}
        self.move: bool = False
        self.output_dir: str | None = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        form = QFormLayout()

        vid_row = QHBoxLayout()
        self._video = QLineEdit()
        self._video.setPlaceholderText("Select a video or image...")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_video)
        vid_row.addWidget(self._video, 1)
        vid_row.addWidget(browse)
        form.addRow("Video:", vid_row)

        self._participants = QLineEdit()
        self._participants.setPlaceholderText("0:S70, 1:S71  (track:label)")
        form.addRow("Participants:", self._participants)

        csv_row = QHBoxLayout()
        self._csv_note = QLabel("")
        self._csv_note.setStyleSheet("color: #888; font-size: 11px;")
        csv_btn = QPushButton("Import CSV...")
        csv_btn.setToolTip("Load a participant_ids.csv row as the participant map")
        csv_btn.clicked.connect(self._import_csv)
        csv_row.addWidget(self._csv_note, 1)
        csv_row.addWidget(csv_btn)
        form.addRow("", csv_row)

        self._conditions = QLineEdit()
        self._conditions.setPlaceholderText("collab, kitchenA")
        form.addRow("Conditions:", self._conditions)

        self._date = QLineEdit()
        self._date.setPlaceholderText("2026-07-02 (optional)")
        form.addRow("Date:", self._date)

        self._session = QLineEdit()
        self._session.setPlaceholderText("dyad-07 (optional)")
        form.addRow("Session:", self._session)

        self._notes = QLineEdit()
        self._notes.setPlaceholderText("free-form notes (optional)")
        form.addRow("Notes:", self._notes)

        lay.addLayout(form)

        from PyQt6.QtWidgets import QCheckBox
        self._move = QCheckBox("Move original into the project (default: copy)")
        lay.addWidget(self._move)

        btns = QDialogButtonBox()
        self._run_btn = btns.addButton("Run now",
                                       QDialogButtonBox.ButtonRole.AcceptRole)
        self._save_btn = btns.addButton("Save to project...",
                                        QDialogButtonBox.ButtonRole.ActionRole)
        btns.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._run_btn.clicked.connect(lambda: self._finish("run"))
        self._save_btn.clicked.connect(lambda: self._finish("save"))
        btns.rejected.connect(self.reject)
        self._save_btn.setEnabled(self._project_path is not None)
        lay.addWidget(btns)

    def _browse_video(self):
        start = str(self._project_path) if self._project_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video or image", start,
            "Media (*.mp4 *.mov *.avi *.mkv *.jpg *.jpeg *.png);;All (*)")
        if path:
            self._video.setText(path)

    def _import_csv(self):
        start = str(self._project_path) if self._project_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import participant_ids.csv", start, "CSV (*.csv);;All (*)")
        if not path:
            return
        try:
            from mindsight.participant_ids import load_participant_csv
            pid_maps = load_participant_csv(Path(path))
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.warning(self, "CSV import", f"Could not read CSV:\n{exc}")
            return
        # Use the first mapping (single-run staging is one video).
        if pid_maps:
            first = next(iter(pid_maps.values()))
            self._participants.setText(
                ", ".join(f"{k}:{v}" for k, v in first.items()))
            self._csv_note.setText(f"Loaded {len(first)} participant(s) from CSV")

    def _parse_participants(self) -> dict | None:
        text = self._participants.text().strip()
        if not text:
            return None
        out: dict = {}
        for pair in text.replace(";", ",").split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                raise ValueError(
                    f"participant '{pair}' must be track:label (e.g. 0:S70)")
            k, v = pair.split(":", 1)
            out[int(k.strip())] = v.strip()
        return out or None

    def _collect_meta(self) -> dict:
        meta: dict = {}
        pid = self._parse_participants()
        if pid is not None:
            meta["participants"] = pid
        conds = [c.strip() for c in self._conditions.text().split(",") if c.strip()]
        if conds:
            meta["conditions"] = conds
        for key, widget in (("date", self._date), ("session", self._session),
                            ("notes", self._notes)):
            val = widget.text().strip()
            if val:
                meta[key] = val
        return meta

    def _finish(self, action: str):
        video = self._video.text().strip()
        if not video or not Path(video).is_file():
            QMessageBox.warning(self, "Add single run",
                                "Choose an existing video first.")
            return
        try:
            meta = self._collect_meta()
        except ValueError as exc:
            QMessageBox.warning(self, "Add single run", str(exc))
            return
        if action == "run":
            out = QFileDialog.getExistingDirectory(self, "Output directory")
            if not out:
                return
            self.output_dir = out
        self.result_action = action
        self.video = video
        self.meta = meta
        self.move = self._move.isChecked()
        self.accept()


# ══════════════════════════════════════════════════════════════════════════════
# Run Study tab
# ══════════════════════════════════════════════════════════════════════════════

_RUN_COLS = ["Run", "Source", "Participants", "Conditions", "Status",
             "Plan", "Progress", "Error"]


class RunStudyTab(QWidget):
    """The Analyze Footage home: open project -> preflight -> run batch."""

    def __init__(self, gaze_tab=None, parent=None):
        super().__init__(parent)
        self._gaze_tab = gaze_tab
        self._project = None
        self._project_path: Path | None = None
        self._resume = True                 # Q6: resume ON by default
        self._dirty = False
        self._run_rows: dict[str, int] = {}  # run_id -> table row

        self._worker = None
        self._one_off_worker = None
        self._stop_requested = False
        self._progress_q: queue.Queue = queue.Queue()
        self._log_q: queue.Queue = queue.Queue()
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll)

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Project picker row + Recent dropdown (D12)
        top = QHBoxLayout()
        top.addWidget(QLabel("Project:"))
        self._project_dir = QLineEdit()
        self._project_dir.setPlaceholderText("Open a MindSight project folder...")
        self._project_dir.setReadOnly(True)
        top.addWidget(self._project_dir, 1)
        open_btn = QPushButton("Open...")
        open_btn.clicked.connect(self._open_project_dialog)
        top.addWidget(open_btn)
        self._recent = QComboBox()
        self._recent.setMinimumWidth(180)
        self._recent.setToolTip("Recently opened projects")
        self._recent.activated.connect(self._open_recent)
        top.addWidget(self._recent)
        outer.addLayout(top)
        self._refresh_recent_dropdown()

        self._status_label = QLabel("No project open.")
        self._status_label.setStyleSheet("color: #888; font-style: italic;")
        outer.addWidget(self._status_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: preflight + runs table + resume controls + manual
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)

        pf_grp = QGroupBox("Preflight")
        pf_lay = QVBoxLayout(pf_grp)
        self._checklist = PreflightChecklist()
        pf_lay.addWidget(self._checklist)
        rerun_pf = QPushButton("Re-run preflight")
        rerun_pf.clicked.connect(self._run_preflight)
        pf_lay.addWidget(rerun_pf)
        left_lay.addWidget(pf_grp)

        runs_grp = QGroupBox("Runs")
        runs_lay = QVBoxLayout(runs_grp)
        self._runs_table = QTableWidget(0, len(_RUN_COLS))
        self._runs_table.setHorizontalHeaderLabels(_RUN_COLS)
        self._runs_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._runs_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._runs_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._runs_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._runs_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._runs_table.customContextMenuRequested.connect(self._run_context_menu)
        runs_lay.addWidget(self._runs_table)

        ctl_row = QHBoxLayout()
        self._rerun_all_btn = QPushButton("Re-run all")
        self._rerun_all_btn.setToolTip(
            "Reprocess every run, ignoring the resume ledger (resume=off)")
        self._rerun_all_btn.clicked.connect(self._toggle_rerun_all)
        add_run_btn = QPushButton("Add single run...")
        add_run_btn.clicked.connect(self._add_single_run)
        ctl_row.addWidget(self._rerun_all_btn)
        ctl_row.addWidget(add_run_btn)
        ctl_row.addStretch(1)
        runs_lay.addLayout(ctl_row)
        left_lay.addWidget(runs_grp, 1)

        # Study setup (collapsible) -- relocated from the retired Project Mode tab
        self._build_study_setup(left_lay)

        splitter.addWidget(left)

        # RIGHT: preview + log
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(6)
        self._preview = QLabel()
        self._preview.setStyleSheet("background: #1a1a2e;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(240, 160)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_lay.addWidget(self._preview, 2)

        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(60)
        self._log_box.setFont(QFont("Courier", 10))
        log_lay.addWidget(self._log_box)
        right_lay.addWidget(log_grp, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        # Status-bar buttons (injected by main_window, T11); provide fallbacks so
        # the tab is testable standalone.
        self._run_btn = QPushButton("▶  Run")
        self._run_btn.clicked.connect(self._start)
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)

    def _build_study_setup(self, parent_lay):
        grp = CollapsibleGroupBox("Study setup")
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)

        # Pipeline picker (D12)
        pipe_row = QHBoxLayout()
        pipe_row.addWidget(QLabel("Pipeline:"))
        self._pipeline_path = QLineEdit()
        self._pipeline_path.setPlaceholderText("Pipeline/pipeline.yaml")
        pipe_browse = QPushButton("Browse...")
        pipe_browse.clicked.connect(self._browse_pipeline)
        pipe_row.addWidget(self._pipeline_path, 1)
        pipe_row.addWidget(pipe_browse)
        lay.addLayout(pipe_row)

        import_gaze = QPushButton("Import from Gaze Tuning")
        import_gaze.setToolTip(
            "Write the current Gaze Tuning settings as this project's "
            "pipeline.yaml")
        import_gaze.clicked.connect(self._import_from_gaze)
        lay.addWidget(import_gaze)

        # Sources table used by the conditions section
        self._source_table = QTableWidget(0, 2)
        self._source_table.setHorizontalHeaderLabels(["Filename", "Conditions"])
        self._source_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._source_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._source_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._source_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)

        part_grp = CollapsibleGroupBox("Participants")
        self._participants = ParticipantsSection()
        self._participants.set_dirty_callback(self._mark_dirty)
        part_grp.set_content(self._participants)
        lay.addWidget(part_grp)

        cond_grp = CollapsibleGroupBox("Conditions")
        self._conditions = ConditionsSection(source_table=self._source_table)
        self._conditions.set_dirty_callback(self._mark_dirty)
        cond_grp.set_content(self._conditions)
        lay.addWidget(cond_grp)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output root:"))
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Default: project/Outputs")
        self._output_dir.textChanged.connect(self._mark_dirty)
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self._output_dir, 1)
        out_row.addWidget(out_browse)
        lay.addLayout(out_row)

        self._save_btn = QPushButton("Save project.yaml")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_project_yaml)
        lay.addWidget(self._save_btn)

        grp.set_content(inner)
        parent_lay.addWidget(grp)

    # ── Project open / recent ────────────────────────────────────────────────

    def _open_project_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open MindSight project folder")
        if path:
            self._open_project(path)

    def _open_recent(self, index: int):
        path = self._recent.itemData(index)
        if path:
            self._open_project(str(path))

    def _refresh_recent_dropdown(self):
        from .settings_manager import SettingsManager
        self._recent.blockSignals(True)
        self._recent.clear()
        self._recent.addItem("Recent projects...", None)
        try:
            for p in SettingsManager().list_recent_projects():
                self._recent.addItem(Path(p).name, p)
        except Exception:
            pass
        self._recent.setCurrentIndex(0)
        self._recent.blockSignals(False)

    def _open_project(self, path_str: str):
        from mindsight.project.project import Project
        try:
            project = Project.open(path_str)
        except (FileNotFoundError, ValueError) as exc:
            self._status_label.setText(f"Invalid project: {exc}")
            self._status_label.setStyleSheet("color: #b22222; font-weight: bold;")
            return
        self._project = project
        self._project_path = project.path
        self._project_dir.setText(str(project.path))
        self._dirty = False
        self._save_btn.setEnabled(True)

        try:
            from .settings_manager import SettingsManager
            SettingsManager().add_recent_project(str(project.path))
        except Exception:
            pass
        self._refresh_recent_dropdown()

        # Load the project pipeline into Gaze Tuning so runs reproduce project
        # numbers (Batch B T7 finding: the GUI namespace must carry the pipeline
        # YAML values, not bare widget defaults).
        self._load_project_pipeline_into_gaze()
        self._populate_study_setup()
        self._run_preflight()
        self._refresh_runs_table()

        self._status_label.setText(f"Open: {project.path.name}")
        self._status_label.setStyleSheet("color: #2a7a2a; font-weight: bold;")

    def _load_project_pipeline_into_gaze(self):
        if not self._gaze_tab or not self._project:
            return
        cfg = self._project.config
        if cfg and cfg.pipeline_path:
            pipe = self._project_path / cfg.pipeline_path
        else:
            pipe = self._project_path / "Pipeline" / "pipeline.yaml"
        if not pipe.is_file():
            return
        try:
            from mindsight.config_compat import load_pipeline
            ns = load_pipeline(str(pipe), Namespace())
            self._gaze_tab.apply_namespace(ns)
        except Exception as exc:  # pragma: no cover - GUI error path
            self._append_log(f"[WARN] could not load pipeline: {exc}")

    def _populate_study_setup(self):
        if not self._project:
            return
        cfg = self._project.config
        from mindsight.project.runner import discover_sources
        sources = discover_sources(self._project_path)
        self._source_table.setRowCount(len(sources))
        for i, src in enumerate(sources):
            self._source_table.setItem(i, 0, QTableWidgetItem(src.name))
            tags = ""
            if cfg and src.name in (cfg.conditions or {}):
                tags = " | ".join(cfg.conditions[src.name])
            self._source_table.setItem(i, 1, QTableWidgetItem(tags))
        if cfg and cfg.pipeline_path:
            self._pipeline_path.setText(cfg.pipeline_path)
        elif (self._project_path / "Pipeline" / "pipeline.yaml").exists():
            self._pipeline_path.setText("Pipeline/pipeline.yaml")
        self._participants.set_sources(sources)
        self._participants.populate(cfg, sources, project_path=self._project_path)
        self._conditions.populate(cfg, sources)
        if cfg and cfg.output and cfg.output.directory:
            self._output_dir.setText(cfg.output.directory)
        self._dirty = False

    # ── Preflight / runs table ───────────────────────────────────────────────

    def _current_ns(self) -> Namespace:
        if self._gaze_tab:
            return self._gaze_tab._build_namespace()
        return Namespace()

    def _run_preflight(self):
        if not self._project:
            self._checklist.render(None)
            return
        try:
            report = self._project.preflight(ns=self._current_ns())
        except Exception as exc:  # pragma: no cover - preflight never raises
            self._append_log(f"[WARN] preflight failed: {exc}")
            return
        self._checklist.render(report)

    def _refresh_runs_table(self):
        if not self._project:
            return
        try:
            specs = self._project.runs()
            statuses = {s.run_id: s for s in self._project.status()}
            plan = self._project.decisions(
                self._current_ns(), resume=self._resume)
        except Exception as exc:
            self._append_log(f"[WARN] could not list runs: {exc}")
            return
        self._run_rows = {}
        self._runs_table.setRowCount(len(specs))
        for i, spec in enumerate(specs):
            self._run_rows[spec.run_id] = i
            st = statuses.get(spec.run_id)
            pid = (", ".join(f"{k}:{v}" for k, v in spec.pid_map.items())
                   if spec.pid_map else "—")
            conds = spec.conditions.replace("|", ", ") if spec.conditions else "—"
            self._set_cell(i, 0, spec.run_id)
            self._set_cell(i, 1, Path(spec.source).name)
            self._set_cell(i, 2, pid)
            self._set_cell(i, 3, conds)
            self._set_cell(i, 4, (st.status if st and st.status else "—"))
            self._set_cell(i, 5, self._plan_text(plan.get(spec.run_id)))
            self._set_cell(i, 6, "")
            self._set_cell(i, 7, (st.error if st and st.error else ""))

    @staticmethod
    def _plan_text(decision) -> str:
        return {
            "skip": "done → will skip",
            "redo": "will process",
            "redo_archive": "changed → re-run + archive",
        }.get(decision, "will process")

    def _set_cell(self, row, col, text):
        self._runs_table.setItem(row, col, QTableWidgetItem(str(text)))

    def _row_status(self, run_id: str, text: str, colour: str | None = None):
        row = self._run_rows.get(run_id)
        if row is None:
            return
        item = QTableWidgetItem(text)
        if colour:
            item.setForeground(QColor(colour))
        self._runs_table.setItem(row, 4, item)

    # ── Resume controls (Q6) ─────────────────────────────────────────────────

    def _toggle_rerun_all(self):
        if self._resume:
            reply = QMessageBox.question(
                self, "Re-run all",
                "Reprocess EVERY run, ignoring the resume ledger?\n"
                "(nothing is archived; existing outputs are overwritten)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._resume = False
            self._rerun_all_btn.setText("Re-run all ✓ (resume off)")
        else:
            self._resume = True
            self._rerun_all_btn.setText("Re-run all")
        self._refresh_runs_table()

    def _run_context_menu(self, pos):
        if not self._project:
            return
        row = self._runs_table.rowAt(pos.y())
        if row < 0:
            return
        run_id_item = self._runs_table.item(row, 0)
        if run_id_item is None:
            return
        run_id = run_id_item.text()
        menu = QMenu(self)
        act = menu.addAction("Re-run this run")
        chosen = menu.exec(self._runs_table.viewport().mapToGlobal(pos))
        if chosen == act:
            self._project.invalidate(run_id)
            self._append_log(f"Marked '{run_id}' for re-run.")
            self._refresh_runs_table()

    # ── Study setup helpers ──────────────────────────────────────────────────

    def _browse_pipeline(self):
        start = str(self._project_path / "Pipeline") if self._project_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pipeline config", start,
            "Pipeline config (*.yaml *.yml *.json)")
        if not path:
            return
        if self._project_path:
            try:
                path = str(Path(path).relative_to(self._project_path))
            except ValueError:
                pass
        self._pipeline_path.setText(path)
        self._mark_dirty()

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self._output_dir.setText(path)

    def _import_from_gaze(self):
        if not self._gaze_tab or not self._project_path:
            QMessageBox.warning(self, "Import", "Open a project first.")
            return
        ns = self._gaze_tab._build_namespace()
        from .pipeline_dialog import _namespace_to_yaml_dict
        yaml_dict = _namespace_to_yaml_dict(ns)
        pipe_text = self._pipeline_path.text().strip() or "Pipeline/pipeline.yaml"
        target = Path(pipe_text)
        if not target.is_absolute():
            target = self._project_path / target
        if target.exists():
            reply = QMessageBox.question(
                self, "Overwrite pipeline?",
                f"{target.name} exists. Overwrite with Gaze Tuning settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.dump(yaml_dict, default_flow_style=False,
                                    sort_keys=False))
        self._pipeline_path.setText(pipe_text)
        self._append_log(f"Wrote {target.name}")

    def _build_project_config(self):
        from mindsight.pipeline_config import ProjectConfig, ProjectOutputConfig
        return ProjectConfig(
            pipeline_path=self._pipeline_path.text().strip() or None,
            conditions=self._conditions.get_conditions(),
            participants=self._participants.get_participants(),
            output=ProjectOutputConfig(
                directory=self._output_dir.text().strip() or None),
        )

    def _save_project_yaml(self):
        if not self._project_path:
            return
        from mindsight.project.runner import save_project_config
        cfg = self._build_project_config()
        save_project_config(self._project_path, cfg)
        self._dirty = False
        self._save_btn.setText("Save project.yaml")
        self._append_log("Saved project.yaml")
        # Reopen so the facade + runs table pick up the edits.
        self._open_project(str(self._project_path))

    def _mark_dirty(self, *_):
        if not self._dirty:
            self._dirty = True
            self._save_btn.setText("Save project.yaml *")

    # ── Manual single run (Q7) ───────────────────────────────────────────────

    def _add_single_run(self):
        dlg = ManualRunDialog(self._project_path, self)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.result_action:
            return
        if dlg.result_action == "save":
            self._save_single_run(dlg)
        else:
            self._run_single_run(dlg)

    def _save_single_run(self, dlg):
        from mindsight.project.staging import stage_run
        try:
            spec = stage_run(self._project_path, dlg.video, dlg.meta,
                             mode="move" if dlg.move else "copy")
        except ValueError as exc:
            QMessageBox.warning(self, "Save to project", str(exc))
            return
        self._append_log(f"Staged run '{spec.run_id}'")
        self._open_project(str(self._project_path))

    def _run_single_run(self, dlg):
        from mindsight.project.staging import single_run_spec
        try:
            spec = single_run_spec(dlg.video, dlg.meta, dlg.output_dir)
        except ValueError as exc:
            QMessageBox.warning(self, "Run now", str(exc))
            return
        # One-off run through the single-source worker: project-shaped output
        # paths from the RunSpec, no ledger (Q7).
        ns = self._current_ns()
        ns.source = str(spec.source)
        ns.log = spec.output_paths["log"]
        ns.summary = spec.output_paths["summary"]
        ns.save = spec.output_paths["save"]
        ns.heatmap = spec.output_paths["heatmap"]
        self._frame_q = queue.Queue(maxsize=2)
        self._log_q = queue.Queue()
        from .workers import GazeWorker
        self._one_off_worker = GazeWorker(ns, self._frame_q, self._log_q)
        self._one_off_worker.start()
        self._append_log(f"Running '{spec.run_id}' now...")
        self._stop_requested = False
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._poll_timer.start(60)

    # ── Batch run / stop / poll ──────────────────────────────────────────────

    def _start(self):
        if self._worker and self._worker.is_alive():
            # Visible feedback instead of a silent no-op (G-FIX-2): after Stop
            # the worker finalizes the current video before exiting (T8).
            self._append_log(
                "Previous run is still finishing -- try again in a moment.")
            return
        if not self._project_path:
            QMessageBox.critical(self, "Run", "Open a project first.")
            return
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved changes", "Save project.yaml before running?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._save_project_yaml()

        ns = self._current_ns()
        ns.project = str(self._project_path)
        ns.no_resume = not self._resume
        project_cfg = self._build_project_config()

        self._progress_q = queue.Queue()
        self._log_q = queue.Queue()
        self._frame_q = queue.Queue(maxsize=2)

        from .workers import ProjectWorker
        self._worker = ProjectWorker(
            str(self._project_path), ns, self._progress_q, self._log_q,
            self._frame_q, project_cfg=project_cfg)
        self._worker.start()
        self._stop_requested = False
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._log_box.clear()
        self._append_log("Starting run...")
        self._poll_timer.start(100)

    def _stop(self):
        """Signal the running worker(s) to cancel (G-FIX-2).

        The stop Event trips the batch CancelToken; the current video finalizes
        through the pipeline's post-run paths (T8), so polling MUST continue
        until the worker's end sentinel arrives -- ``_finish_run`` then flips the
        buttons back and logs the terminal state.
        """
        stopped_any = False
        if self._worker and self._worker.is_alive():
            self._worker.stop()
            stopped_any = True
        if self._one_off_worker and self._one_off_worker.is_alive():
            self._one_off_worker.stop()
            stopped_any = True
        self._stop_requested = True
        self._stop_btn.setEnabled(False)
        if stopped_any:
            self._append_log("Cancelling -- finishing the current video...")
            # keep the poll timer running; _finish_run fires on the sentinel
        else:
            self._finish_run()

    def _poll(self):
        try:
            while True:
                self._append_log(self._log_q.get_nowait())
        except queue.Empty:
            pass

        # Frames -> preview.  The paint happens OUTSIDE the drain try-block:
        # queue.Empty is the NORMAL exit of the drain loop and must not skip
        # painting the last frame pulled (G-FIX-1).
        frame = None
        try:
            while True:
                f = self._frame_q.get_nowait()
                if f is None:
                    break
                frame = f
        except queue.Empty:
            pass
        if frame is not None:
            pw = self._preview.width() or 480
            ph = self._preview.height() or 320
            self._preview.setPixmap(_bgr_to_pixmap(frame, pw, ph))

        try:
            while True:
                event = self._progress_q.get_nowait()
                if event is None:
                    self._finish_run()
                    return
                self._handle_progress(event)
        except queue.Empty:
            pass

        alive = ((self._worker and self._worker.is_alive())
                 or (self._one_off_worker and self._one_off_worker.is_alive()))
        if not alive:
            self._finish_run()

    def _handle_progress(self, event: dict):
        etype = event.get("type")
        if etype == "progress":
            run_id = event.get("source_name", "")
            self._row_status(run_id, "processing", "#1e6fb8")
            row = self._run_rows.get(run_id)
            if row is not None:
                self._set_cell(row, 6, "running...")
        elif etype == "skipped":
            self._row_status(event.get("run_id", ""), "skipped", "#888")
        elif etype == "archived":
            self._row_status(event.get("run_id", ""), "archived", "#b8860b")
        elif etype == "video_done":
            run_id = event.get("run_id", "")
            self._row_status(run_id, "done", "#2a7a2a")
            row = self._run_rows.get(run_id)
            if row is not None:
                self._set_cell(row, 6, "100%")
        elif etype == "video_error":
            run_id = event.get("run_id", "")
            self._row_status(run_id, "error", "#b22222")
            row = self._run_rows.get(run_id)
            if row is not None:
                self._set_cell(row, 7, event.get("error", ""))

    def _finish_run(self):
        """Terminal UI transition: buttons back, log line, workers cleared so a
        following Start launches a FRESH worker (G-FIX-2)."""
        self._poll_timer.stop()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._stop_requested:
            self._append_log("Cancelled.")
        self._stop_requested = False
        self._worker = None
        self._one_off_worker = None
        if self._project:
            self._refresh_runs_table()

    def _append_log(self, msg: str):
        self._log_box.append(str(msg))
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum())
