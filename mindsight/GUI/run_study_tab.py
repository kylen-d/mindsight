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
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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

    def _default_output_dir(self) -> str:
        """The open project's Outputs root (B1 F1), or "" when no project."""
        if not self._project_path:
            return ""
        try:
            from mindsight.project.runner import load_project_config
            from mindsight.project.staging import _out_root
            cfg = load_project_config(Path(self._project_path))
            return str(_out_root(Path(self._project_path), cfg))
        except Exception:
            return str(Path(self._project_path) / "Outputs")

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
            # B1 F1: start the picker at (and default to) the open project's
            # Outputs root instead of leaving output_dir empty -- an empty dir
            # used to send one-off outputs to a CWD-relative Outputs/ that
            # vanishes for a Finder/installer launch.
            default_out = self._default_output_dir()
            out = QFileDialog.getExistingDirectory(
                self, "Output directory", default_out)
            chosen = out or default_out
            if not chosen:
                QMessageBox.warning(self, "Run now",
                                    "Choose an output directory.")
                return
            self.output_dir = chosen
        self.result_action = action
        self.video = video
        self.meta = meta
        self.move = self._move.isChecked()
        self.accept()


class EditRunDialog(QDialog):
    """Edit ONE staged run's participants / conditions before running (G-DEFER-1).

    Pre-filled from the run's current values; exposes ``participants`` (a
    ``{track_id: label}`` map or ``None`` to clear) and ``conditions`` (a list of
    tags, possibly empty) after Accept.  The WRITE is the caller's job (the
    ``staging.update_run_metadata`` thin-caller contract, D11).
    """

    def __init__(self, run_id: str, pid_map: dict | None,
                 conditions: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit run: {run_id}")
        self.participants: dict | None = None
        self.conditions: list | None = None
        self._build_ui(pid_map, conditions)

    def _build_ui(self, pid_map, conditions):
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self._participants = QLineEdit()
        self._participants.setPlaceholderText("0:S70, 1:S71  (track:label)")
        if pid_map:
            self._participants.setText(
                ", ".join(f"{k}:{v}" for k, v in pid_map.items()))
        form.addRow("Participants:", self._participants)
        self._conditions = QLineEdit()
        self._conditions.setPlaceholderText("collab, kitchenA")
        if conditions:
            self._conditions.setText(", ".join(
                t for t in conditions.split("|") if t))
        form.addRow("Conditions:", self._conditions)
        lay.addLayout(form)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._finish)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

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

    def _finish(self):
        try:
            self.participants = self._parse_participants()
        except ValueError as exc:
            QMessageBox.warning(self, "Edit run", str(exc))
            return
        self.conditions = [c.strip() for c in self._conditions.text().split(",")
                           if c.strip()]
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
        self._dashboard_q: queue.Queue = queue.Queue(maxsize=30)
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll)

        # One-click weight fetch (missing required weights from preflight).
        self._weights_q: queue.Queue = queue.Queue()
        self._weight_threads: list = []
        self._weight_timer = None
        self._fetchable: list = []

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Project picker row: editable path field (type/paste + Enter or Open,
        # G-FIX-3) + Browse dialog + Recent dropdown (D12)
        top = QHBoxLayout()
        top.addWidget(QLabel("Project:"))
        self._project_dir = QLineEdit()
        self._project_dir.setPlaceholderText(
            "Type or paste a project folder path, or Browse...")
        self._project_dir.returnPressed.connect(self._open_typed_path)
        top.addWidget(self._project_dir, 1)
        go_btn = QPushButton("Open")
        go_btn.setToolTip("Open the typed project path")
        go_btn.clicked.connect(self._open_typed_path)
        top.addWidget(go_btn)
        open_btn = QPushButton("Browse...")
        open_btn.clicked.connect(self._open_project_dialog)
        top.addWidget(open_btn)
        new_btn = QPushButton("New Project...")
        new_btn.setToolTip("Create a blank project folder and open it")
        new_btn.clicked.connect(self._new_project_dialog)
        top.addWidget(new_btn)
        self._recent = QComboBox()
        self._recent.setMinimumWidth(180)
        self._recent.setToolTip("Recently opened projects")
        self._recent.activated.connect(self._open_recent)
        top.addWidget(self._recent)
        outer.addLayout(top)
        self._refresh_recent_dropdown()

        self._status_label = QLabel(
            "No project open -- quick analysis available below, or open a "
            "project.")
        self._status_label.setStyleSheet("color: #888; font-style: italic;")
        outer.addWidget(self._status_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: preflight + runs table + resume controls + manual, stacked in a
        # vertical splitter (B9) so the RA can drag the boundaries -- e.g. grow
        # the runs list without losing the study-setup area below.
        left_split = QSplitter(Qt.Orientation.Vertical)
        self._left_split = left_split

        # Quick analysis (UP1 D3): a first-class projectless run path. Sits at
        # the top of the left column; expanded and alone when no project is
        # open, collapsed-but-available once a project is.
        self._quick_grp = self._build_quick_analysis()
        left_split.addWidget(self._quick_grp)

        pf_grp = QGroupBox("Preflight")
        self._pf_grp = pf_grp
        pf_lay = QVBoxLayout(pf_grp)
        self._checklist = PreflightChecklist()
        pf_lay.addWidget(self._checklist)
        self._fetch_btn = QPushButton("Download missing weights")
        self._fetch_btn.setVisible(False)
        self._fetch_btn.clicked.connect(self._start_weight_fetch)
        pf_lay.addWidget(self._fetch_btn)
        rerun_pf = QPushButton("Re-run preflight")
        rerun_pf.clicked.connect(self._run_preflight)
        pf_lay.addWidget(rerun_pf)
        left_split.addWidget(pf_grp)

        runs_grp = QGroupBox("Runs")
        self._runs_grp = runs_grp
        runs_lay = QVBoxLayout(runs_grp)
        self._runs_table = QTableWidget(0, len(_RUN_COLS))
        # G-DEFER-2: keep the runs window compact + scrollable so the study-setup
        # area below gets the room; the table scrolls internally past a few runs.
        self._runs_table.setMaximumHeight(220)
        self._runs_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
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
        left_split.addWidget(runs_grp)

        # Study setup (collapsible) -- relocated from the retired Project Mode tab.
        # G-DEFER-2: it takes the growing space (stretch) so participants /
        # conditions editing has room; the runs table above stays compact.
        self._study_setup_grp = self._build_study_setup()
        left_split.addWidget(self._study_setup_grp)
        # Preserve the prior proportions: quick/preflight/runs stay compact,
        # study setup takes the growing space (indices shifted by the quick
        # group now at 0).
        left_split.setStretchFactor(0, 0)
        left_split.setStretchFactor(1, 0)
        left_split.setStretchFactor(2, 0)
        left_split.setStretchFactor(3, 1)

        splitter.addWidget(left_split)

        # RIGHT: preview + output tabs, stacked in a vertical splitter (B9) so
        # the preview / output-panel boundary is draggable.
        right_split = QSplitter(Qt.Orientation.Vertical)
        self._right_split = right_split
        self._preview = QLabel()
        self._preview.setStyleSheet("background: #1a1a2e;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(240, 160)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_split.addWidget(self._preview)
        right_split.addWidget(self._build_output_tabs())
        right_split.setStretchFactor(0, 2)
        right_split.setStretchFactor(1, 1)

        splitter.addWidget(right_split)
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

        # UP1 D3: start in projectless mode (Quick group expanded, project-only
        # groups hidden) until a project is opened.
        self._apply_project_mode()

    # ── Quick analysis (UP1 D3) ──────────────────────────────────────────────

    def _build_quick_analysis(self):
        """The projectless Quick analysis panel: pick a single video file OR a
        live camera, accept/edit an output folder, press Analyze (UP1 D3)."""
        from PyQt6.QtWidgets import QButtonGroup, QRadioButton

        grp = CollapsibleGroupBox("Quick analysis -- no project needed")
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(4, 4, 4, 4)

        # Source choice: video file vs live camera.
        src_row = QHBoxLayout()
        self._quick_src_video = QRadioButton("Video file")
        self._quick_src_camera = QRadioButton("Camera")
        self._quick_src_video.setChecked(True)
        self._quick_src_group = QButtonGroup(inner)
        self._quick_src_group.addButton(self._quick_src_video)
        self._quick_src_group.addButton(self._quick_src_camera)
        self._quick_src_video.toggled.connect(self._quick_update_source_mode)
        src_row.addWidget(self._quick_src_video)
        src_row.addWidget(self._quick_src_camera)
        src_row.addStretch(1)
        lay.addLayout(src_row)

        # Video file row.
        vid_row = QHBoxLayout()
        vid_row.addWidget(QLabel("Video:"))
        self._quick_video = QLineEdit()
        self._quick_video.setPlaceholderText("Select a video file...")
        self._quick_video.textChanged.connect(self._quick_recompute_output)
        vid_browse = QPushButton("Browse...")
        vid_browse.clicked.connect(self._quick_browse_video)
        vid_row.addWidget(self._quick_video, 1)
        vid_row.addWidget(vid_browse)
        lay.addLayout(vid_row)

        # Camera row (no device probing at build time -- probing triggers macOS
        # permission prompts + multi-second hangs; the capture opens only on
        # Analyze, and a bad index surfaces the pipeline's plain-English error).
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self._quick_camera = QComboBox()
        self._quick_camera.addItems([f"Camera {i}" for i in range(4)])
        self._quick_camera.currentIndexChanged.connect(
            self._quick_recompute_output)
        cam_row.addWidget(self._quick_camera, 1)
        lay.addLayout(cam_row)

        # Output folder row (auto-defaulted, user edits stick via a dirty flag).
        self._quick_output_dirty = False
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._quick_output = QLineEdit()
        self._quick_output.setPlaceholderText("Output folder...")
        self._quick_output.textEdited.connect(self._quick_mark_output_dirty)
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._quick_browse_output)
        out_row.addWidget(self._quick_output, 1)
        out_row.addWidget(out_browse)
        lay.addLayout(out_row)

        analyze = QPushButton("Analyze")
        analyze.setToolTip("Run inference on the chosen video or camera")
        analyze.clicked.connect(self._run_quick)
        lay.addWidget(analyze)

        grp.set_content(inner)
        self._quick_update_source_mode()
        return grp

    def _quick_update_source_mode(self, *_):
        video_mode = self._quick_src_video.isChecked()
        self._quick_video.setEnabled(video_mode)
        self._quick_camera.setEnabled(not video_mode)
        self._quick_recompute_output()

    def _quick_mark_output_dirty(self, *_):
        self._quick_output_dirty = True

    def _quick_default_output(self) -> str:
        """The auto output default: ``<PROJECT_ROOT>/Outputs/<stem>`` for a
        video, ``<PROJECT_ROOT>/Outputs/camera<idx>`` for a camera (UP1 ruling
        1 -- PROJECT_ROOT honors MINDSIGHT_HOME)."""
        from mindsight.constants import PROJECT_ROOT
        outputs = PROJECT_ROOT / "Outputs"
        if self._quick_src_camera.isChecked():
            return str(outputs / f"camera{self._quick_camera.currentIndex()}")
        stem = Path(self._quick_video.text().strip()).stem
        return str(outputs / stem) if stem else ""

    def _quick_recompute_output(self, *_):
        """Refresh the auto output default unless the user has edited the field."""
        if self._quick_output_dirty:
            return
        # setText fires textChanged, not textEdited, so the dirty flag stays off.
        self._quick_output.setText(self._quick_default_output())

    def _quick_browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video", "",
            "Video (*.mp4 *.mov *.avi *.mkv);;All (*)")
        if path:
            self._quick_src_video.setChecked(True)
            self._quick_video.setText(path)

    def _quick_browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Output folder")
        if path:
            self._quick_output.setText(path)
            self._quick_output_dirty = True

    def _run_quick(self):
        """Launch a projectless quick run from the current source choice (UP1
        D3): a single video via ``single_run_spec`` or a live camera via
        ``camera_run_spec``, then the shared one-off launch tail."""
        if self._any_worker_alive():
            # B1 F2: never start a second worker over a live one (shared guard).
            self._append_log(
                "Previous run is still finishing -- try again in a moment.")
            return
        from mindsight.project.staging import camera_run_spec, single_run_spec
        output_dir = self._quick_output.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Quick analysis",
                                "Choose an output folder.")
            return
        try:
            if self._quick_src_camera.isChecked():
                spec = camera_run_spec(self._quick_camera.currentIndex(),
                                       output_dir)
            else:
                spec = single_run_spec(self._quick_video.text().strip(),
                                       meta=None, output_dir=output_dir)
        except ValueError as exc:
            QMessageBox.warning(self, "Quick analysis", str(exc))
            return
        self._launch_one_off(spec)

    def _apply_project_mode(self):
        """Toggle the tab between projectless (Quick analysis expanded, alone)
        and project (Preflight/Runs/Study-setup shown, Quick collapsed but
        available) modes (UP1 D3).  Does NOT touch the status label -- the open
        success/failure paths own that text."""
        has_project = self._project is not None
        self._pf_grp.setVisible(has_project)
        self._runs_grp.setVisible(has_project)
        self._study_setup_grp.setVisible(has_project)
        self._quick_grp.setChecked(not has_project)

    def _build_study_setup(self):
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

        # Anonymize Footage toggle (G-DEFER-3): maps to the existing anonymize
        # namespace field; default OFF (byte-neutral -- project runs identical to
        # today when unchecked).  A small mode picker mirrors the Gaze Tuning
        # output section (blur / black).
        anon_row = QHBoxLayout()
        self._anonymize_cb = QCheckBox("Anonymize Footage")
        self._anonymize_cb.setToolTip(
            "Obscure faces in the output video for every run in this study.")
        self._anonymize_mode = QComboBox()
        self._anonymize_mode.addItems(["blur", "black"])
        self._anonymize_mode.setEnabled(False)
        self._anonymize_cb.toggled.connect(self._anonymize_mode.setEnabled)
        anon_row.addWidget(self._anonymize_cb)
        anon_row.addWidget(self._anonymize_mode)
        anon_row.addStretch(1)
        lay.addLayout(anon_row)

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

        # G-DEFER-2: wrap the setup content in a scroll area so everything stays
        # reachable at the default window size even when both tables are populated.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(inner)
        grp.set_content(scroll)
        return grp

    # ── Output panel: Log | Charts | Output CSVs (G-ENH-4) ──────────────────

    def _build_output_tabs(self) -> QTabWidget:
        """Tabbed bottom-right panel: run log, in-GUI phenomena charts rendered
        from the already-written CSVs (display-only, D11 -- nothing is written
        into the project's Outputs/ tree), and a read-only CSV viewer."""
        tabs = QTabWidget()
        self._output_tabs = tabs

        # -- Log ----------------------------------------------------------
        log_w = QWidget()
        log_lay = QVBoxLayout(log_w)
        log_lay.setContentsMargins(4, 4, 4, 4)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(60)
        self._log_box.setFont(QFont("Courier", 10))
        log_lay.addWidget(self._log_box)
        tabs.addTab(log_w, "Log")

        # -- Charts (in-GUI, from written CSVs) -----------------------------
        charts_w = QWidget()
        charts_lay = QVBoxLayout(charts_w)
        charts_lay.setContentsMargins(4, 4, 4, 4)
        chart_row = QHBoxLayout()
        chart_row.addWidget(QLabel("Run:"))
        self._charts_run = QComboBox()
        self._charts_run.currentIndexChanged.connect(self._render_charts)
        chart_row.addWidget(self._charts_run, 1)
        charts_lay.addLayout(chart_row)
        self._chart_placeholder = QLabel(
            "Charts appear here once a run has written its CSVs.")
        self._chart_placeholder.setStyleSheet("color: #888; font-style: italic;")
        self._chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        charts_lay.addWidget(self._chart_placeholder, 1)
        # B4a: host the canvas in a scroll area (widgetResizable False so the
        # figure keeps its NATURAL height) -- with N phenomena panels the canvas
        # is taller than the tab, so it scrolls instead of squashing every panel.
        self._chart_scroll = QScrollArea()
        self._chart_scroll.setWidgetResizable(False)
        self._chart_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._chart_scroll.setVisible(False)
        charts_lay.addWidget(self._chart_scroll, 1)
        self._chart_canvas = None            # created lazily on first render
        self._charts_layout = charts_lay
        tabs.addTab(charts_w, "Charts")

        # -- Live (live dashboard during a run, B6) -------------------------
        from .live_dashboard import LiveDashboardPanel
        self._dashboard_panel = LiveDashboardPanel(self._dashboard_q)
        tabs.addTab(self._dashboard_panel, "Live")

        # -- Output CSVs (read-only viewer) ---------------------------------
        csv_w = QWidget()
        csv_lay = QVBoxLayout(csv_w)
        csv_lay.setContentsMargins(4, 4, 4, 4)
        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("Run:"))
        self._csv_run = QComboBox()
        self._csv_run.currentIndexChanged.connect(self._populate_csv_files)
        csv_row.addWidget(self._csv_run, 1)
        csv_row.addWidget(QLabel("File:"))
        self._csv_file = QComboBox()
        self._csv_file.currentIndexChanged.connect(self._load_csv_table)
        csv_row.addWidget(self._csv_file, 1)
        csv_lay.addLayout(csv_row)
        self._csv_table = QTableWidget(0, 0)
        self._csv_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        csv_lay.addWidget(self._csv_table, 1)
        self._csv_note = QLabel("")
        self._csv_note.setStyleSheet("color: #888; font-size: 11px;")
        csv_lay.addWidget(self._csv_note)
        tabs.addTab(csv_w, "Output CSVs")

        self._run_outputs = {}               # run_id -> RunOutputs
        return tabs

    def _refresh_output_panels(self):
        """Rediscover which runs have written CSVs; repopulate both selectors
        (keeping the current selection when it survives)."""
        if not self._project:
            return
        from .run_outputs import discover_global_outputs, discover_run_outputs
        try:
            outputs = discover_run_outputs(self._project.runs())
        except Exception as exc:
            self._append_log(f"[WARN] could not scan run outputs: {exc}")
            return
        self._run_outputs = {o.run_id: o for o in outputs}
        # Project-level aggregates (Global_*.csv + By Condition/) are staged by
        # no RunSpec -- surface them as a "Global (project)" entry in both
        # selectors when the batch post-processing has written any.  Resolve the
        # output root the same way the runner does (Project._out_root).
        try:
            glob = discover_global_outputs(self._project._out_root())
        except Exception as exc:
            glob = None
            self._append_log(f"[WARN] could not scan global outputs: {exc}")
        if glob is not None:
            self._run_outputs[glob.run_id] = glob
        for combo, on_change in ((self._charts_run, self._render_charts),
                                 (self._csv_run, self._populate_csv_files)):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(list(self._run_outputs))
            idx = combo.findText(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
            on_change()

    def _render_charts(self):
        """Render the selected run's phenomena charts in-GUI (display-only)
        from its written summary/Events CSVs."""
        out = self._run_outputs.get(self._charts_run.currentText())
        if out is None or (out.summary is None and out.events is None):
            self._chart_scroll.setVisible(False)
            self._chart_placeholder.setVisible(True)
            return

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from .run_outputs import gaze_timeline, look_time_table

        panels = []
        if out.summary is not None:
            try:
                table = look_time_table(out.summary)
                if table:
                    panels.append(("look", table))
            except Exception:
                pass
        if out.events is not None:
            try:
                objects, per = gaze_timeline(out.events)
                if objects and per:
                    panels.append(("timeline", (objects, per)))
            except Exception:
                pass
        if not panels:
            self._chart_scroll.setVisible(False)
            self._chart_placeholder.setVisible(True)
            return

        fig = Figure(figsize=(6, 3.2 * len(panels)), facecolor="#121212")
        for i, (kind, data) in enumerate(panels):
            ax = fig.add_subplot(len(panels), 1, i + 1)
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#cccccc", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#2a2a2a")
            if kind == "look":
                self._draw_look_time(ax, data)
            else:
                self._draw_timeline(ax, *data)
            ax.legend(fontsize=7, facecolor="#1e1e1e",
                      labelcolor="#cccccc", edgecolor="#2a2a2a")
        fig.tight_layout()

        if self._chart_canvas is None:
            self._chart_canvas = FigureCanvasQTAgg(fig)
            self._chart_scroll.setWidget(self._chart_canvas)
        else:
            self._chart_canvas.figure = fig
            fig.set_canvas(self._chart_canvas)
        # Size the canvas to the figure's natural pixel size so the scroll area
        # (widgetResizable False) scrolls past the tab height instead of
        # squashing the stacked panels.
        w_in, h_in = fig.get_size_inches()
        dpi = fig.get_dpi()
        self._chart_canvas.setMinimumSize(int(w_in * dpi), int(h_in * dpi))
        self._chart_canvas.resize(int(w_in * dpi), int(h_in * dpi))
        self._chart_scroll.setVisible(True)
        self._chart_canvas.setVisible(True)
        self._chart_placeholder.setVisible(False)
        self._chart_canvas.draw_idle()

    @staticmethod
    def _draw_look_time(ax, table: dict):
        objects = sorted({o for objs in table.values() for o in objs})
        participants = sorted(table)
        width = 0.8 / max(1, len(participants))
        for pi, who in enumerate(participants):
            vals = [table[who].get(o, 0.0) for o in objects]
            xs = [i + pi * width for i in range(len(objects))]
            ax.bar(xs, vals, width=width, label=who)
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(objects))])
        ax.set_xticklabels(objects, rotation=30, ha="right",
                           color="#cccccc", fontsize=7)
        ax.set_ylabel("% of video", color="#cccccc", fontsize=8)
        ax.set_title("Object look time", color="#cccccc", fontsize=9,
                     loc="left")

    @staticmethod
    def _draw_timeline(ax, objects: list, per: dict):
        for who in sorted(per):
            xs, ys = per[who]
            ax.scatter(xs, ys, s=4, label=who)
        ax.set_yticks(range(len(objects)))
        ax.set_yticklabels(objects, color="#cccccc", fontsize=7)
        ax.set_xlabel("t (seconds)", color="#cccccc", fontsize=8)
        ax.set_title("Gaze target timeline", color="#cccccc", fontsize=9,
                     loc="left")

    def _populate_csv_files(self):
        out = self._run_outputs.get(self._csv_run.currentText())
        self._csv_file.blockSignals(True)
        self._csv_file.clear()
        if out is not None:
            for p in out.csv_paths:
                self._csv_file.addItem(p.name, str(p))
        self._csv_file.setCurrentIndex(0 if self._csv_file.count() else -1)
        self._csv_file.blockSignals(False)
        self._load_csv_table()

    def _load_csv_table(self):
        path = self._csv_file.currentData()
        if not path or not Path(path).is_file():
            self._csv_table.setRowCount(0)
            self._csv_table.setColumnCount(0)
            self._csv_note.setText("")
            self._csv_note.setStyleSheet("color: #888; font-size: 11px;")
            return
        from .run_outputs import load_csv_rows
        try:
            header, rows, total_rows = load_csv_rows(path)
        except Exception as exc:
            self._csv_note.setText(f"Could not read CSV: {exc}")
            return
        self._csv_table.setColumnCount(len(header))
        self._csv_table.setHorizontalHeaderLabels(header)
        self._csv_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row[:len(header)]):
                self._csv_table.setItem(r, c, QTableWidgetItem(val))
        name = Path(path).name
        if total_rows > len(rows):
            # Loud, warn-coloured so a capped view is never mistaken for the
            # run's true row count.
            self._csv_note.setText(
                f"{name}: showing first {len(rows):,} of {total_rows:,} rows"
                " -- open the file for the full data")
            self._csv_note.setStyleSheet(
                f"color: {_SEV_COLOUR['warn']}; font-size: 11px;"
                " font-weight: bold;")
        else:
            self._csv_note.setText(f"{name}: {total_rows:,} row(s)")
            self._csv_note.setStyleSheet("color: #888; font-size: 11px;")

    # ── Project open / recent ────────────────────────────────────────────────

    def _open_project_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open MindSight project folder")
        if path:
            self._open_project(path)

    def _new_project_dialog(self):
        """Prompt for a parent folder + name, scaffold a blank project, open it."""
        parent = QFileDialog.getExistingDirectory(
            self, "Choose where to create the new project")
        if not parent:
            return
        name, ok = QInputDialog.getText(
            self, "New Project", "Project name:")
        if not ok or not name.strip():
            return
        from mindsight.project.runner import create_project
        try:
            project = create_project(parent, name.strip())
        except (ValueError, OSError) as exc:
            self._status_label.setText(f"Could not create project: {exc}")
            self._status_label.setStyleSheet("color: #b22222; font-weight: bold;")
            return
        # Reuse the normal open path so validation + preflight + state wiring run.
        self._open_project(str(project))

    def _open_typed_path(self):
        """Open the path typed/pasted into the project field (G-FIX-3)."""
        text = self._project_dir.text().strip()
        if not text:
            return
        self._open_project(str(Path(text).expanduser()))

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
        except Exception as exc:
            # A typed/pasted path can be anything (missing dir, a file, a
            # broken project.yaml...) -- always a readable inline error, never
            # a crash (G-FIX-3).
            self._status_label.setText(f"Invalid project: {exc}")
            self._status_label.setStyleSheet("color: #b22222; font-weight: bold;")
            # UP1 D3: reflect current project state (unchanged on a failed open),
            # keeping the Quick panel available -- the error text stays put.
            self._apply_project_mode()
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
        self._refresh_output_panels()

        self._status_label.setText(f"Open: {project.path.name}")
        self._status_label.setStyleSheet("color: #2a7a2a; font-weight: bold;")
        # UP1 D3: a project is open -- show the project groups, collapse Quick.
        self._apply_project_mode()

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
            self._update_fetch_offer([])
            return
        try:
            report = self._project.preflight(ns=self._current_ns())
        except Exception as exc:  # pragma: no cover - preflight never raises
            self._append_log(f"[WARN] preflight failed: {exc}")
            return
        self._checklist.render(report)
        offer = self._fetchable_missing() if not report.ok else []
        self._update_fetch_offer(offer)

    # ── One-click fetch of missing weights (Step 11) ─────────────────────────

    def _fetchable_missing(self) -> list:
        """Manifest entries the manager can fetch for this config's missing
        weights (consume-don't-compute: the manifest module decides, D11)."""
        from mindsight import weights
        from mindsight.outputs import provenance
        try:
            collected = provenance.collect_weights(self._current_ns())
        except Exception:
            return []
        missing = [Path(w.get("resolved", "")).name
                   for w in collected.values() if w.get("sha256") == "missing"]
        try:
            return weights.downloadable_missing(missing)
        except Exception:
            return []

    def _update_fetch_offer(self, entries: list):
        self._fetchable = entries
        if entries:
            self._fetch_btn.setText(
                f"Download {len(entries)} missing weight(s)")
            self._fetch_btn.setEnabled(True)
            self._fetch_btn.setVisible(True)
        else:
            self._fetch_btn.setVisible(False)

    def _start_weight_fetch(self):
        from .workers import WeightsDownloadWorker
        entries = self._fetchable
        if not entries:
            return
        self._fetch_btn.setEnabled(False)
        self._fetch_btn.setText("Downloading weights...")
        worker = WeightsDownloadWorker(entries, self._weights_q)
        self._weight_threads.append(worker)
        worker.start()
        if self._weight_timer is None:
            self._weight_timer = QTimer()
            self._weight_timer.setInterval(150)
            self._weight_timer.timeout.connect(self._drain_weight_fetch)
            self._weight_timer.start()

    def _drain_weight_fetch(self):
        """Apply fetch-worker results on the GUI thread; re-run preflight when
        the batch finishes.  Safe to call directly (tests)."""
        finished = False
        try:
            while True:
                kind, entry, payload = self._weights_q.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    self._append_log(f"Downloaded {entry['filename']}.")
                elif kind == "error":
                    self._append_log(
                        f"[WARN] weight download failed: {payload}")
                elif kind == "finished":
                    finished = True
        except queue.Empty:
            pass
        if finished:
            self._run_preflight()

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
        edit_act = menu.addAction("Edit run...")
        rerun_act = menu.addAction("Re-run this run")
        chosen = menu.exec(self._runs_table.viewport().mapToGlobal(pos))
        if chosen == edit_act:
            self._edit_run(run_id)
        elif chosen == rerun_act:
            self._project.invalidate(run_id)
            self._append_log(f"Marked '{run_id}' for re-run.")
            self._refresh_runs_table()

    def _edit_run(self, run_id: str):
        """Edit a staged run's participants / conditions before running
        (G-DEFER-1).  The WRITE goes through ``staging.update_run_metadata``
        (run.yaml for run-folder projects, project.yaml for legacy); the tab
        just collects values and refreshes preflight + the runs table."""
        from mindsight.project.staging import update_run_metadata
        spec = next((s for s in self._project.runs() if s.run_id == run_id), None)
        if spec is None:
            return
        dlg = EditRunDialog(run_id, spec.pid_map, spec.conditions, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            update_run_metadata(self._project_path, run_id,
                                participants=dlg.participants,
                                conditions=dlg.conditions)
        except ValueError as exc:
            QMessageBox.warning(self, "Edit run", str(exc))
            return
        self._append_log(f"Updated metadata for '{run_id}'.")
        # Reopen so the facade reloads project.yaml / run.yaml, then refresh.
        self._open_project(str(self._project_path))

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

    def _apply_anonymize(self, ns):
        """Set ``ns.anonymize`` from the study-setup toggle (G-DEFER-3).

        The study-setup checkbox is the single control for project-run
        anonymization: checked -> the selected mode; unchecked -> ``None`` (today's
        behavior -- project runs never anonymized), so an unchecked box keeps runs
        byte-identical regardless of any stray Gaze Tuning value.
        """
        ns.anonymize = (self._anonymize_mode.currentText()
                        if self._anonymize_cb.isChecked() else None)

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
        if self._any_worker_alive():
            # B1 F2: never start a second worker over a live one (shared guard).
            self._append_log(
                "Previous run is still finishing -- try again in a moment.")
            return
        from mindsight.project.staging import single_run_spec
        try:
            # B1 F1: pass the open project so an omitted output_dir defaults to
            # the project's Outputs root, not a CWD-relative Outputs/ that
            # vanishes for a Finder/installer launch.
            spec = single_run_spec(dlg.video, dlg.meta, dlg.output_dir,
                                   project=self._project_path)
        except ValueError as exc:
            QMessageBox.warning(self, "Run now", str(exc))
            return
        self._launch_one_off(spec)

    def _launch_one_off(self, spec):
        """Start a one-off single-source/camera run from *spec* (shared tail of
        the "Run now" dialog and Quick analysis paths, UP1 D3).

        Project-shaped output paths come from the RunSpec, no ledger (Q7).  The
        output folder is created here (UP1 ruling 1) -- the CSV writer opens the
        log path directly, so a not-yet-existing quick-run default would crash.
        """
        # One-off run through the single-source worker: project-shaped output
        # paths from the RunSpec, no ledger (Q7).
        ns = self._current_ns()
        self._apply_anonymize(ns)
        ns.source = str(spec.source)
        # Save-on-run checkpoint (warn-not-raise): persist the launched config.
        from .settings_manager import checkpoint
        checkpoint(ns)
        ns.log = spec.output_paths["log"]
        ns.summary = spec.output_paths["summary"]
        ns.save = spec.output_paths["save"]
        ns.heatmap = spec.output_paths["heatmap"]
        # UP1 ruling 1: create the output folder on run (writers open the log
        # path directly and do not mkdir it).
        Path(spec.output_paths["log"]).parent.mkdir(parents=True, exist_ok=True)
        self._frame_q = queue.Queue(maxsize=2)
        self._log_q = queue.Queue()
        from .workers import GazeWorker
        self._one_off_worker = GazeWorker(ns, self._frame_q, self._log_q,
                                          dashboard_q=self._dashboard_q)
        self._one_off_worker.start()
        # B6: live dashboard follows the one-off run too.
        self._dashboard_panel.reset()
        self._dashboard_panel.start()
        self._append_log(f"Running '{spec.run_id}' now...")
        # B1 F1: always tell the user where the one-off files land (absolute).
        out_dir = str(Path(spec.output_paths["log"]).resolve().parent)
        self._append_log(f"One-off outputs -> {out_dir}")
        self._stop_requested = False
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._poll_timer.start(60)

    # ── Batch run / stop / poll ──────────────────────────────────────────────

    def _any_worker_alive(self) -> bool:
        """True while EITHER the batch or the one-off worker is running (B1 F2).

        Both run paths share this guard so a second worker can never start
        alongside a live one -- two writers on the same Events/summary CSV would
        truncate/interleave the file.
        """
        return bool((self._worker and self._worker.is_alive())
                    or (self._one_off_worker and self._one_off_worker.is_alive()))

    def _start(self):
        if self._any_worker_alive():
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
        self._apply_anonymize(ns)
        project_cfg = self._build_project_config()

        # Save-on-run checkpoint (warn-not-raise): persist the launched config so
        # a crash mid-batch does not lose the session.
        from .settings_manager import checkpoint
        checkpoint(ns)

        self._progress_q = queue.Queue()
        self._log_q = queue.Queue()
        self._frame_q = queue.Queue(maxsize=2)

        from .workers import ProjectWorker
        self._worker = ProjectWorker(
            str(self._project_path), ns, self._progress_q, self._log_q,
            self._frame_q, project_cfg=project_cfg,
            dashboard_q=self._dashboard_q)
        self._worker.start()
        # B6: live dashboard follows the run (mirror gaze_tab.py's wiring).
        self._dashboard_panel.reset()
        self._dashboard_panel.start()
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
            # This run's CSVs just landed -- make them selectable (G-ENH-4).
            self._refresh_output_panels()
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
        self._dashboard_panel.stop()   # B6: live dashboard stops with the run
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._stop_requested:
            self._append_log("Cancelled.")
        self._stop_requested = False
        self._worker = None
        self._one_off_worker = None
        if self._project:
            self._refresh_runs_table()
            self._refresh_output_panels()

    def _append_log(self, msg: str):
        self._log_box.append(str(msg))
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum())
