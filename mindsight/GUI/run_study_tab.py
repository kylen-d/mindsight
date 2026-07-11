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
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
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
    QStackedWidget,
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

# UP1r2: the tab's three input modes and their segmented-switch styling.
_MODES = ("project", "video", "camera")
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
_SEG_QSS = """
QPushButton {
    padding: 5px 22px;
    border: 1px solid #4a4a5e;
    background: transparent;
    color: #999;
}
QPushButton:hover:!checked { background: rgba(80, 100, 140, 0.25); }
QPushButton:checked {
    background: #2d5a88;
    color: white;
    font-weight: bold;
}
"""

# Primary go buttons: green idle -> red Stop while a run is live (matches the
# old status-bar Run/Stop colour coding; the buttons now live in the source
# cards so every mode's primary action sits in the same place).
_GO_GREEN = ("QPushButton{background:#2a7a2a;color:white;"
             "font-weight:bold;padding:4px 26px;}"
             "QPushButton:disabled{background:#33333f;color:#777;}")
_GO_RED = ("QPushButton{background:#7a2a2a;color:white;"
           "font-weight:bold;padding:4px 26px;}")


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

    def __init__(self, gaze_tab=None, settings=None, parent=None):
        super().__init__(parent)
        self._gaze_tab = gaze_tab
        # Decoupling (UP2): the RunSettings store -- not Gaze Tuning -- is the
        # config source for every run launched here.  One instance is passed by
        # the main window; a self-constructed one keeps offscreen tests working.
        from .run_settings import RunSettingsStore
        self._settings = settings if settings is not None else RunSettingsStore()
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

        # UP1r2: segmented mode switch -- Project | Video File | Camera.  The
        # ENTIRE tab commits to the chosen mode: the source card and both panes
        # morph, so nothing mode-irrelevant is ever on screen.
        outer.addLayout(self._build_mode_switch())
        outer.addWidget(self._build_source_stack())

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: a stack -- page 0 = the project column (preflight + runs +
        # study setup in a vertical splitter, B9), page 1 = live charts (UP1r2:
        # in video/camera modes the left pane is exclusively live charts).
        left_split = QSplitter(Qt.Orientation.Vertical)
        self._left_split = left_split

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
        self._record_btn = QPushButton("⏺ Record Session...")
        self._record_btn.setToolTip(
            "Record a live session with this camera into the project -- the "
            "raw feed becomes the run's video and analysis starts when you "
            "end the session (UP5)")
        self._record_btn.clicked.connect(self._record_session_dialog)
        ctl_row.addWidget(self._rerun_all_btn)
        ctl_row.addWidget(add_run_btn)
        ctl_row.addWidget(self._record_btn)
        ctl_row.addStretch(1)
        runs_lay.addLayout(ctl_row)
        left_split.addWidget(runs_grp)

        # Study setup (collapsible) -- relocated from the retired Project Mode tab.
        # G-DEFER-2: it takes the growing space (stretch) so participants /
        # conditions editing has room; the runs table above stays compact.
        self._study_setup_grp = self._build_study_setup()
        left_split.addWidget(self._study_setup_grp)
        # Preflight/runs stay compact; study setup takes the growing space.
        left_split.setStretchFactor(0, 0)
        left_split.setStretchFactor(1, 0)
        left_split.setStretchFactor(2, 1)

        # Live-charts pane (page 1): hosts the shared LiveDashboardPanel while
        # a quick mode is active (the panel is reparented, see _place_dashboard).
        charts_pane = QWidget()
        self._charts_pane_lay = QVBoxLayout(charts_pane)
        self._charts_pane_lay.setContentsMargins(0, 0, 0, 0)
        hdr = QLabel("Live charts")
        hdr.setStyleSheet("font-weight: bold;")
        self._charts_pane_lay.addWidget(hdr)
        self._charts_hint = QLabel(
            "Live charts appear here during analysis.")
        self._charts_hint.setStyleSheet("color: #888; font-style: italic;")
        self._charts_pane_lay.addWidget(self._charts_hint)
        self._charts_pane_lay.addStretch(1)

        self._left_stack = QStackedWidget()
        self._left_stack.addWidget(left_split)     # page 0: project column
        self._left_stack.addWidget(charts_pane)    # page 1: live charts
        splitter.addWidget(self._left_stack)

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

        # Drag & drop: a video file switches to Video File mode, a folder opens
        # as a project (UP1r2 extras).
        self.setAcceptDrops(True)

        # Preset indicator follows the RunSettings store.
        self._settings.changed.connect(self._refresh_preset_labels)
        self._refresh_preset_labels()

        # Start in the last-used mode (stored GUI state, not inference config).
        # Run state: every mode's primary action is its card's inline go
        # button (green -> red Stop while a run is live); there are no
        # status-bar Run/Stop buttons for this tab anymore (UP1r3).
        self._running = False
        self._run_kind = None            # "project" | "quick" while running
        # UP5 live-session recording state.
        self._recorder = None
        self._recording_meta = None
        self._record_timer = QTimer(self)
        self._record_timer.timeout.connect(self._recording_tick)
        from .settings_manager import SettingsManager
        saved = SettingsManager().load_gui_state().get("analyze_mode")
        self._set_mode(saved if saved in _MODES else "project", persist=False)

    # ── Mode switch + source cards (UP1r2) ───────────────────────────────────

    def _build_mode_switch(self):
        """The segmented Project | Video File | Camera switch.  One mode, one
        UI: everything below the switch commits to the selected mode."""
        row = QHBoxLayout()
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_btns = {}
        for key, label in (("project", "Project"),
                           ("video", "Video File"),
                           ("camera", "Camera")):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet(_SEG_QSS)
            btn.clicked.connect(lambda _=False, k=key: self._set_mode(k))
            self._mode_group.addButton(btn)
            self._mode_btns[key] = btn
            seg.addWidget(btn)
        row.addLayout(seg)
        row.addStretch(1)
        return row

    def _build_source_stack(self):
        """One source card per mode, stacked -- only the active mode's inputs
        are ever on screen."""
        self._source_stack = QStackedWidget()
        self._source_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._source_stack.addWidget(self._build_project_card())
        self._source_stack.addWidget(self._build_video_card())
        self._source_stack.addWidget(self._build_camera_card())
        return self._source_stack

    def _card(self):
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        return card, lay

    def _build_project_card(self):
        """Project picker row (editable path + Enter or Open, G-FIX-3; Browse;
        New Project; Inference Settings; Recent, D12) + the status line."""
        card, lay = self._card()
        top = QHBoxLayout()
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
        infer_btn = QPushButton("Inference Settings...")
        infer_btn.setToolTip(
            "Edit the inference configuration every run launched here uses")
        infer_btn.clicked.connect(self._open_inference_settings)
        top.addWidget(infer_btn)
        self._recent = QComboBox()
        self._recent.setMinimumWidth(180)
        self._recent.setToolTip("Recently opened projects")
        self._recent.activated.connect(self._open_recent)
        top.addWidget(self._recent)
        lay.addLayout(top)
        self._refresh_recent_dropdown()

        bottom = QHBoxLayout()
        self._status_label = QLabel("No project open.")
        self._status_label.setStyleSheet("color: #888; font-style: italic;")
        bottom.addWidget(self._status_label)
        bottom.addStretch(1)
        self._project_go = QPushButton("▶  Run")
        self._project_go.setMinimumHeight(34)
        self._project_go.clicked.connect(self._go_clicked)
        bottom.addWidget(self._project_go)
        lay.addLayout(bottom)
        return card

    def _build_video_card(self):
        """Single-video card: file + output + preset/settings + Analyze."""
        card, lay = self._card()
        row = QHBoxLayout()
        row.addWidget(QLabel("Video:"))
        self._quick_video = QLineEdit()
        self._quick_video.setPlaceholderText(
            "Choose a video file, or drop one anywhere on this tab...")
        self._quick_video.textChanged.connect(self._video_recompute_output)
        row.addWidget(self._quick_video, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._quick_browse_video)
        row.addWidget(browse)
        lay.addLayout(row)

        self._video_output_dirty = False
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._video_output = QLineEdit()
        self._video_output.setPlaceholderText("Output folder...")
        self._video_output.textEdited.connect(self._video_mark_output_dirty)
        out_row.addWidget(self._video_output, 1)
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(
            lambda: self._browse_output_into(self._video_output, "video"))
        out_row.addWidget(out_browse)
        lay.addLayout(out_row)

        bottom = QHBoxLayout()
        self._video_preset = QLabel("")
        self._video_preset.setStyleSheet("color: #888;")
        bottom.addWidget(self._video_preset)
        settings_btn = QPushButton("Inference Settings...")
        settings_btn.clicked.connect(self._open_inference_settings)
        bottom.addWidget(settings_btn)
        bottom.addStretch(1)
        self._video_go = QPushButton("▶  Analyze")
        self._video_go.setMinimumHeight(34)
        self._video_go.clicked.connect(self._go_clicked)
        bottom.addWidget(self._video_go)
        lay.addLayout(bottom)
        return card

    def _build_camera_card(self):
        """Live-camera card: device + output + preset/settings + Start.  No
        device probing at build time -- probing triggers macOS permission
        prompts + multi-second hangs; Refresh probes on demand and the capture
        opens only on Start (a bad index surfaces the pipeline's plain-English
        error)."""
        card, lay = self._card()
        row = QHBoxLayout()
        row.addWidget(QLabel("Camera:"))
        self._camera_combo = QComboBox()
        self._camera_combo.addItems([f"Camera {i}" for i in range(4)])
        self._camera_combo.currentIndexChanged.connect(
            self._camera_recompute_output)
        row.addWidget(self._camera_combo, 1)
        refresh = QPushButton("Refresh")
        refresh.setToolTip(
            "Detect connected cameras and show their names (may trigger a "
            "one-time camera permission prompt)")
        refresh.clicked.connect(self._refresh_cameras)
        row.addWidget(refresh)
        lay.addLayout(row)

        self._camera_output_dirty = False
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._camera_output = QLineEdit()
        self._camera_output.setPlaceholderText("Output folder...")
        self._camera_output.textEdited.connect(self._camera_mark_output_dirty)
        out_row.addWidget(self._camera_output, 1)
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(
            lambda: self._browse_output_into(self._camera_output, "camera"))
        out_row.addWidget(out_browse)
        lay.addLayout(out_row)

        # UP5 ruling 2: one-off live captures can carry session details (for
        # labs using MindSight alongside other tools, without a project).
        details = CollapsibleGroupBox("Session details (optional)")
        inner = QWidget()
        dform = QFormLayout(inner)
        dform.setContentsMargins(4, 2, 4, 2)
        self._cam_participants = QLineEdit()
        self._cam_participants.setPlaceholderText(
            "left to right on screen, comma-separated -- e.g. S80, S81")
        dform.addRow("Participants:", self._cam_participants)
        self._cam_conditions = QLineEdit()
        self._cam_conditions.setPlaceholderText(
            "optional -- separate multiple with |")
        dform.addRow("Conditions:", self._cam_conditions)
        self._cam_session = QLineEdit()
        dform.addRow("Session:", self._cam_session)
        self._cam_notes = QLineEdit()
        dform.addRow("Notes:", self._cam_notes)
        details.set_content(inner)
        details.setChecked(False)
        lay.addWidget(details)

        bottom = QHBoxLayout()
        self._camera_preset = QLabel("")
        self._camera_preset.setStyleSheet("color: #888;")
        bottom.addWidget(self._camera_preset)
        settings_btn = QPushButton("Inference Settings...")
        settings_btn.clicked.connect(self._open_inference_settings)
        bottom.addWidget(settings_btn)
        bottom.addStretch(1)
        self._camera_go = QPushButton("▶  Start Camera")
        self._camera_go.setMinimumHeight(34)
        self._camera_go.clicked.connect(self._go_clicked)
        bottom.addWidget(self._camera_go)
        lay.addLayout(bottom)
        return card

    # ── Mode machinery (UP1r2) ───────────────────────────────────────────────

    def _set_mode(self, mode: str, persist: bool = True):
        """Commit the whole tab to *mode*: source card, left pane, output
        tabs, status-bar buttons, and empty-state hints all follow.  No
        mode-irrelevant UI survives the switch."""
        self._mode = mode
        self._source_stack.setCurrentIndex(_MODES.index(mode))
        btn = self._mode_btns[mode]
        if not btn.isChecked():
            btn.setChecked(True)
        project_mode = mode == "project"
        self._left_stack.setCurrentIndex(0 if project_mode else 1)
        self._place_dashboard(project_mode)
        # Post-run Charts render per-project run; quick modes chart LIVE on the
        # left pane instead.
        self._output_tabs.setTabVisible(1, project_mode)
        self._update_go_buttons()
        self._apply_preview_hint()
        if mode == "video":
            self._video_recompute_output()
        elif mode == "camera":
            self._camera_recompute_output()
        if persist:
            from .settings_manager import SettingsManager
            try:
                SettingsManager().save_gui_state({"analyze_mode": mode})
            except Exception:  # pragma: no cover - GUI state is best-effort
                pass

    def _place_dashboard(self, project_mode: bool):
        """Reparent the single LiveDashboardPanel: the Live output tab in
        project mode, the left charts pane in the quick modes."""
        panel = self._dashboard_panel
        in_tabs = self._output_tabs.indexOf(panel)
        if project_mode:
            if in_tabs < 0:
                self._charts_pane_lay.removeWidget(panel)
                self._output_tabs.insertTab(2, panel, "Live")
        else:
            if in_tabs >= 0:
                self._output_tabs.removeTab(in_tabs)
            if self._charts_pane_lay.indexOf(panel) < 0:
                # Before the trailing stretch, taking the growing space.
                self._charts_pane_lay.insertWidget(
                    self._charts_pane_lay.count() - 1, panel, 1)
                panel.show()

    def _apply_preview_hint(self):
        """Empty-state hint in the preview area (only while no frame shown)."""
        pm = self._preview.pixmap()
        if pm is not None and not pm.isNull():
            return
        hints = {
            "project": "Open a project and press Run -- the preview shows here.",
            "video": "Choose a video (or drop one here), then press Analyze.",
            "camera": "Pick a camera and press Start Camera.",
        }
        self._preview.setText(hints[self._mode])
        self._preview.setStyleSheet(
            "background: #1a1a2e; color: #556; font-style: italic;")

    def _refresh_preset_labels(self):
        """The quick cards' 'Preset: <source> (modified)' indicator follows the
        RunSettings store."""
        text = "Preset: " + self._settings.source_label()
        if self._settings.is_modified():
            text += " (modified)"
        for lab in (self._video_preset, self._camera_preset):
            lab.setText(text)

    def _update_go_buttons(self):
        """One primary button per source card: green go, flipping to a red
        Stop while ANY run is live (stopping is global, whichever card shows).
        The project Run greys out until a project is open.  While a live
        session records (UP5), the project button is the red End Session."""
        if self._recorder is not None:
            self._project_go.setText("■  End Session")
            self._project_go.setStyleSheet(_GO_RED)
            self._project_go.setEnabled(True)
            for btn in (self._video_go, self._camera_go):
                btn.setEnabled(False)
            return
        running = self._running
        for btn, idle in ((self._project_go, "▶  Run"),
                          (self._video_go, "▶  Analyze"),
                          (self._camera_go, "▶  Start Camera")):
            btn.setText("■  Stop" if running else idle)
            btn.setStyleSheet(_GO_RED if running else _GO_GREEN)
            btn.setEnabled(True)     # clears the _stop() cancelling grey-out
        self._project_go.setEnabled(running or self._project is not None)

    def _go_clicked(self):
        """The inline primary button: end the recording, stop the live run,
        or launch the active mode's run."""
        if self._recorder is not None:
            self._end_session_recording()
        elif self._running:
            self._stop()
        elif self._mode == "project":
            self._start()
        else:
            self._run_quick()

    def _refresh_cameras(self):
        """Probe cameras ON DEMAND (never at startup) and show device names
        where the OS provides them; falls back to a cv2 open/close probe, then
        to the blind Camera 0-3 list."""
        names = []
        try:
            from PyQt6.QtMultimedia import QMediaDevices
            names = [d.description() or f"Camera {i}"
                     for i, d in enumerate(QMediaDevices.videoInputs())]
        except Exception:  # pragma: no cover - QtMultimedia optional
            pass
        if not names:
            import cv2
            for i in range(6):
                cap = cv2.VideoCapture(i)
                ok = cap.isOpened()
                cap.release()
                if ok:
                    names.append(f"Camera {i}")
        if not names:
            names = [f"Camera {i}" for i in range(4)]
        current = max(self._camera_combo.currentIndex(), 0)
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        self._camera_combo.addItems(names)
        self._camera_combo.setCurrentIndex(min(current, len(names) - 1))
        self._camera_combo.blockSignals(False)
        self._camera_recompute_output()

    # ── Drag & drop: video file -> Video File mode, folder -> project ───────

    def dragEnterEvent(self, event):
        if any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir():
                self._set_mode("project")
                self._project_dir.setText(str(path))
                self._open_project(str(path))
                return
            if path.suffix.lower() in _VIDEO_EXTS:
                self._set_mode("video")
                self._quick_video.setText(str(path))
                return

    # ── Output defaults (auto-filled, user edits stick via dirty flags) ─────

    def _default_output_for(self, stem: str) -> str:
        """``<PROJECT_ROOT>/Outputs/<stem>`` (UP1 ruling 1 -- PROJECT_ROOT
        honors MINDSIGHT_HOME)."""
        from mindsight.constants import PROJECT_ROOT
        return str(PROJECT_ROOT / "Outputs" / stem) if stem else ""

    def _video_mark_output_dirty(self, *_):
        self._video_output_dirty = True

    def _camera_mark_output_dirty(self, *_):
        self._camera_output_dirty = True

    def _video_recompute_output(self, *_):
        if self._video_output_dirty:
            return
        # setText fires textChanged, not textEdited, so the dirty flag stays off.
        stem = Path(self._quick_video.text().strip()).stem
        self._video_output.setText(self._default_output_for(stem))

    def _camera_recompute_output(self, *_):
        if self._camera_output_dirty:
            return
        self._camera_output.setText(self._default_output_for(
            f"camera{self._camera_combo.currentIndex()}"))

    def _quick_browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video", "",
            "Video (*.mp4 *.mov *.avi *.mkv);;All (*)")
        if path:
            self._quick_video.setText(path)

    def _browse_output_into(self, edit, which: str):
        path = QFileDialog.getExistingDirectory(self, "Output folder")
        if path:
            edit.setText(path)
            if which == "video":
                self._video_output_dirty = True
            else:
                self._camera_output_dirty = True

    def _run_quick(self):
        """Launch a run from the active quick mode (UP1r2): a single video via
        ``single_run_spec`` or a live camera via ``camera_run_spec``, then the
        shared one-off launch tail."""
        if self._any_worker_alive():
            # B1 F2: never start a second worker over a live one (shared guard).
            self._append_log(
                "Previous run is still finishing -- try again in a moment.")
            return
        from mindsight.project.staging import camera_run_spec, single_run_spec
        camera = self._mode == "camera"
        output_dir = (self._camera_output if camera
                      else self._video_output).text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Quick analysis",
                                "Choose an output folder.")
            return
        try:
            if camera:
                meta = self._camera_session_meta()
                spec = camera_run_spec(self._camera_combo.currentIndex(),
                                       output_dir, meta=meta)
                if meta:
                    # A run.yaml-shaped record beside the outputs, so a
                    # one-off is later importable into a project (UP5r2).
                    out = Path(output_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / f"{spec.run_id}_session.yaml").write_text(
                        yaml.dump(meta, default_flow_style=False,
                                  sort_keys=False))
            else:
                spec = single_run_spec(self._quick_video.text().strip(),
                                       meta=None, output_dir=output_dir)
        except ValueError as exc:
            QMessageBox.warning(self, "Quick analysis", str(exc))
            return
        self._launch_one_off(spec)

    def _camera_session_meta(self) -> dict:
        """The camera card's optional session details as a Q2 meta dict."""
        meta: dict = {}
        labels = [p.strip() for p in self._cam_participants.text().split(",")
                  if p.strip()]
        if labels:
            meta["participants"] = {i: lab for i, lab in enumerate(labels)}
        tags = [t.strip() for t in self._cam_conditions.text().split("|")
                if t.strip()]
        if tags:
            meta["conditions"] = tags
        if self._cam_session.text().strip():
            meta["session"] = self._cam_session.text().strip()
        if self._cam_notes.text().strip():
            meta["notes"] = self._cam_notes.text().strip()
        return meta

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

    def open_project_browse(self):
        """Public entry (menu bar, UP1 D4): browse for + open a project.

        Thin delegate to the existing browse-open flow -- no logic duplicated."""
        self._open_project_dialog()

    def open_project_path(self, path: str):
        """Public entry (Projects tab, UP3): open *path* as the project."""
        self._project_dir.setText(str(path))
        self._open_project(str(path))

    def new_project(self):
        """Public entry (menu bar, UP1 D4): scaffold + open a new project.

        Thin delegate to the existing new-project flow -- no logic duplicated."""
        self._new_project_dialog()

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

        # Load the project pipeline into the RunSettings store so runs reproduce
        # project numbers.  Decoupling (UP2): Gaze Tuning is NOT updated on
        # project open anymore -- the store is the run-config source.
        self._load_project_pipeline_into_settings()
        self._populate_study_setup()
        self._run_preflight()
        self._refresh_runs_table()
        self._refresh_output_panels()

        self._status_label.setText(f"Open: {project.path.name}")
        self._status_label.setStyleSheet("color: #2a7a2a; font-weight: bold;")
        # UP1r2: opening a project commits the tab to project mode.
        self._set_mode("project")

    def _project_pipeline_path(self) -> Path | None:
        """The project's pipeline.yaml path (for Save-to-project), or None when
        no project is open."""
        if not self._project or not self._project_path:
            return None
        cfg = self._project.config
        if cfg and cfg.pipeline_path:
            return self._project_path / cfg.pipeline_path
        return self._project_path / "Pipeline" / "pipeline.yaml"

    def _open_inference_settings(self):
        """Open the Inference Settings dialog over the shared RunSettings store
        (both quick and project modes -- the store drives every run here)."""
        from .inference_settings import InferenceSettingsDialog
        dlg = InferenceSettingsDialog(
            self._settings, self, gaze_tab=self._gaze_tab,
            project_pipeline_path=self._project_pipeline_path())
        dlg.exec()

    def _load_project_pipeline_into_settings(self):
        if not self._project:
            return
        cfg = self._project.config
        if cfg and cfg.pipeline_path:
            pipe = self._project_path / cfg.pipeline_path
        else:
            pipe = self._project_path / "Pipeline" / "pipeline.yaml"
        if not pipe.is_file():
            return
        try:
            self._settings.apply_yaml(str(pipe), source_label="project pipeline")
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
        # Single seam for preflight, batch runs, per-row re-runs, manual dialog
        # runs, quick runs, camera runs, and weight collection.  Reads the
        # RunSettings store (decoupling ruling); Gaze Tuning no longer feeds it.
        return self._settings.ns()

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
            from mindsight.project.staging import planned_runs
            planned = planned_runs(self._project_path)
        except Exception as exc:
            self._append_log(f"[WARN] could not list runs: {exc}")
            return
        self._run_rows = {}
        self._planned_ids = {p.run_id for p in planned}
        self._runs_table.setRowCount(len(specs) + len(planned))
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
        # UP5: planned sessions -- awaiting a live recording or attached
        # footage; right-click offers both.
        for j, info in enumerate(planned):
            i = len(specs) + j
            pid = (", ".join(f"{k}:{v}"
                             for k, v in (info.meta.pid_map or {}).items())
                   or "—")
            self._set_cell(i, 0, info.run_id)
            self._set_cell(i, 1, "(no video yet)")
            self._set_cell(i, 2, pid)
            self._set_cell(i, 3, ", ".join(info.meta.conditions) or "—")
            self._set_cell(i, 4, "awaiting recording")
            self._set_cell(i, 5, "record live or attach footage")
            self._set_cell(i, 6, "")
            self._set_cell(i, 7, "")

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
        if run_id in getattr(self, "_planned_ids", set()):
            # UP5: a planned session offers its two fulfillment paths.
            menu = QMenu(self)
            record_act = menu.addAction("Record this session...")
            attach_act = menu.addAction("Attach footage...")
            chosen = menu.exec(self._runs_table.viewport().mapToGlobal(pos))
            if chosen == record_act:
                from mindsight.project.staging import planned_runs
                from .record_session_dialog import RecordSessionDialog
                dlg = RecordSessionDialog(planned_runs(self._project_path),
                                          self, preselect=run_id)
                if dlg.exec():
                    self._start_session_recording(dlg.camera_index,
                                                  dlg.run_id, dlg.meta)
            elif chosen == attach_act:
                self._attach_footage(run_id)
            return
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
        # Decoupling: the store drives runs, so re-apply the freshly written
        # project pipeline into it -- the two stay coherent after an import.
        self._load_project_pipeline_into_settings()

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
        # Output toggles (Q7/A3): Events + summary paths are ALWAYS set from the
        # RunSpec; annotated video / heatmap follow the store toggles (ON -> the
        # RunSpec path, OFF -> None).  Charts have no per-run path, so they never
        # write to disk here (the in-GUI Charts tab renders regardless) -- keep
        # today's one-off behavior byte-neutral.
        from .run_settings import want_artifact
        ns.log = spec.output_paths["log"]
        ns.summary = spec.output_paths["summary"]
        ns.save = spec.output_paths["save"] if want_artifact(ns, "save") else None
        ns.heatmap = (spec.output_paths["heatmap"]
                      if want_artifact(ns, "heatmap") else None)
        ns.charts = None
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
        # UP1r2: remember where this one-off writes so the CSV viewer can pick
        # the files up on finish (quick runs have no project discovery).
        self._last_one_off = (spec.run_id, out_dir)
        self._running = True
        self._run_kind = "quick"
        self._update_go_buttons()
        self._charts_hint.setVisible(False)
        self._stop_requested = False
        self._poll_timer.start(60)

    # ── Live session recording (UP5) ─────────────────────────────────────────

    def _record_session_dialog(self):
        if not self._project:
            QMessageBox.information(self, "Record Live Session",
                                    "Open a project first.")
            return
        if self._recorder is not None or self._any_worker_alive():
            self._append_log(
                "A run or recording is already in progress -- finish it "
                "first.")
            return
        from mindsight.project.staging import planned_runs
        from .record_session_dialog import RecordSessionDialog
        dlg = RecordSessionDialog(planned_runs(self._project_path), self)
        if dlg.exec():
            self._start_session_recording(dlg.camera_index, dlg.run_id,
                                          dlg.meta)

    def _start_session_recording(self, camera_index: int, run_id: str,
                                 meta: dict):
        """Begin the raw capture (record-then-analyze, UP5 ruling 1)."""
        from mindsight.io.live_capture import LiveRecorder
        from mindsight.project.staging import _sanitize_run_id
        rid = _sanitize_run_id(run_id)
        tmp = (self._project_path / "Inputs" / "Runs"
               / f"_recording_{rid}.mp4")   # a FILE here is invisible to
        tmp.parent.mkdir(parents=True, exist_ok=True)  # run discovery
        self._recorder = LiveRecorder(str(camera_index), tmp)
        self._recording_meta = (rid, meta)
        self._recorder.start()
        for w in (self._record_btn, self._rerun_all_btn):
            w.setEnabled(False)
        for b in self._mode_btns.values():
            b.setEnabled(False)
        self._update_go_buttons()
        self._append_log(f"⏺ Recording session '{rid}' -- press End Session "
                         "to stop and analyze.")
        self._record_timer.start(200)

    def _recording_tick(self):
        rec = self._recorder
        if rec is None:
            self._record_timer.stop()
            return
        frame = rec.latest_frame()
        if frame is not None:
            pw = self._preview.width() or 480
            ph = self._preview.height() or 320
            self._preview.setPixmap(_bgr_to_pixmap(frame, pw, ph))
        rid, _ = self._recording_meta
        mins, secs = divmod(int(rec.elapsed), 60)
        self._status_label.setText(
            f"⏺ REC {rid} — {mins:02d}:{secs:02d} · "
            f"{rec.frames_captured} frames")
        self._status_label.setStyleSheet("color: #b22222; font-weight: bold;")
        if not rec.is_alive() and rec.error:
            # Camera died / never delivered -- surface it, clean up.
            self._end_session_recording()

    def _end_session_recording(self):
        rec, (rid, meta) = self._recorder, self._recording_meta
        self._record_timer.stop()
        self._recorder = None
        self._recording_meta = None
        rec.stop()
        rec.join(timeout=15)
        for w in (self._record_btn, self._rerun_all_btn):
            w.setEnabled(True)
        for b in self._mode_btns.values():
            b.setEnabled(True)
        self._update_go_buttons()
        self._status_label.setText(f"Open: {self._project_path.name}")
        self._status_label.setStyleSheet("color: #2a7a2a; font-weight: bold;")
        if rec.error:
            QMessageBox.warning(
                self, "Recording failed",
                f"{rec.error}\n\nNothing was saved. Try Refresh in the "
                "Record dialog to pick a different camera.")
            if rec.dest.exists():
                rec.dest.unlink()
            return
        from mindsight.project.staging import attach_recording
        try:
            dest = attach_recording(self._project_path, rid, rec.dest,
                                    sidecar=rec.sidecar, meta=meta,
                                    mode="move")
        except ValueError as exc:
            QMessageBox.warning(
                self, "Could not stage the recording",
                f"{exc}\n\nThe recording is safe at:\n{rec.dest}")
            return
        self._append_log(
            f"Session '{rid}' recorded ({rec.frames_captured} frames, "
            f"{rec.measured_fps:.1f} fps measured) -> {Path(dest).name}. "
            "Starting analysis...")
        self._open_project(str(self._project_path))   # rediscover runs
        self._start()                                  # resume skips done runs

    def _attach_footage(self, run_id: str):
        """UP5r2: fulfill a planned session with footage from another device
        (copied in; the original stays untouched)."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Attach footage for '{run_id}'", "",
            "Video (*.mp4 *.mov *.avi *.mkv);;All (*)")
        if not path:
            return
        from mindsight.project.staging import attach_recording
        try:
            attach_recording(self._project_path, run_id, path, mode="copy")
        except ValueError as exc:
            QMessageBox.warning(self, "Attach footage", str(exc))
            return
        self._append_log(f"Footage attached to session '{run_id}'.")
        self._open_project(str(self._project_path))
        if QMessageBox.question(
                self, "Attach footage",
                f"Footage attached to '{run_id}'. Analyze it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self._start()

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
        self._running = True
        self._run_kind = "project"
        self._update_go_buttons()
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
        # Immediate visible feedback: the red Stop greys out while the current
        # video finalizes; _finish_run re-enables via _update_go_buttons.
        for btn in (self._project_go, self._video_go, self._camera_go):
            btn.setEnabled(False)
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
        was_quick = self._run_kind == "quick"
        self._running = False
        self._run_kind = None
        self._update_go_buttons()
        if self._stop_requested:
            self._append_log("Cancelled.")
        self._stop_requested = False
        self._worker = None
        self._one_off_worker = None
        if self._project:
            self._refresh_runs_table()
            self._refresh_output_panels()
        elif was_quick:
            # UP1r2: surface the finished quick run's CSVs in the viewer.
            self._register_one_off_outputs()

    def _register_one_off_outputs(self):
        """Make a finished quick run's CSVs selectable in the Output CSVs tab
        (quick runs bypass project discovery)."""
        if not getattr(self, "_last_one_off", None):
            return
        run_id, out_dir = self._last_one_off
        # UP4 follow-up (real session): a macOS camera can open but deliver
        # ZERO frames (e.g. a Continuity Camera placeholder) -- the run then
        # "succeeds" with empty outputs and nothing says so. Check the saved
        # recording's frame count and say it plainly.
        video = Path(out_dir) / f"{run_id}_Video_Output.mp4"
        if video.is_file():
            import cv2
            cap = cv2.VideoCapture(str(video))
            frames = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                      if cap.isOpened() else 0)
            cap.release()
            if frames == 0:
                self._append_log(
                    "Note: this run captured NO frames. If the source was a "
                    "camera, it may be a placeholder device -- press Refresh "
                    "to list real cameras, try another one, or check the "
                    "macOS camera permission.")
        from .run_outputs import RunOutputs
        csvs = tuple(sorted(Path(out_dir).glob(f"{run_id}*.csv")))
        if not csvs:
            return
        out = RunOutputs(
            run_id=run_id, stem=run_id, csv_paths=csvs,
            summary=next((p for p in csvs
                          if p.name == f"{run_id}_summary.csv"), None),
            events=next((p for p in csvs
                         if p.name == f"{run_id}_Events.csv"), None))
        self._run_outputs[run_id] = out
        if self._csv_run.findText(run_id) < 0:
            self._csv_run.addItem(run_id)
        self._csv_run.setCurrentText(run_id)
        self._populate_csv_files()

    def _append_log(self, msg: str):
        self._log_box.append(str(msg))
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum())
