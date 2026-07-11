"""
projects_tab.py
---------------
The **Projects** tab (UP3 / HP3): create and browse projects without knowing
the folder layout.

Two stacked states (UP1r2 principle -- the whole tab commits to one):

- **Landing**: action row (Build New Project... / Create Blank Project... /
  Open Existing...) over a full-height "Your projects" list built from the
  recent-projects history, enriched with on-disk facts (run count, missing
  folders greyed out).
- **Overview**: one project's read-only summary -- runs with their tags and
  ledger status, study notes, and system-default output conveniences.
  Editing and in-GUI viewing live in Analyze Footage (one surface per job);
  the green button jumps there.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_GO_GREEN = ("QPushButton{background:#2a7a2a;color:white;"
             "font-weight:bold;padding:4px 26px;}"
             "QPushButton:disabled{background:#33333f;color:#777;}")
_CARD_BTN = ("QPushButton{padding:10px 22px;font-weight:bold;}")

_PROJECT_COLS = ["Project", "Location", "Runs"]
_RUN_COLS = ["Run", "Participants", "Conditions", "Date", "Status"]


def _count_runs(project: Path) -> int:
    """Cheap on-disk run count for the landing list (either layout)."""
    runs_dir = project / "Inputs" / "Runs"
    if runs_dir.is_dir():
        return sum(1 for p in runs_dir.iterdir() if p.is_dir())
    videos = project / "Inputs" / "Videos"
    if videos.is_dir():
        return sum(1 for p in videos.iterdir() if p.is_file())
    return 0


class ProjectsTab(QWidget):
    """Landing <-> overview; emits ``open_in_analyze(path)`` for main_window."""

    open_in_analyze = pyqtSignal(str)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._current: Path | None = None
        self._run_outputs_map: dict = {}
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_landing())
        self._stack.addWidget(self._build_overview())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(self._stack)
        self.refresh_landing()

    # ── Landing ──────────────────────────────────────────────────────────────

    def _build_landing(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        actions = QHBoxLayout()
        build_btn = QPushButton("⊕  Build New Project...")
        build_btn.setStyleSheet(_GO_GREEN)
        build_btn.setMinimumHeight(44)
        build_btn.setToolTip(
            "Guided setup: videos, participants, conditions, and analysis "
            "settings, step by step")
        build_btn.clicked.connect(self.launch_wizard)
        actions.addWidget(build_btn)
        blank_btn = QPushButton("Create Blank Project...")
        blank_btn.setStyleSheet(_CARD_BTN)
        blank_btn.setMinimumHeight(44)
        blank_btn.setToolTip("An empty project folder to fill in yourself")
        blank_btn.clicked.connect(self._create_blank)
        actions.addWidget(blank_btn)
        actions.addStretch(1)
        open_btn = QPushButton("Open Existing...")
        open_btn.clicked.connect(self._browse_open)
        actions.addWidget(open_btn)
        lay.addLayout(actions)

        head = QLabel("Your projects")
        head.setStyleSheet("font-weight: bold; margin-top: 6px;")
        lay.addWidget(head)

        self._project_table = QTableWidget(0, len(_PROJECT_COLS))
        self._project_table.setHorizontalHeaderLabels(_PROJECT_COLS)
        self._project_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._project_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._project_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._project_table.doubleClicked.connect(
            lambda idx: self._open_row(idx.row()))
        lay.addWidget(self._project_table, 1)

        self._landing_hint = QLabel(
            "No projects yet -- build your first with the wizard above, or "
            "open an existing project folder.")
        self._landing_hint.setStyleSheet("color: #888; font-style: italic;")
        self._landing_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._landing_hint)
        return page

    def refresh_landing(self):
        """Rebuild the project list from the recent-projects history."""
        from .settings_manager import SettingsManager
        recents = SettingsManager().list_recent_projects()
        self._project_table.setRowCount(len(recents))
        for r, path_str in enumerate(recents):
            p = Path(path_str)
            exists = p.is_dir()
            name_item = QTableWidgetItem(
                p.name + ("" if exists else "  (missing)"))
            loc_item = QTableWidgetItem(str(p.parent))
            runs_item = QTableWidgetItem(
                str(_count_runs(p)) if exists else "-")
            for item in (name_item, loc_item, runs_item):
                if not exists:
                    item.setForeground(Qt.GlobalColor.gray)
            name_item.setData(Qt.ItemDataRole.UserRole, path_str)
            self._project_table.setItem(r, 0, name_item)
            self._project_table.setItem(r, 1, loc_item)
            self._project_table.setItem(r, 2, runs_item)
        self._landing_hint.setVisible(not recents)

    def _open_row(self, row: int):
        item = self._project_table.item(row, 0)
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if not path.is_dir():
            QMessageBox.warning(self, "Open project",
                                f"That folder no longer exists:\n{path}")
            return
        self.show_overview(path)

    def _browse_open(self):
        path = QFileDialog.getExistingDirectory(self, "Open a project folder")
        if path:
            self.show_overview(Path(path))

    def _create_blank(self):
        from mindsight.project.runner import create_project
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Folder to create the project inside")
        if not parent_dir:
            return
        name, ok = QInputDialog.getText(self, "Create Blank Project",
                                        "Project name:")
        if not ok or not name.strip():
            return
        try:
            project = create_project(parent_dir, name.strip())
        except ValueError as exc:
            QMessageBox.warning(self, "Create Blank Project", str(exc))
            return
        self._remember(project)
        self.show_overview(project)

    def launch_wizard(self):
        from .project_wizard import BuildProjectWizard
        wiz = BuildProjectWizard(settings=self._settings, parent=self)
        if wiz.exec() and wiz.created_path:
            self._remember(wiz.created_path)
            self.show_overview(wiz.created_path)

    def _remember(self, project: Path):
        from .settings_manager import SettingsManager
        SettingsManager().add_recent_project(str(project))
        self.refresh_landing()

    # ── Overview ─────────────────────────────────────────────────────────────

    def _build_overview(self):
        # Header + notes span the full width; below them the runs area sits
        # beside the data pane (eyes-on r2: the pane must NOT intrude into
        # the header row, and stays hidden until a run is selected).
        page = QWidget()
        page_lay = QVBoxLayout(page)
        from PyQt6.QtWidgets import QSplitter
        ov_split = QSplitter(Qt.Orientation.Horizontal)
        left_col = QWidget()
        lay = QVBoxLayout(left_col)
        lay.setContentsMargins(0, 0, 0, 0)

        head = QHBoxLayout()
        back = QPushButton("‹  All projects")
        back.clicked.connect(self.show_landing)
        head.addWidget(back)
        self._ov_name = QLabel("")
        self._ov_name.setStyleSheet("font-weight: bold; font-size: 15px;")
        head.addWidget(self._ov_name)
        self._ov_path = QLabel("")
        self._ov_path.setStyleSheet("color: #888;")
        head.addWidget(self._ov_path, 1)
        reveal = QPushButton("Reveal Folder")
        reveal.clicked.connect(self._reveal)
        head.addWidget(reveal)
        crop = QPushButton("Crop && Adjust Videos...")
        crop.setToolTip(
            "Crop the raw videos and/or change their frame rate before "
            "running (originals kept unless you say otherwise)")
        crop.clicked.connect(self._open_crop_tool)
        head.addWidget(crop)
        analyze = QPushButton("▶  Open in Analyze Footage")
        analyze.setStyleSheet(_GO_GREEN)
        analyze.setMinimumHeight(34)
        analyze.clicked.connect(
            lambda: self._current and self.open_in_analyze.emit(
                str(self._current)))
        head.addWidget(analyze)
        page_lay.addLayout(head)

        self._ov_notes = QLabel("")
        self._ov_notes.setStyleSheet("color: #999; font-style: italic;")
        self._ov_notes.setWordWrap(True)
        page_lay.addWidget(self._ov_notes)

        runs_head_row = QHBoxLayout()
        runs_head = QLabel("Runs")
        runs_head.setStyleSheet("font-weight: bold;")
        runs_head_row.addWidget(runs_head)
        runs_head_row.addStretch(1)
        plan_btn = QPushButton("＋ Plan Session...")
        plan_btn.setToolTip(
            "Add a future session now -- name and tags only, no footage yet. "
            "It shows as 'awaiting recording' until you record it live or "
            "attach its footage in Analyze Footage (UP5).")
        plan_btn.clicked.connect(self._plan_session)
        runs_head_row.addWidget(plan_btn)
        lay.addLayout(runs_head_row)
        self._runs_table = QTableWidget(0, len(_RUN_COLS))
        self._runs_table.setHorizontalHeaderLabels(_RUN_COLS)
        self._runs_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._runs_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._runs_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._runs_table.itemSelectionChanged.connect(self._on_run_selected)
        lay.addWidget(self._runs_table, 1)

        edit_hint = QLabel(
            "Participants, conditions, and run editing live in Analyze "
            "Footage -- open the project there to change them or view "
            "results in the app.")
        edit_hint.setStyleSheet("color: #888; font-style: italic;")
        edit_hint.setWordWrap(True)
        lay.addWidget(edit_hint)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Outputs:"))
        open_out = QPushButton("Open Outputs Folder")
        open_out.clicked.connect(lambda: self._open_path("Outputs"))
        out_row.addWidget(open_out)
        open_csv = QPushButton("Open CSV Files")
        open_csv.clicked.connect(lambda: self._open_path("Outputs/CSV Files"))
        out_row.addWidget(open_csv)
        open_charts = QPushButton("Open Charts")
        open_charts.clicked.connect(lambda: self._open_path("Outputs/Charts"))
        out_row.addWidget(open_charts)
        out_row.addStretch(1)
        lay.addLayout(out_row)

        from .data_pane import RunDataPane
        self._data_pane = RunDataPane()
        # Hidden until a run is selected -- no dead pane on an empty screen.
        self._data_pane.setVisible(False)
        ov_split.addWidget(left_col)
        ov_split.addWidget(self._data_pane)
        ov_split.setStretchFactor(0, 3)
        ov_split.setStretchFactor(1, 2)
        page_lay.addWidget(ov_split, 1)
        return page

    def _on_run_selected(self):
        row = self._runs_table.currentRow()
        item = self._runs_table.item(row, 0) if row >= 0 else None
        if item is None:
            self._data_pane.setVisible(False)
            return
        run_id = item.text()
        self._data_pane.set_outputs(
            run_id, self._run_outputs_map.get(run_id))
        self._data_pane.setVisible(True)

    def show_landing(self):
        self._current = None
        self.refresh_landing()
        self._stack.setCurrentIndex(0)

    def show_overview(self, project: Path):
        self._current = Path(project)
        self._data_pane.setVisible(False)
        self._runs_table.clearSelection()
        self._remember(self._current)
        self._ov_name.setText(self._current.name)
        self._ov_path.setText(str(self._current))
        notes = self._current / "notes.md"
        self._ov_notes.setText(
            notes.read_text().strip() if notes.is_file() else "")
        self._ov_notes.setVisible(notes.is_file())
        self._refresh_runs()
        self._stack.setCurrentIndex(1)

    def _refresh_runs(self):
        rows = []
        self._run_outputs_map = {}
        try:
            from mindsight.project.project import Project
            from mindsight.project.staging import planned_runs
            from .run_outputs import discover_run_outputs
            proj = Project.open(str(self._current))
            statuses = {s.run_id: s for s in proj.status()}
            self._run_outputs_map = {
                o.run_id: o for o in discover_run_outputs(proj.runs())}
            for spec in proj.runs():
                st = statuses.get(spec.run_id)
                pid = ", ".join(str(v) for v in (spec.pid_map or {}).values())
                date = str((spec.meta or {}).get("date", "") or "")
                rows.append((spec.run_id, pid, spec.conditions or "",
                             date, (st.status if st else "") or "not run"))
            # Planned sessions (UP5): metadata staged, footage still to come.
            for info in planned_runs(self._current):
                pid = ", ".join(
                    str(v) for v in (info.meta.pid_map or {}).values())
                conds = "|".join(info.meta.conditions or [])
                date = str(info.meta.manifest_meta.get("date", "") or "")
                rows.append((info.run_id, pid, conds, date,
                             "awaiting recording"))
        except Exception as exc:  # noqa: BLE001 -- unreadable project stays viewable
            rows = [("(could not read runs: %s)" % exc, "", "", "", "")]
        self._runs_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self._runs_table.setItem(r, c, QTableWidgetItem(str(val)))

    def _plan_session(self):
        """Add a planned session (UP5) to the open project: name + tags now,
        footage recorded live or attached later from Analyze Footage."""
        if not self._current:
            return
        dlg = _PlanSessionDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        from mindsight.project.staging import plan_run
        try:
            plan_run(self._current, dlg.run_id(), meta=dlg.meta())
        except Exception as exc:  # noqa: BLE001 -- plain-English, not a crash
            QMessageBox.warning(self, "Plan session", str(exc))
            return
        self._refresh_runs()

    def _open_crop_tool(self):
        """Launch the Crop & Adjust dialog (UP4); refresh runs afterwards."""
        if not self._current:
            return
        from .crop_dialog import CropVideosDialog
        dlg = CropVideosDialog(self._current, parent=self)
        dlg.exec()
        self._refresh_runs()

    def _reveal(self):
        if self._current:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self._current)))

    def _open_path(self, rel: str):
        if not self._current:
            return
        target = self._current / rel
        if not target.exists():
            QMessageBox.information(
                self, "Outputs",
                "Nothing there yet -- run the project in Analyze Footage "
                "first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


class _PlanSessionDialog(QDialog):
    """Name + tags for a future session -- no footage yet (UP5)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plan a session")
        form = QFormLayout(self)
        hint = QLabel(
            "Define a session before its footage exists. It appears as "
            "'awaiting recording' until you record it live (⏺ Record "
            "Session) or attach its video, both in Analyze Footage.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        form.addRow(hint)
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. session03")
        form.addRow("Session name:", self._name)
        self._participants = QLineEdit()
        self._participants.setPlaceholderText(
            "left to right on screen, comma-separated -- e.g. S80, S81")
        form.addRow("Participants:", self._participants)
        self._conditions = QLineEdit()
        self._conditions.setPlaceholderText(
            "optional -- separate multiple with |")
        form.addRow("Conditions:", self._conditions)
        self._date = QLineEdit()
        self._date.setPlaceholderText("optional -- YYYY-MM-DD")
        form.addRow("Date:", self._date)
        self._session = QLineEdit()
        form.addRow("Session label:", self._session)
        self._notes = QLineEdit()
        form.addRow("Notes:", self._notes)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _accept(self):
        if not self._name.text().strip():
            QMessageBox.warning(self, "Plan session", "Name the session.")
            return
        self.accept()

    def run_id(self) -> str:
        return self._name.text().strip()

    def meta(self) -> dict:
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
