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
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
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
        page = QWidget()
        lay = QVBoxLayout(page)

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
        analyze = QPushButton("▶  Open in Analyze Footage")
        analyze.setStyleSheet(_GO_GREEN)
        analyze.setMinimumHeight(34)
        analyze.clicked.connect(
            lambda: self._current and self.open_in_analyze.emit(
                str(self._current)))
        head.addWidget(analyze)
        lay.addLayout(head)

        self._ov_notes = QLabel("")
        self._ov_notes.setStyleSheet("color: #999; font-style: italic;")
        self._ov_notes.setWordWrap(True)
        lay.addWidget(self._ov_notes)

        runs_head = QLabel("Runs")
        runs_head.setStyleSheet("font-weight: bold;")
        lay.addWidget(runs_head)
        self._runs_table = QTableWidget(0, len(_RUN_COLS))
        self._runs_table.setHorizontalHeaderLabels(_RUN_COLS)
        self._runs_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._runs_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._runs_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
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
        return page

    def show_landing(self):
        self._current = None
        self.refresh_landing()
        self._stack.setCurrentIndex(0)

    def show_overview(self, project: Path):
        self._current = Path(project)
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
        try:
            from mindsight.project.project import Project
            proj = Project.open(str(self._current))
            statuses = {s.run_id: s for s in proj.status()}
            for spec in proj.runs():
                st = statuses.get(spec.run_id)
                pid = ", ".join(str(v) for v in (spec.pid_map or {}).values())
                date = str((spec.meta or {}).get("date", "") or "")
                rows.append((spec.run_id, pid, spec.conditions or "",
                             date, (st.status if st else "") or "not run"))
        except Exception as exc:  # noqa: BLE001 -- unreadable project stays viewable
            rows = [("(could not read runs: %s)" % exc, "", "", "", "")]
        self._runs_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self._runs_table.setItem(r, c, QTableWidgetItem(str(val)))

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
