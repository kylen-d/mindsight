"""
GUI/main_window.py — Main application window for MindSight.

Hosts six tabs (Analyze Footage, Projects, VP Builder, Inference Tuning,
Models, About) and provides the menu bar for presets, pipeline
import/export, and application settings.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
)

from ..config_compat import known_good_preset_path, load_pipeline
from .gaze_tab import GazeTab
from .models_tab import ModelsTab
from .run_study_tab import RunStudyTab
from .vp_builder_tab import VisualPromptBuilderTab

# Tab display indices (T11: status-bar button visibility is wired by index).
_TAB_ANALYZE = 0
_TAB_PROJECTS = 1
_TAB_VP = 2
_TAB_TUNING = 3
_TAB_MODELS = 4
_TAB_ABOUT = 5


class MainWindow(QMainWindow):
    """Top-level window: Analyze Footage, Projects, VP Builder,
    Inference Tuning, Models, About."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MindSight")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        # ── Tabs ──────────────────────────────────────────────────────────────
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Gaze Tuning hosts the namespace seam every other tab reads from; build
        # it first so Analyze Footage + Models can hold a reference to it.
        self._gaze_tab = GazeTab()
        self._vp_tab = VisualPromptBuilderTab()
        # UP2 decoupling: one RunSettings store drives every Analyze Footage run
        # (project / quick / camera), independent of Gaze Tuning.
        from .run_settings import RunSettingsStore
        self._settings = RunSettingsStore()
        self._run_study_tab = RunStudyTab(gaze_tab=self._gaze_tab,
                                          settings=self._settings)
        self._models_tab = ModelsTab(gaze_tab=self._gaze_tab)
        # UP3: project creation/browsing home; opening jumps to Analyze Footage.
        from .projects_tab import ProjectsTab
        self._projects_tab = ProjectsTab(settings=self._settings)
        self._projects_tab.open_in_analyze.connect(self._open_project_from_tab)

        # About: program identity + in-app doc reader (eyes-on 2026-07-11).
        from .about_tab import AboutTab
        self._about_tab = AboutTab()

        tabs.addTab(self._run_study_tab, "  Analyze Footage  ")
        tabs.addTab(self._projects_tab, "  Projects  ")
        tabs.addTab(self._vp_tab, "  VP Builder  ")
        tabs.addTab(self._gaze_tab, "  Inference Tuning  ")
        tabs.addTab(self._models_tab, "  Models  ")
        tabs.addTab(self._about_tab, "  About  ")

        # ── Menu bar ──────────────────────────────────────────────────────────
        self._build_menu_bar()

        # ── Status-bar action buttons (per-tab visibility) ────────────────────
        self._build_statusbar_buttons()
        self._tabs = tabs
        tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(0)  # set initial visibility

        # ── Initialise plugin panels ─────────────────────────────────────────
        self._init_plugin_panels()

        # ── Seed defaults, then restore last session ──────────────────────────
        # Ordering contract: widget/schema defaults < preset seed <
        # last_used.json restore < explicit project pipeline load. The preset
        # seed runs FIRST so a saved session (restored next) still wins for
        # every dest it carries.
        self._seed_from_preset()
        self._try_restore_last_session()

    def _build_menu_bar(self):
        """Build the application menu bar."""
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("&File")

        # Project entries (MP1 slice) delegate to the Analyze Footage tab's
        # existing flows -- no logic duplicated (UP1 D4).
        file_menu.addAction("&Build New Project...", self._menu_build_project)
        file_menu.addAction("&New Project...", self._menu_new_project)
        file_menu.addAction("&Open Project...", self._menu_open_project)
        file_menu.addSeparator()
        file_menu.addAction("&Load Preset...", self._load_preset)
        file_menu.addAction("&Save Preset...", self._save_preset)
        file_menu.addSeparator()
        file_menu.addAction("&Import Pipeline YAML...", self._import_pipeline)
        file_menu.addAction("&Export Pipeline YAML...", self._export_pipeline)
        file_menu.addSeparator()
        file_menu.addAction("&Quit", self.close)

        # View menu: theme (eyes-on 2026-07-11 -- auto follows the OS).
        view_menu = menu.addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        from PyQt6.QtGui import QActionGroup
        from .settings_manager import SettingsManager
        from .theming import THEME_MODES
        current = SettingsManager().load_gui_state().get("theme", "auto")
        group = QActionGroup(self)
        for mode in THEME_MODES:
            act = theme_menu.addAction(mode.capitalize())
            act.setCheckable(True)
            act.setChecked(mode == current)
            act.triggered.connect(
                lambda _c, m=mode: self._set_theme(m))
            group.addAction(act)

        # Tools menu (UP2 B2): the Inference Settings dialog entry point.
        tools_menu = menu.addMenu("&Tools")
        tools_menu.addAction("Inference Settings...",
                             self._open_inference_settings)

        # Help menu (UP1 D4)
        help_menu = menu.addMenu("&Help")
        help_menu.addAction("Documentation", self._open_documentation)
        help_menu.addAction("About MindSight", self._show_about)

    # ── Menu handlers: projects + help (UP1 D4) ─────────────────────────────

    #: Published documentation site (mirrors README / docs config).
    _DOCS_URL = "https://kylen-d.github.io/mindsight-docs/"

    def _menu_build_project(self):
        """Switch to Projects and launch the Build New Project wizard (UP3)."""
        self._tabs.setCurrentIndex(_TAB_PROJECTS)
        self._projects_tab.launch_wizard()

    def _open_project_from_tab(self, path: str):
        """Projects tab 'Open in Analyze Footage': switch + open (UP3)."""
        self._tabs.setCurrentIndex(_TAB_ANALYZE)
        self._run_study_tab.open_project_path(path)

    def _menu_new_project(self):
        """Switch to Analyze Footage and start the new-project flow there."""
        self._tabs.setCurrentIndex(_TAB_ANALYZE)
        self._run_study_tab.new_project()

    def _menu_open_project(self):
        """Switch to Analyze Footage and browse-open a project there."""
        self._tabs.setCurrentIndex(_TAB_ANALYZE)
        self._run_study_tab.open_project_browse()

    def _open_inference_settings(self):
        """Switch to Analyze Footage and open the Inference Settings dialog
        there (it owns the shared RunSettings store + project context)."""
        self._tabs.setCurrentIndex(_TAB_ANALYZE)
        self._run_study_tab._open_inference_settings()

    def _open_documentation(self):
        """Open the documentation site in the user's browser."""
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(self._DOCS_URL))

    def _show_about(self):
        """Jump to the About tab (version, logo, in-app guides)."""
        self._tabs.setCurrentIndex(_TAB_ABOUT)

    def _set_theme(self, mode: str):
        """Apply + persist a View > Theme choice."""
        from PyQt6.QtWidgets import QApplication
        from .settings_manager import SettingsManager
        from .theming import apply_theme
        apply_theme(QApplication.instance(), mode)
        SettingsManager().save_gui_state({"theme": mode})

    def _init_plugin_panels(self):
        """Initialise and embed the plugin panel into the Gaze Tuning tab.

        Phenomena controls are now rendered by the schema-generated panel inside
        the Gaze Tuning tab itself; only the introspection-driven plugin panel is
        injected here.
        """
        try:
            from .plugin_panel import PluginPanel
            panel = PluginPanel()
            self._gaze_tab.set_plugin_panel(panel)
        except ImportError:
            pass

    def _seed_from_preset(self):
        """Seed the Gaze Tuning widgets from the shipped known-good preset.

        Bug B2: without this, every phenomena tracker defaults OFF and a default
        run records no phenomena data. The preset (configs/pipeline_known_good.
        yaml) enables all trackers, so seeding it at startup makes the shipped
        defaults record the full phenomena census. Uses the same proven load
        path as the Import Pipeline button (load_pipeline -> apply_namespace).

        Runs BEFORE _try_restore_last_session so a saved session still wins.
        Never fatal: a missing/broken preset must not kill startup.
        """
        path = None
        try:
            path = known_good_preset_path()
            if path is None:
                return
            ns = load_pipeline(str(path), Namespace())
            self._gaze_tab.apply_namespace(ns)
        except Exception as exc:  # noqa: BLE001 -- startup must survive a bad preset
            print(f"[WARN] could not seed from preset {path}: {exc}")

    def _try_restore_last_session(self):
        """Attempt to restore the last-used settings on startup.

        Never fatal: a corrupt/unreadable last_used.json must not kill startup.
        Failures are WARNED (printed), never silently swallowed -- a silent
        except:pass here once hid a real restore regression.
        """
        try:
            from .settings_manager import SettingsManager
            mgr = SettingsManager()
            ns = mgr.load_last_used()
            if ns is not None:
                self._gaze_tab.apply_namespace(ns)
        except Exception as exc:  # noqa: BLE001 -- startup must survive a bad session
            print(f"[WARN] could not restore last session: {exc}")

    # ── Status-bar buttons ─────────────────────────────────────────────────────

    _BTN_GREEN = ("QPushButton{background:#2a7a2a;color:white;"
                  "font-weight:bold;padding:6px 16px;}")
    _BTN_RED   = ("QPushButton{background:#7a2a2a;color:white;"
                  "font-weight:bold;padding:6px 16px;}")
    _BTN_VP    = ("QPushButton{background:#5a2a7a;color:white;"
                  "font-weight:bold;padding:6px 14px;}")

    def _build_statusbar_buttons(self):
        """Create the per-tab status-bar action buttons (T11).

        Preset and pipeline buttons remain in the Gaze Tuning settings panel.
        Each button set is shown/hidden by _on_tab_changed keyed on the new tab
        indices (Analyze Footage 0, VP Builder 1, Gaze Tuning 2, Models 3).
        """
        sb = self.statusBar()
        sb.setStyleSheet("QStatusBar{padding:6px 4px;}")

        # Wire the preset/pipeline buttons that live in the Gaze Tuning tab
        self._gaze_tab._load_preset_btn.clicked.connect(self._load_preset)
        self._gaze_tab._save_preset_btn.clicked.connect(self._save_preset)
        self._gaze_tab._import_pipeline_btn.clicked.connect(self._import_pipeline)
        self._gaze_tab._export_pipeline_btn.clicked.connect(self._export_pipeline)

        # -- Analyze Footage (tab 0): no status-bar buttons -- every mode's
        # primary action is the inline go button in its source card (UP1r3).
        self._analyze_buttons = []

        # -- VP Builder button (tab 1) ----------------------------------------
        self._use_vp_btn = QPushButton("Use saved VP in Inference Tuning")
        self._use_vp_btn.setStyleSheet(self._BTN_VP)
        self._use_vp_btn.clicked.connect(self._push_vp_to_gaze)
        sb.addPermanentWidget(self._use_vp_btn)
        self._vp_buttons = [self._use_vp_btn]

        # -- Gaze Tuning: Start / Stop (tab 2) --------------------------------
        self._gaze_tab._start_btn = QPushButton("\u25b6  Start")
        self._gaze_tab._start_btn.setStyleSheet(self._BTN_GREEN)
        self._gaze_tab._stop_btn = QPushButton("\u25a0  Stop")
        self._gaze_tab._stop_btn.setStyleSheet(self._BTN_RED)
        self._gaze_tab._stop_btn.setEnabled(False)
        self._gaze_tab._start_btn.clicked.connect(self._gaze_tab._start)
        self._gaze_tab._stop_btn.clicked.connect(self._gaze_tab._stop)
        self._gaze_buttons = [
            self._gaze_tab._start_btn,
            self._gaze_tab._stop_btn,
        ]
        for btn in self._gaze_buttons:
            sb.addPermanentWidget(btn)

        # Models (tab 3) has no status-bar action.

    def _on_tab_changed(self, index: int):
        """Show only the buttons relevant to the active tab (T11); the title
        bar follows the tab (eyes-on 2026-07-11)."""
        self.setWindowTitle(f"MindSight — {self._tabs.tabText(index).strip()}")
        for btn in self._analyze_buttons:
            btn.setVisible(index == _TAB_ANALYZE)
        for btn in self._vp_buttons:
            btn.setVisible(index == _TAB_VP)
        for btn in self._gaze_buttons:
            btn.setVisible(index == _TAB_TUNING)

    # ── VP transfer ───────────────────────────────────────────────────────────

    def _push_vp_to_gaze(self):
        """Transfer the last saved VP file from the VP Builder to Gaze Tuning."""
        path = self._vp_tab.current_vp_path()
        if not path or not Path(path).exists():
            QMessageBox.warning(
                self, "No VP file",
                "Save a VP file in the VP Builder tab first.")
            return
        self._gaze_tab.set_vp_file(path)
        self._tabs.setCurrentIndex(_TAB_TUNING)

    # ── Preset / pipeline actions ─────────────────────────────────────────────

    def _load_preset(self):
        """Load a saved preset into the Inference Tuning tab."""
        from PyQt6.QtWidgets import QInputDialog

        from .settings_manager import SettingsManager
        mgr = SettingsManager()
        presets = mgr.list_presets()
        if not presets:
            QMessageBox.information(self, "No Presets", "No saved presets found.")
            return
        name, ok = QInputDialog.getItem(
            self, "Load Preset", "Select preset:", presets, 0, False)
        if ok and name:
            ns = mgr.load_preset(name)
            self._gaze_tab.apply_namespace(ns)

    def _save_preset(self):
        """Save the Inference Tuning tab's current settings as a preset."""
        from PyQt6.QtWidgets import QInputDialog

        from .settings_manager import SettingsManager
        mgr = SettingsManager()
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:")
        if ok and name.strip():
            ns = self._gaze_tab._build_namespace()
            mgr.save_preset(name.strip(), ns)
            QMessageBox.information(
                self, "Saved", f"Preset '{name.strip()}' saved.")

    def _import_pipeline(self):
        """Import settings from a pipeline YAML file."""
        from .pipeline_dialog import import_pipeline
        ns = import_pipeline(self)
        if ns is not None:
            self._gaze_tab.apply_namespace(ns)

    def _export_pipeline(self):
        """Export current settings to a pipeline YAML file."""
        from .pipeline_dialog import export_pipeline
        ns = self._gaze_tab._build_namespace()
        export_pipeline(self, ns)

    # ── Close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Clean up workers and save session on exit.

        UP4 follow-up (found in a real user session): the workers are daemon
        threads, so quitting mid-run killed them before output finalization --
        a live camera run lost its summary CSV. A live run now prompts, and
        "stop and quit" WAITS (bounded) for the workers to finalize."""
        workers = [w for w in (
            getattr(self._gaze_tab, "_worker", None),
            getattr(self._vp_tab, "_vp_worker", None),
            getattr(self._run_study_tab, "_worker", None),
            getattr(self._run_study_tab, "_one_off_worker", None),
            getattr(self._run_study_tab, "_recorder", None),   # UP5 capture
        ) if w is not None and w.is_alive()]
        if workers:
            reply = QMessageBox.question(
                self, "Run in progress",
                "A run or live recording is still in progress. Stop it and "
                "finalize its outputs before quitting?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel)
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            for w in workers:
                w.stop()
            # Finalization (CSV/summary/video writers) happens inside the
            # worker threads; wait for them, but bounded so a hung worker
            # can never trap the user in a window that won't close.
            import time
            deadline = time.monotonic() + 15.0
            while (any(w.is_alive() for w in workers)
                   and time.monotonic() < deadline):
                QApplication.processEvents()
                time.sleep(0.05)
        # Save last-used settings. Never fatal: an unwritable settings dir must
        # not block the window from closing. Failures are WARNED, never silent.
        try:
            from .settings_manager import SettingsManager
            mgr = SettingsManager()
            ns = self._gaze_tab._build_namespace()
            mgr.save_last_used(ns)
        except Exception as exc:  # noqa: BLE001 -- close must never be blocked
            print(f"[WARN] could not save session: {exc}")
        event.accept()


def main():
    """Entry point for the MindSight GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Theme before any window exists; "auto" natively tracks the OS scheme.
    from .settings_manager import SettingsManager
    from .theming import apply_theme
    apply_theme(app, SettingsManager().load_gui_state().get("theme", "auto"))
    icon_path = Path(__file__).resolve().parents[2] / "assets" / "mindsight_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
