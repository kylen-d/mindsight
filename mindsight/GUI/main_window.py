"""
GUI/main_window.py — Main application window for MindSight.

Hosts four tabs (Analyze Footage, VP Builder, Gaze Tuning, Models) and provides
the menu bar for presets, pipeline import/export, and application settings.
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
_TAB_VP = 1
_TAB_TUNING = 2
_TAB_MODELS = 3


class MainWindow(QMainWindow):
    """Top-level window: Analyze Footage, VP Builder, Gaze Tuning, Models."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MindSight — Analyze Footage + Visual Prompt Builder")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        # ── Tabs ──────────────────────────────────────────────────────────────
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Gaze Tuning hosts the namespace seam every other tab reads from; build
        # it first so Analyze Footage + Models can hold a reference to it.
        self._gaze_tab = GazeTab()
        self._vp_tab = VisualPromptBuilderTab()
        self._run_study_tab = RunStudyTab(gaze_tab=self._gaze_tab)
        self._models_tab = ModelsTab(gaze_tab=self._gaze_tab)

        tabs.addTab(self._run_study_tab, "  Analyze Footage  ")
        tabs.addTab(self._vp_tab, "  VP Builder  ")
        tabs.addTab(self._gaze_tab, "  Gaze Tuning  ")
        tabs.addTab(self._models_tab, "  Models  ")

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

        file_menu.addAction("&Load Preset...", self._load_preset)
        file_menu.addAction("&Save Preset...", self._save_preset)
        file_menu.addSeparator()
        file_menu.addAction("&Import Pipeline YAML...", self._import_pipeline)
        file_menu.addAction("&Export Pipeline YAML...", self._export_pipeline)
        file_menu.addSeparator()
        file_menu.addAction("&Quit", self.close)

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
        """Attempt to restore the last-used settings on startup."""
        try:
            from .settings_manager import SettingsManager
            mgr = SettingsManager()
            ns = mgr.load_last_used()
            if ns is not None:
                self._gaze_tab.apply_namespace(ns)
        except (ImportError, Exception):
            pass  # Settings manager not yet created or no saved session

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

        # -- Analyze Footage: Run / Stop (tab 0) ------------------------------
        self._run_study_tab._run_btn = QPushButton("\u25b6  Run")
        self._run_study_tab._run_btn.setStyleSheet(self._BTN_GREEN)
        self._run_study_tab._stop_btn = QPushButton("\u25a0  Stop")
        self._run_study_tab._stop_btn.setStyleSheet(self._BTN_RED)
        self._run_study_tab._stop_btn.setEnabled(False)
        self._run_study_tab._run_btn.clicked.connect(self._run_study_tab._start)
        self._run_study_tab._stop_btn.clicked.connect(self._run_study_tab._stop)
        self._analyze_buttons = [
            self._run_study_tab._run_btn,
            self._run_study_tab._stop_btn,
        ]
        for btn in self._analyze_buttons:
            sb.addPermanentWidget(btn)

        # -- VP Builder button (tab 1) ----------------------------------------
        self._use_vp_btn = QPushButton("Use saved VP in Gaze Tuning")
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
        """Show only the buttons relevant to the active tab (T11)."""
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

    # ── Preset / pipeline stubs ───────────────────────────────────────────────

    def _load_preset(self):
        """Load a saved preset (placeholder — wired in Phase 4)."""
        try:
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
        except ImportError:
            QMessageBox.information(self, "Not Available",
                                   "Settings manager not yet implemented.")

    def _save_preset(self):
        """Save current settings as a preset (placeholder — wired in Phase 4)."""
        try:
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
        except ImportError:
            QMessageBox.information(self, "Not Available",
                                   "Settings manager not yet implemented.")

    def _import_pipeline(self):
        """Import settings from a pipeline YAML file."""
        try:
            from .pipeline_dialog import import_pipeline
            ns = import_pipeline(self)
            if ns is not None:
                self._gaze_tab.apply_namespace(ns)
        except ImportError:
            QMessageBox.information(self, "Not Available",
                                   "Pipeline dialogue not yet implemented.")

    def _export_pipeline(self):
        """Export current settings to a pipeline YAML file."""
        try:
            from .pipeline_dialog import export_pipeline
            ns = self._gaze_tab._build_namespace()
            export_pipeline(self, ns)
        except ImportError:
            QMessageBox.information(self, "Not Available",
                                   "Pipeline dialogue not yet implemented.")

    # ── Close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Clean up workers and save session on exit."""
        # Stop gaze worker
        if hasattr(self._gaze_tab, '_worker') and self._gaze_tab._worker:
            if self._gaze_tab._worker.is_alive():
                self._gaze_tab._worker.stop()
        # Stop VP inference worker
        if hasattr(self._vp_tab, '_vp_worker') and self._vp_tab._vp_worker:
            if self._vp_tab._vp_worker.is_alive():
                self._vp_tab._vp_worker.stop()
        # Stop Analyze Footage (project) worker
        if hasattr(self._run_study_tab, '_worker') and self._run_study_tab._worker:
            if self._run_study_tab._worker.is_alive():
                self._run_study_tab._worker.stop()
        # Save last-used settings
        try:
            from .settings_manager import SettingsManager
            mgr = SettingsManager()
            ns = self._gaze_tab._build_namespace()
            mgr.save_last_used(ns)
        except (ImportError, Exception):
            pass
        event.accept()


def main():
    """Entry point for the MindSight GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    icon_path = Path(__file__).resolve().parents[2] / "assets" / "mindsight_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
