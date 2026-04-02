"""
GUI/main_window.py — Main application window for MindSight.

Hosts three tabs (Gaze Tracker, VP Builder, Project Mode) and provides
the menu bar for presets, pipeline import/export, and application settings.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMenuBar, QMessageBox,
    QPushButton, QTabWidget,
)

# Ensure repo root is on sys.path for imports
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from .gaze_tab import GazeTab
from .vp_builder_tab import VisualPromptBuilderTab


class MainWindow(QMainWindow):
    """Top-level window hosting the Gaze Tracker, VP Builder, and Project Mode tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MindSight — Gaze Tracker + Visual Prompt Builder")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        # ── Tabs ──────────────────────────────────────────────────────────────
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self._gaze_tab = GazeTab()
        self._vp_tab = VisualPromptBuilderTab()
        tabs.addTab(self._gaze_tab, "  Gaze Tracker  ")
        tabs.addTab(self._vp_tab, "  VP Builder  ")

        # Project tab — receives a reference to gaze_tab for namespace building
        from .project_tab import ProjectTab
        self._project_tab = ProjectTab(gaze_tab=self._gaze_tab)
        tabs.addTab(self._project_tab, "  Project Mode  ")

        # ── Menu bar ──────────────────────────────────────────────────────────
        self._build_menu_bar()

        # ── Status bar: "Use VP in Gaze Tracker" button ──────────────────────
        self._use_vp_btn = QPushButton("Use saved VP in Gaze Tracker")
        self._use_vp_btn.setStyleSheet(
            "QPushButton{background:#5a2a7a;color:white;font-weight:bold;"
            "padding:4px 10px;}")
        self._use_vp_btn.clicked.connect(self._push_vp_to_gaze)
        self.statusBar().addPermanentWidget(self._use_vp_btn)

        # ── Initialise plugin panels ─────────────────────────────────────────
        self._init_plugin_panels()

        # ── Restore last session ──────────────────────────────────────────────
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
        """Initialise and embed the phenomena and plugin panels into the gaze tab."""
        try:
            from .phenomena_panel import PhenomenaPanel
            panel = PhenomenaPanel()
            self._gaze_tab.set_phenomena_panel(panel)
        except ImportError:
            pass  # Panel not yet created — will be added in Phase 3

        try:
            from .plugin_panel import PluginPanel
            panel = PluginPanel()
            self._gaze_tab.set_plugin_panel(panel)
        except ImportError:
            pass  # Panel not yet created — will be added in Phase 3

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

    # ── VP transfer ───────────────────────────────────────────────────────────

    def _push_vp_to_gaze(self):
        """Transfer the last saved VP file from the VP Builder to the Gaze tab."""
        path = self._vp_tab.current_vp_path()
        if not path or not Path(path).exists():
            QMessageBox.warning(
                self, "No VP file",
                "Save a VP file in the VP Builder tab first.")
            return
        self._gaze_tab.set_vp_file(path)
        self.centralWidget().setCurrentIndex(0)

    # ── Preset / pipeline stubs ───────────────────────────────────────────────

    def _load_preset(self):
        """Load a saved preset (placeholder — wired in Phase 4)."""
        try:
            from .settings_manager import SettingsManager
            from PyQt6.QtWidgets import QInputDialog
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
            from .settings_manager import SettingsManager
            from PyQt6.QtWidgets import QInputDialog
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
        # Stop project worker
        if hasattr(self._project_tab, '_worker') and self._project_tab._worker:
            if self._project_tab._worker.is_alive():
                self._project_tab._worker.stop()
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
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
