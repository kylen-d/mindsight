"""
GUI/project_tab.py — Project batch-processing tab.

Provides a UI for selecting a MindSight project directory, viewing its
contents, and batch-processing all videos with progress tracking.
"""
from __future__ import annotations

import queue
from argparse import Namespace
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMessageBox, QProgressBar, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)

from .widgets import _hrow, _browse_btn, _bgr_to_pixmap


class ProjectTab(QWidget):
    """Batch-process all videos in a MindSight project directory."""

    def __init__(self, gaze_tab=None, parent=None):
        super().__init__(parent)
        self._gaze_tab = gaze_tab  # reference to GazeTab for namespace building
        self._worker = None
        self._progress_q = queue.Queue()
        self._log_q = queue.Queue()
        self._frame_q = queue.Queue(maxsize=2)
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Project directory ─────────────────────────────────────────────
        dir_grp = QGroupBox("Project Directory")
        dir_lay = QFormLayout(dir_grp)

        self._project_dir = QLineEdit()
        self._project_dir.setPlaceholderText("Select a MindSight project folder...")
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_project)
        dir_lay.addRow("Project:", _hrow(self._project_dir, browse_btn))

        self._status_label = QLabel("No project selected.")
        self._status_label.setStyleSheet("color: #888; font-style: italic;")
        dir_lay.addRow(self._status_label)
        outer.addWidget(dir_grp)

        # ── Project contents (horizontal split) ──────────────────────────
        content = QWidget()
        content_lay = QHBoxLayout(content)
        content_lay.setContentsMargins(0, 0, 0, 0)

        # Left: source list
        src_grp = QGroupBox("Sources (Inputs/Videos/)")
        src_lay = QVBoxLayout(src_grp)
        self._source_list = QListWidget()
        src_lay.addWidget(self._source_list)
        content_lay.addWidget(src_grp, stretch=1)

        # Right: project info
        info_grp = QGroupBox("Project Info")
        info_lay = QFormLayout(info_grp)

        self._pipeline_label = QLabel("—")
        info_lay.addRow("Pipeline:", self._pipeline_label)

        self._vp_label = QLabel("—")
        info_lay.addRow("VP File:", self._vp_label)

        self._source_count = QLabel("0")
        info_lay.addRow("Sources:", self._source_count)

        content_lay.addWidget(info_grp, stretch=1)

        # Preview (small)
        self._preview = QLabel()
        self._preview.setStyleSheet("background: #1a1a2e;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setFixedSize(320, 240)
        content_lay.addWidget(self._preview)

        outer.addWidget(content, stretch=1)

        # ── Progress ─────────────────────────────────────────────────────
        prog_grp = QGroupBox("Progress")
        prog_lay = QVBoxLayout(prog_grp)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        prog_lay.addWidget(self._progress_bar)

        self._current_file = QLabel("Idle")
        self._current_file.setStyleSheet("color: #aaa;")
        prog_lay.addWidget(self._current_file)

        outer.addWidget(prog_grp)

        # ── Log ──────────────────────────────────────────────────────────
        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setFixedHeight(150)
        self._log_box.setFont(QFont("Courier", 10))
        log_lay.addWidget(self._log_box)
        outer.addWidget(log_grp)

        # ── Run / Stop ───────────────────────────────────────────────────
        btn_row = _hrow()
        self._run_btn = QPushButton("▶  Run Project")
        self._run_btn.setStyleSheet(
            "QPushButton{background:#2a7a2a;color:white;font-weight:bold;padding:6px;}")
        self._run_btn.clicked.connect(self._start)
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setStyleSheet(
            "QPushButton{background:#7a2a2a;color:white;font-weight:bold;padding:6px;}")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        btn_row.layout().addWidget(self._run_btn, 1)
        btn_row.layout().addWidget(self._stop_btn, 1)
        outer.addWidget(btn_row)

    # ── Browse & validate ─────────────────────────────────────────────────

    def _browse_project(self):
        path = QFileDialog.getExistingDirectory(self, "Select MindSight project folder")
        if path:
            self._project_dir.setText(path)
            self._validate_project(path)

    def _validate_project(self, path_str: str):
        """Validate the project directory and populate the UI."""
        try:
            from project_runner import validate_project, discover_sources, discover_vp_file

            project = validate_project(path_str)
            sources = discover_sources(project)
            vp = discover_vp_file(project)
            pipeline_yaml = project / "Pipeline" / "pipeline.yaml"

            self._source_list.clear()
            for src in sources:
                self._source_list.addItem(src.name)

            self._source_count.setText(str(len(sources)))
            self._pipeline_label.setText(
                pipeline_yaml.name if pipeline_yaml.exists() else "Not found")
            self._vp_label.setText(Path(vp).name if vp else "Not found")

            colour = "#2a7a2a" if sources else "#7a2a2a"
            msg = f"Valid project: {len(sources)} source(s)" if sources else "No sources found"
            self._status_label.setText(msg)
            self._status_label.setStyleSheet(f"color: {colour}; font-weight: bold;")

        except (FileNotFoundError, ValueError) as e:
            self._status_label.setText(f"Invalid: {e}")
            self._status_label.setStyleSheet("color: #7a2a2a; font-weight: bold;")
            self._source_list.clear()
            self._source_count.setText("0")

    # ── Start / Stop / Poll ───────────────────────────────────────────────

    def _start(self):
        if self._worker and self._worker.is_alive():
            return
        project_dir = self._project_dir.text().strip()
        if not project_dir:
            QMessageBox.critical(self, "Error", "Select a project directory first.")
            return

        # Build namespace from gaze tab settings (if available)
        if self._gaze_tab:
            ns = self._gaze_tab._build_namespace()
        else:
            ns = Namespace()

        # Ensure project-relevant attributes exist
        ns.project = project_dir

        self._progress_q = queue.Queue()
        self._log_q = queue.Queue()
        self._frame_q = queue.Queue(maxsize=2)

        from .workers import ProjectWorker
        self._worker = ProjectWorker(
            project_dir, ns, self._progress_q, self._log_q, self._frame_q)
        self._worker.start()

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        self._current_file.setText("Starting...")
        self._log_box.clear()
        self._poll_timer.start(100)

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self._poll_timer.stop()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._current_file.setText("Stopped.")

    def _poll(self):
        # Drain log queue
        try:
            while True:
                msg = self._log_q.get_nowait()
                self._log_box.append(msg)
                self._log_box.verticalScrollBar().setValue(
                    self._log_box.verticalScrollBar().maximum())
        except queue.Empty:
            pass

        # Drain frame queue (show latest frame in preview)
        try:
            frame = None
            while True:
                f = self._frame_q.get_nowait()
                if f is None:
                    break
                frame = f
            if frame is not None:
                self._preview.setPixmap(_bgr_to_pixmap(frame, 320, 240))
        except queue.Empty:
            pass

        # Drain progress queue
        try:
            while True:
                event = self._progress_q.get_nowait()
                if event is None:
                    # Sentinel: worker done
                    self._poll_timer.stop()
                    self._run_btn.setEnabled(True)
                    self._stop_btn.setEnabled(False)
                    self._current_file.setText("Done.")
                    self._progress_bar.setValue(100)
                    return
                if event["type"] == "start":
                    self._progress_bar.setRange(0, event["total"])
                    self._progress_bar.setValue(0)
                elif event["type"] == "progress":
                    self._progress_bar.setValue(event["current"])
                    self._current_file.setText(
                        f"[{event['current']}/{event['total']}] {event['source_name']}")
                elif event["type"] == "done":
                    self._progress_bar.setValue(self._progress_bar.maximum())
                    self._current_file.setText("All sources processed.")
                elif event["type"] == "error":
                    self._current_file.setText(f"Error: {event['message']}")
        except queue.Empty:
            pass

        # Check if worker died
        if self._worker and not self._worker.is_alive():
            self._poll_timer.stop()
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
