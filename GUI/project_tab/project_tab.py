"""
project_tab.py
--------------
Project batch-processing tab coordinator.  Composes section widgets for
participants and conditions management, and handles pipeline import,
project validation, config building, and start/stop/poll lifecycle.
"""

from __future__ import annotations

import queue
from argparse import Namespace
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..widgets import CollapsibleGroupBox, _bgr_to_pixmap
from .conditions_section import ConditionsSection
from .participants_section import ParticipantsSection


class ProjectTab(QWidget):
    """Batch-process all videos in a MindSight project directory."""

    def __init__(self, gaze_tab=None, parent=None):
        super().__init__(parent)
        self._gaze_tab = gaze_tab
        self._worker = None
        self._progress_q = queue.Queue()
        self._log_q = queue.Queue()
        self._frame_q = queue.Queue(maxsize=2)
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll)
        self._dirty = False
        self._project_path = None
        self._discovered_sources = []
        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Top: project directory + save
        top_row = QHBoxLayout()
        self._project_dir = QLineEdit()
        self._project_dir.setPlaceholderText(
            "Select a MindSight project folder...")
        browse_btn = QPushButton("Browse...")
        browse_btn.setMinimumWidth(60)
        browse_btn.clicked.connect(self._browse_project)
        top_row.addWidget(QLabel("Project:"))
        top_row.addWidget(self._project_dir, stretch=1)
        top_row.addWidget(browse_btn)

        self._save_btn = QPushButton("Save project.yaml")
        self._save_btn.setMinimumWidth(120)
        self._save_btn.clicked.connect(self._save_project_yaml)
        self._save_btn.setEnabled(False)
        top_row.addWidget(self._save_btn)
        outer.addLayout(top_row)

        self._status_label = QLabel("No project selected.")
        self._status_label.setStyleSheet(
            "color: #888; font-style: italic;")
        outer.addWidget(self._status_label)

        # Main splitter: Config (left) | Monitor (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── LEFT: configuration ──────────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)

        # Pipeline section
        pipe_grp = CollapsibleGroupBox("Pipeline")
        pipe_grp.setChecked(True)
        pipe_w = QWidget()
        pipe_lay = QVBoxLayout(pipe_w)
        pipe_lay.setContentsMargins(0, 0, 0, 0)

        pipe_browse_row = QHBoxLayout()
        self._pipeline_path = QLineEdit()
        self._pipeline_path.setPlaceholderText("Pipeline/pipeline.yaml")
        pipe_browse_btn = QPushButton("Browse...")
        pipe_browse_btn.setMinimumWidth(60)
        pipe_browse_btn.clicked.connect(self._browse_pipeline)
        pipe_browse_row.addWidget(self._pipeline_path, stretch=1)
        pipe_browse_row.addWidget(pipe_browse_btn)
        pipe_lay.addLayout(pipe_browse_row)

        self._pipeline_summary = QLabel("No pipeline loaded.")
        self._pipeline_summary.setStyleSheet(
            "color: #aaa; font-size: 11px;")
        self._pipeline_summary.setWordWrap(True)
        pipe_lay.addWidget(self._pipeline_summary)

        import_gaze_btn = QPushButton("Import from Gaze Tab")
        import_gaze_btn.setToolTip(
            "Export the current Gaze Tracker tab settings as this "
            "project's pipeline.yaml")
        import_gaze_btn.clicked.connect(self._import_from_gaze_tab)
        pipe_lay.addWidget(import_gaze_btn)
        pipe_grp.set_content(pipe_w)
        left_lay.addWidget(pipe_grp)

        # Participants section
        part_grp = CollapsibleGroupBox("Participants")
        part_grp.setChecked(True)
        self._participants = ParticipantsSection()
        self._participants.set_dirty_callback(self._mark_dirty)
        part_grp.set_content(self._participants)
        left_lay.addWidget(part_grp)

        # Sources table (right panel, needed by conditions)
        self._source_table = QTableWidget(0, 2)
        self._source_table.setHorizontalHeaderLabels(
            ["Filename", "Conditions"])
        self._source_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._source_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._source_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._source_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._source_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)

        # Conditions section
        cond_grp = CollapsibleGroupBox("Conditions")
        cond_grp.setChecked(True)
        self._conditions = ConditionsSection(
            source_table=self._source_table)
        self._conditions.set_dirty_callback(self._mark_dirty)
        self._conditions.changed.connect(self._update_output_info)
        cond_grp.set_content(self._conditions)
        left_lay.addWidget(cond_grp)

        # Output settings section
        out_grp = CollapsibleGroupBox("Output Settings")
        out_grp.setChecked(True)
        out_w = QWidget()
        out_lay = QFormLayout(out_w)
        out_lay.setContentsMargins(0, 0, 0, 0)

        out_dir_row = QHBoxLayout()
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Default: project/Outputs")
        self._output_dir.textChanged.connect(self._mark_dirty)
        out_browse_btn = QPushButton("Browse...")
        out_browse_btn.setMinimumWidth(60)
        out_browse_btn.clicked.connect(self._browse_output_dir)
        out_dir_row.addWidget(self._output_dir, stretch=1)
        out_dir_row.addWidget(out_browse_btn)
        out_lay.addRow("Output Root:", out_dir_row)

        self._output_resolved = QLabel("")
        self._output_resolved.setStyleSheet(
            "color: #888; font-size: 11px;")
        self._output_resolved.setWordWrap(True)
        out_lay.addRow(self._output_resolved)

        self._output_info = QLabel("")
        self._output_info.setStyleSheet(
            "color: #aaa; font-size: 11px;")
        self._output_info.setWordWrap(True)
        out_lay.addRow("Will generate:", self._output_info)
        out_grp.set_content(out_w)
        left_lay.addWidget(out_grp)

        left_lay.addStretch()
        splitter.addWidget(left)

        # ── RIGHT: monitoring ────────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(6)

        src_grp = QGroupBox("Sources (Inputs/Videos/)")
        src_lay = QVBoxLayout(src_grp)
        src_lay.addWidget(self._source_table)
        right_lay.addWidget(src_grp, stretch=2)

        self._preview = QLabel()
        self._preview.setStyleSheet("background: #1a1a2e;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(200, 120)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
        right_lay.addWidget(self._preview, stretch=1)

        prog_grp = QGroupBox("Progress")
        prog_lay = QVBoxLayout(prog_grp)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        prog_lay.addWidget(self._progress_bar)
        self._current_file = QLabel("Idle")
        self._current_file.setStyleSheet("color: #aaa;")
        prog_lay.addWidget(self._current_file)
        right_lay.addWidget(prog_grp)

        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(60)
        self._log_box.setFont(QFont("Courier", 10))
        log_lay.addWidget(self._log_box)
        right_lay.addWidget(log_grp, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter, stretch=1)

    # ── Browse & validate ───────────────────────────────────────────────────

    def _browse_project(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select MindSight project folder")
        if path:
            self._project_dir.setText(path)
            self._validate_project(path)

    def _browse_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pipeline YAML", "",
            "YAML files (*.yaml *.yml)")
        if path:
            if self._project_path:
                try:
                    rel = Path(path).relative_to(self._project_path)
                    self._pipeline_path.setText(str(rel))
                except ValueError:
                    self._pipeline_path.setText(path)
            else:
                self._pipeline_path.setText(path)
            self._update_pipeline_summary()
            self._mark_dirty()

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select output directory")
        if path:
            self._output_dir.setText(path)
            self._update_output_resolved()

    def _validate_project(self, path_str: str):
        """Validate the project directory and populate all UI panels."""
        try:
            from project_runner import (
                discover_sources,
                load_project_config,
                validate_project,
            )

            project = Path(path_str).resolve()
            if not project.is_dir():
                raise FileNotFoundError(f"Not a directory: {project}")
            if not (project / "Inputs" / "Videos").is_dir():
                raise ValueError("Missing Inputs/Videos/ directory")

            self._project_path = project
            sources = discover_sources(project)
            self._discovered_sources = sources

            project_cfg = load_project_config(project)
            validate_project(path_str, project_cfg)

            # Sources table
            self._source_table.setRowCount(len(sources))
            for i, src in enumerate(sources):
                self._source_table.setItem(
                    i, 0, QTableWidgetItem(src.name))
                tags = ""
                if (project_cfg
                        and src.name in project_cfg.conditions):
                    tags = " | ".join(
                        project_cfg.conditions[src.name])
                self._source_table.setItem(
                    i, 1, QTableWidgetItem(tags))

            # Pipeline
            if project_cfg and project_cfg.pipeline_path:
                self._pipeline_path.setText(
                    project_cfg.pipeline_path)
            else:
                pipeline_yaml = (
                    project / "Pipeline" / "pipeline.yaml")
                if pipeline_yaml.exists():
                    self._pipeline_path.setText(
                        "Pipeline/pipeline.yaml")
                else:
                    self._pipeline_path.setText("")
            self._update_pipeline_summary()

            # Delegate to section widgets
            self._participants.set_sources(sources)
            self._participants.populate(
                project_cfg, sources, project_path=project)
            self._conditions.populate(project_cfg, sources)

            # Output settings
            if project_cfg and project_cfg.output.directory:
                self._output_dir.setText(
                    project_cfg.output.directory)
            else:
                self._output_dir.setText("")
            self._update_output_resolved()
            self._update_output_info()

            # Status
            colour = "#2a7a2a" if sources else "#7a2a2a"
            msg = f"Valid project: {len(sources)} source(s)"
            if project_cfg:
                msg += " (project.yaml loaded)"
            self._status_label.setText(msg)
            self._status_label.setStyleSheet(
                f"color: {colour}; font-weight: bold;")

            self._save_btn.setEnabled(True)
            self._dirty = False

        except (FileNotFoundError, ValueError) as e:
            self._status_label.setText(f"Invalid: {e}")
            self._status_label.setStyleSheet(
                "color: #7a2a2a; font-weight: bold;")
            self._source_table.setRowCount(0)
            self._project_path = None
            self._discovered_sources = []

    # ── Pipeline import & summary ───────────────────────────────────────────

    def _import_from_gaze_tab(self):
        if not self._gaze_tab:
            QMessageBox.warning(
                self, "Error", "Gaze Tab not available.")
            return
        if not self._project_path:
            QMessageBox.warning(
                self, "Error", "Load a project first.")
            return

        ns = self._gaze_tab._build_namespace()
        from ..pipeline_dialog import _namespace_to_yaml_dict
        yaml_dict = _namespace_to_yaml_dict(ns)

        pipe_text = self._pipeline_path.text().strip()
        if not pipe_text:
            pipe_text = "Pipeline/pipeline.yaml"
        target = Path(pipe_text)
        if not target.is_absolute():
            target = self._project_path / target

        if target.exists():
            reply = QMessageBox.question(
                self, "Overwrite Pipeline?",
                f"{target.name} already exists.\n"
                "Overwrite with Gaze Tab settings?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as fh:
            yaml.dump(yaml_dict, fh, default_flow_style=False,
                      sort_keys=False)

        self._pipeline_path.setText(pipe_text)
        self._update_pipeline_summary()
        self._mark_dirty()
        self._status_label.setText(
            f"Imported Gaze Tab settings to {target.name}")
        self._status_label.setStyleSheet(
            "color: #2a7a2a; font-weight: bold;")

    def _update_pipeline_summary(self):
        path_text = self._pipeline_path.text().strip()
        if not path_text or not self._project_path:
            self._pipeline_summary.setText("No pipeline loaded.")
            return

        yaml_path = Path(path_text)
        if not yaml_path.is_absolute():
            yaml_path = self._project_path / yaml_path

        if not yaml_path.is_file():
            self._pipeline_summary.setText(
                f"File not found: {path_text}")
            return

        try:
            with open(yaml_path) as fh:
                cfg = yaml.safe_load(fh) or {}

            parts = []
            det = cfg.get("detection", {})
            if det.get("model"):
                parts.append(f"Model: {det['model']}")
            if det.get("conf"):
                parts.append(f"Conf: {det['conf']}")
            if det.get("classes"):
                cls_str = det["classes"]
                if isinstance(cls_str, list):
                    cls_str = ", ".join(cls_str)
                parts.append(f"Classes: {cls_str}")

            gaze = cfg.get("gaze", {})
            if gaze:
                gaze_parts = []
                if gaze.get("ray_length"):
                    gaze_parts.append(
                        f"ray={gaze['ray_length']}")
                if gaze.get("adaptive_ray"):
                    gaze_parts.append(
                        f"adaptive={gaze['adaptive_ray']}")
                if gaze_parts:
                    parts.append(
                        f"Gaze: {', '.join(gaze_parts)}")

            phenomena = cfg.get("phenomena", [])
            if isinstance(phenomena, list) and phenomena:
                names = []
                for item in phenomena:
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, dict):
                        names.extend(item.keys())
                parts.append(
                    f"Phenomena ({len(names)}): "
                    f"{', '.join(names)}")

            self._pipeline_summary.setText(
                " | ".join(parts) if parts
                else "Pipeline loaded (no notable settings)")
        except Exception as e:
            self._pipeline_summary.setText(
                f"Error reading pipeline: {e}")

    def _update_output_resolved(self):
        if not self._project_path:
            self._output_resolved.setText("")
            return
        text = self._output_dir.text().strip()
        if text:
            p = Path(text)
            if not p.is_absolute():
                p = self._project_path / p
            self._output_resolved.setText(f"Resolved: {p}")
        else:
            self._output_resolved.setText(
                f"Default: {self._project_path / 'Outputs'}")

    def _update_output_info(self):
        n = len(self._discovered_sources)
        if n == 0:
            self._output_info.setText("No sources loaded")
            return
        tags = self._conditions.get_all_tags()
        lines = [f"Per-video: {n} Summary + {n} Events CSVs"]
        lines.append(
            "Global: Global_Summary.csv + Global_Events.csv")
        if tags:
            lines.append(
                f"By Condition: {len(tags)} tag(s) \u2192 "
                f"{len(tags) * 2} condition CSVs")
        else:
            lines.append("By Condition: No conditions defined")
        self._output_info.setText("\n".join(lines))

    # ── Save project.yaml ───────────────────────────────────────────────────

    def _build_project_config(self):
        from pipeline_config import ProjectConfig, ProjectOutputConfig
        pipe = self._pipeline_path.text().strip() or None
        conditions = self._conditions.get_conditions()
        participants = self._participants.get_participants()
        out_dir = self._output_dir.text().strip() or None
        output_cfg = ProjectOutputConfig(directory=out_dir)
        return ProjectConfig(
            pipeline_path=pipe,
            conditions=conditions,
            participants=participants,
            output=output_cfg,
        )

    def _save_project_yaml(self):
        if not self._project_path:
            QMessageBox.warning(
                self, "Error", "No project loaded.")
            return
        from project_runner import save_project_config
        cfg = self._build_project_config()
        path = save_project_config(self._project_path, cfg)
        self._dirty = False
        self._save_btn.setText("Save project.yaml")
        self._status_label.setText(
            f"Saved: {path.name}  "
            f"({len(self._discovered_sources)} source(s))")
        self._status_label.setStyleSheet(
            "color: #2a7a2a; font-weight: bold;")

    def _mark_dirty(self, *_args):
        if not self._dirty:
            self._dirty = True
            self._save_btn.setText("Save project.yaml *")

    # ── Start / Stop / Poll ─────────────────────────────────────────────────

    def _start(self):
        if self._worker and self._worker.is_alive():
            return
        project_dir = self._project_dir.text().strip()
        if not project_dir:
            QMessageBox.critical(
                self, "Error",
                "Select a project directory first.")
            return

        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Save project.yaml before running?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._save_project_yaml()

        if self._gaze_tab:
            ns = self._gaze_tab._build_namespace()
        else:
            ns = Namespace()
        ns.project = project_dir

        project_cfg = self._build_project_config()

        self._progress_q = queue.Queue()
        self._log_q = queue.Queue()
        self._frame_q = queue.Queue(maxsize=2)

        from ..workers import ProjectWorker
        self._worker = ProjectWorker(
            project_dir, ns, self._progress_q, self._log_q,
            self._frame_q, project_cfg=project_cfg)
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
        try:
            while True:
                msg = self._log_q.get_nowait()
                self._log_box.append(msg)
                self._log_box.verticalScrollBar().setValue(
                    self._log_box.verticalScrollBar().maximum())
        except queue.Empty:
            pass

        try:
            frame = None
            while True:
                f = self._frame_q.get_nowait()
                if f is None:
                    break
                frame = f
            if frame is not None:
                self._preview.setPixmap(
                    _bgr_to_pixmap(frame, 320, 240))
        except queue.Empty:
            pass

        try:
            while True:
                event = self._progress_q.get_nowait()
                if event is None:
                    self._poll_timer.stop()
                    self._run_btn.setEnabled(True)
                    self._stop_btn.setEnabled(False)
                    self._current_file.setText("Done.")
                    self._progress_bar.setValue(100)
                    return
                if event["type"] == "start":
                    self._progress_bar.setRange(
                        0, event["total"])
                    self._progress_bar.setValue(0)
                elif event["type"] == "progress":
                    self._progress_bar.setValue(event["current"])
                    self._current_file.setText(
                        f"[{event['current']}/{event['total']}] "
                        f"{event['source_name']}")
                elif event["type"] == "done":
                    self._progress_bar.setValue(
                        self._progress_bar.maximum())
                    self._current_file.setText(
                        "All sources processed.")
                elif event["type"] == "error":
                    self._current_file.setText(
                        f"Error: {event['message']}")
        except queue.Empty:
            pass

        if self._worker and not self._worker.is_alive():
            self._poll_timer.stop()
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
