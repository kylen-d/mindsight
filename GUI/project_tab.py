"""
GUI/project_tab.py — Project batch-processing tab.

Provides a UI for selecting a MindSight project directory, configuring
study metadata (conditions, participants, output settings), viewing a
read-only pipeline summary, and batch-processing all videos with progress
tracking.
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
    QComboBox,
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
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .widgets import CollapsibleGroupBox, _bgr_to_pixmap


class _VideoComboDelegate(QStyledItemDelegate):
    """Dropdown delegate for the Video column in the participants table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_names: list[str] = []

    def set_video_names(self, names: list[str]):
        self._video_names = list(names)

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self._video_names)
        combo.setEditable(True)
        return combo

    def setEditorData(self, editor, index):
        current = index.data()
        idx = editor.findText(current or "")
        if idx >= 0:
            editor.setCurrentIndex(idx)
        elif current:
            editor.setEditText(current)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())


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
        self._dirty = False           # unsaved changes indicator
        self._project_path = None     # resolved project Path
        self._discovered_sources = [] # list of Path objects
        self._video_delegate = _VideoComboDelegate(self)
        self._build_ui()

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ── Top: project directory + status + save ───────────────────────
        top_row = QHBoxLayout()

        self._project_dir = QLineEdit()
        self._project_dir.setPlaceholderText("Select a MindSight project folder...")
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
        self._status_label.setStyleSheet("color: #888; font-style: italic;")
        outer.addWidget(self._status_label)

        # ── Main splitter: Config (left) | Monitor (right) ──────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── LEFT: configuration panel ────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)

        # Pipeline section
        pipe_grp = CollapsibleGroupBox("Pipeline")
        pipe_grp.setChecked(True)
        pipe_lay = QVBoxLayout()

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
        self._pipeline_summary.setStyleSheet("color: #aaa; font-size: 11px;")
        self._pipeline_summary.setWordWrap(True)
        pipe_lay.addWidget(self._pipeline_summary)

        import_gaze_btn = QPushButton("Import from Gaze Tab")
        import_gaze_btn.setToolTip(
            "Export the current Gaze Tracker tab settings as this project's pipeline.yaml")
        import_gaze_btn.clicked.connect(self._import_from_gaze_tab)
        pipe_lay.addWidget(import_gaze_btn)

        pipe_grp.setLayout(pipe_lay)
        left_lay.addWidget(pipe_grp)

        # Participants section
        part_grp = CollapsibleGroupBox("Participants")
        part_grp.setChecked(True)
        part_lay = QVBoxLayout()

        self._part_table = QTableWidget(0, 3)
        self._part_table.setHorizontalHeaderLabels(["Video", "Track ID", "Label"])
        self._part_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._part_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self._part_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self._part_table.setItemDelegateForColumn(0, self._video_delegate)
        self._part_table.setMinimumHeight(100)
        self._part_table.cellChanged.connect(self._mark_dirty)
        part_lay.addWidget(self._part_table)

        part_btn_row = QHBoxLayout()
        add_part_btn = QPushButton("+ Add Row")
        add_part_btn.clicked.connect(self._add_participant_row)
        rm_part_btn = QPushButton("- Remove Row")
        rm_part_btn.clicked.connect(self._remove_participant_row)
        auto_pop_btn = QPushButton("Auto-populate")
        auto_pop_btn.setToolTip("Create one P0 row per discovered video")
        auto_pop_btn.clicked.connect(self._auto_populate_participants)
        bulk_add_btn = QPushButton("+ Track to All")
        bulk_add_btn.setToolTip("Add a new participant track to every video")
        bulk_add_btn.clicked.connect(self._bulk_add_participant)
        part_btn_row.addWidget(add_part_btn)
        part_btn_row.addWidget(rm_part_btn)
        part_btn_row.addWidget(auto_pop_btn)
        part_btn_row.addWidget(bulk_add_btn)
        part_btn_row.addStretch()
        part_lay.addLayout(part_btn_row)

        part_grp.setLayout(part_lay)
        left_lay.addWidget(part_grp)

        # Conditions section
        cond_grp = CollapsibleGroupBox("Conditions")
        cond_grp.setChecked(True)
        cond_lay = QVBoxLayout()

        self._cond_table = QTableWidget(0, 2)
        self._cond_table.setHorizontalHeaderLabels(["Video", "Conditions"])
        self._cond_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._cond_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._cond_table.setMinimumHeight(100)
        self._cond_table.cellChanged.connect(self._mark_dirty)
        self._cond_table.cellChanged.connect(self._on_cond_table_changed)
        cond_lay.addWidget(self._cond_table)

        cond_hint = QLabel("Edit cells directly, or select rows and use buttons below. "
                          "Separate multiple tags with |")
        cond_hint.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        cond_hint.setWordWrap(True)
        cond_lay.addWidget(cond_hint)

        cond_action_row = QHBoxLayout()
        self._cond_tag_input = QLineEdit()
        self._cond_tag_input.setPlaceholderText("Tag to apply/remove...")
        apply_btn = QPushButton("Apply to Selected")
        apply_btn.setToolTip("Add this tag to all selected rows")
        apply_btn.clicked.connect(self._apply_condition_to_selected)
        clear_tag_btn = QPushButton("Remove Tag")
        clear_tag_btn.setToolTip("Remove this specific tag from selected rows")
        clear_tag_btn.clicked.connect(self._remove_condition_from_selected)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setToolTip("Clear all tags from selected rows")
        clear_all_btn.clicked.connect(self._clear_conditions_from_selected)
        cond_action_row.addWidget(self._cond_tag_input, stretch=1)
        cond_action_row.addWidget(apply_btn)
        cond_action_row.addWidget(clear_tag_btn)
        cond_action_row.addWidget(clear_all_btn)
        cond_lay.addLayout(cond_action_row)

        cond_grp.setLayout(cond_lay)
        left_lay.addWidget(cond_grp)

        # Output settings section
        out_grp = CollapsibleGroupBox("Output Settings")
        out_grp.setChecked(True)
        out_lay = QFormLayout()

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
        self._output_resolved.setStyleSheet("color: #888; font-size: 11px;")
        self._output_resolved.setWordWrap(True)
        out_lay.addRow(self._output_resolved)

        self._output_info = QLabel("")
        self._output_info.setStyleSheet("color: #aaa; font-size: 11px;")
        self._output_info.setWordWrap(True)
        out_lay.addRow("Will generate:", self._output_info)

        out_grp.setLayout(out_lay)
        left_lay.addWidget(out_grp)

        left_lay.addStretch()
        splitter.addWidget(left)

        # ── RIGHT: monitoring panel ──────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(6)

        # Sources table
        src_grp = QGroupBox("Sources (Inputs/Videos/)")
        src_lay = QVBoxLayout(src_grp)
        self._source_table = QTableWidget(0, 2)
        self._source_table.setHorizontalHeaderLabels(["Filename", "Conditions"])
        self._source_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._source_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._source_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._source_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._source_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        src_lay.addWidget(self._source_table)
        right_lay.addWidget(src_grp, stretch=2)

        # Preview
        self._preview = QLabel()
        self._preview.setStyleSheet("background: #1a1a2e;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(200, 120)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_lay.addWidget(self._preview, stretch=1)

        # Progress
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

        # Log
        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(60)
        self._log_box.setFont(QFont("Courier", 10))
        log_lay.addWidget(self._log_box)
        right_lay.addWidget(log_grp, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)  # left config ~40%
        splitter.setStretchFactor(1, 3)  # right monitor ~60%

        outer.addWidget(splitter, stretch=1)

    # ══════════════════════════════════════════════════════════════════════════
    # Browse & validate
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_project(self):
        path = QFileDialog.getExistingDirectory(self, "Select MindSight project folder")
        if path:
            self._project_dir.setText(path)
            self._validate_project(path)

    def _browse_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pipeline YAML", "", "YAML files (*.yaml *.yml)")
        if path:
            if self._project_path:
                # Store relative to project root if possible
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
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
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

            # Update video name dropdown delegate
            self._video_delegate.set_video_names([s.name for s in sources])

            # Load existing project.yaml (may be None)
            project_cfg = load_project_config(project)

            # Validate (creates output dirs)
            validate_project(path_str, project_cfg)

            # ── Sources table ────────────────────────────────────────────
            self._source_table.setRowCount(len(sources))
            for i, src in enumerate(sources):
                self._source_table.setItem(i, 0, QTableWidgetItem(src.name))
                tags = ""
                if project_cfg and src.name in project_cfg.conditions:
                    tags = " | ".join(project_cfg.conditions[src.name])
                self._source_table.setItem(i, 1, QTableWidgetItem(tags))

            # ── Pipeline ─────────────────────────────────────────────────
            if project_cfg and project_cfg.pipeline_path:
                self._pipeline_path.setText(project_cfg.pipeline_path)
            else:
                pipeline_yaml = project / "Pipeline" / "pipeline.yaml"
                if pipeline_yaml.exists():
                    self._pipeline_path.setText("Pipeline/pipeline.yaml")
                else:
                    self._pipeline_path.setText("")
            self._update_pipeline_summary()

            # ── Participants table ───────────────────────────────────────
            self._populate_participants(project_cfg, sources)

            # ── Conditions table ─────────────────────────────────────────
            self._populate_conditions(project_cfg, sources)

            # ── Output settings ──────────────────────────────────────────
            if project_cfg and project_cfg.output.directory:
                self._output_dir.setText(project_cfg.output.directory)
            else:
                self._output_dir.setText("")
            self._update_output_resolved()
            self._update_output_info()

            # ── Status ───────────────────────────────────────────────────
            colour = "#2a7a2a" if sources else "#7a2a2a"
            msg = f"Valid project: {len(sources)} source(s)"
            if project_cfg:
                msg += " (project.yaml loaded)"
            self._status_label.setText(msg)
            self._status_label.setStyleSheet(f"color: {colour}; font-weight: bold;")

            self._save_btn.setEnabled(True)
            self._dirty = False

        except (FileNotFoundError, ValueError) as e:
            self._status_label.setText(f"Invalid: {e}")
            self._status_label.setStyleSheet("color: #7a2a2a; font-weight: bold;")
            self._source_table.setRowCount(0)
            self._project_path = None
            self._discovered_sources = []

    def _populate_participants(self, project_cfg, sources):
        """Fill the participants table from project config or participant_ids.csv."""
        self._part_table.blockSignals(True)

        if project_cfg and project_cfg.participants:
            # From project.yaml
            rows = []
            for video, mapping in sorted(project_cfg.participants.items()):
                for tid, label in sorted(mapping.items()):
                    rows.append((video, tid, label))
            self._part_table.setRowCount(len(rows))
            for i, (video, tid, label) in enumerate(rows):
                self._part_table.setItem(i, 0, QTableWidgetItem(video))
                self._part_table.setItem(i, 1, QTableWidgetItem(str(tid)))
                self._part_table.setItem(i, 2, QTableWidgetItem(label))
        else:
            # Try fallback to participant_ids.csv
            from participant_ids import load_participant_csv
            csv_path = self._project_path / "participant_ids.csv"
            if csv_path.is_file():
                pid_maps = load_participant_csv(csv_path)
                rows = []
                for video, mapping in sorted(pid_maps.items()):
                    for tid, label in sorted(mapping.items()):
                        rows.append((video, tid, label))
                self._part_table.setRowCount(len(rows))
                for i, (video, tid, label) in enumerate(rows):
                    self._part_table.setItem(i, 0, QTableWidgetItem(video))
                    self._part_table.setItem(i, 1, QTableWidgetItem(str(tid)))
                    self._part_table.setItem(i, 2, QTableWidgetItem(label))
            else:
                # Pre-populate with one row per video, track_id=0
                self._part_table.setRowCount(len(sources))
                for i, src in enumerate(sources):
                    self._part_table.setItem(i, 0, QTableWidgetItem(src.name))
                    self._part_table.setItem(i, 1, QTableWidgetItem("0"))
                    self._part_table.setItem(i, 2, QTableWidgetItem("P0"))

        self._part_table.blockSignals(False)

    def _populate_conditions(self, project_cfg, sources):
        """Fill the conditions table — one row per video."""
        self._cond_table.blockSignals(True)
        self._cond_table.setRowCount(len(sources))
        for i, src in enumerate(sources):
            name_item = QTableWidgetItem(src.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._cond_table.setItem(i, 0, name_item)

            tags = ""
            if project_cfg and src.name in project_cfg.conditions:
                tags = " | ".join(project_cfg.conditions[src.name])
            self._cond_table.setItem(i, 1, QTableWidgetItem(tags))
        self._cond_table.blockSignals(False)

    # ══════════════════════════════════════════════════════════════════════════
    # Pipeline import
    # ══════════════════════════════════════════════════════════════════════════

    def _import_from_gaze_tab(self):
        """Export the Gaze Tab's current settings as this project's pipeline.yaml."""
        if not self._gaze_tab:
            QMessageBox.warning(self, "Error", "Gaze Tab not available.")
            return
        if not self._project_path:
            QMessageBox.warning(self, "Error", "Load a project first.")
            return

        ns = self._gaze_tab._build_namespace()
        from .pipeline_dialog import _namespace_to_yaml_dict
        yaml_dict = _namespace_to_yaml_dict(ns)

        # Determine target path
        pipe_text = self._pipeline_path.text().strip()
        if not pipe_text:
            pipe_text = "Pipeline/pipeline.yaml"
        target = Path(pipe_text)
        if not target.is_absolute():
            target = self._project_path / target

        # Confirm overwrite if file exists
        if target.exists():
            reply = QMessageBox.question(
                self, "Overwrite Pipeline?",
                f"{target.name} already exists.\nOverwrite with Gaze Tab settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as fh:
            yaml.dump(yaml_dict, fh, default_flow_style=False, sort_keys=False)

        self._pipeline_path.setText(pipe_text)
        self._update_pipeline_summary()
        self._mark_dirty()
        self._status_label.setText(
            f"Imported Gaze Tab settings to {target.name}")
        self._status_label.setStyleSheet("color: #2a7a2a; font-weight: bold;")

    # ══════════════════════════════════════════════════════════════════════════
    # Pipeline summary
    # ══════════════════════════════════════════════════════════════════════════

    def _update_pipeline_summary(self):
        """Load the selected pipeline YAML and display a read-only summary."""
        path_text = self._pipeline_path.text().strip()
        if not path_text or not self._project_path:
            self._pipeline_summary.setText("No pipeline loaded.")
            return

        yaml_path = Path(path_text)
        if not yaml_path.is_absolute():
            yaml_path = self._project_path / yaml_path

        if not yaml_path.is_file():
            self._pipeline_summary.setText(f"File not found: {path_text}")
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
                    gaze_parts.append(f"ray={gaze['ray_length']}")
                if gaze.get("adaptive_ray"):
                    gaze_parts.append(f"adaptive={gaze['adaptive_ray']}")
                if gaze_parts:
                    parts.append(f"Gaze: {', '.join(gaze_parts)}")

            phenomena = cfg.get("phenomena", [])
            if isinstance(phenomena, list) and phenomena:
                names = []
                for item in phenomena:
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, dict):
                        names.extend(item.keys())
                parts.append(f"Phenomena ({len(names)}): {', '.join(names)}")

            self._pipeline_summary.setText(
                " | ".join(parts) if parts else "Pipeline loaded (no notable settings)")
        except Exception as e:
            self._pipeline_summary.setText(f"Error reading pipeline: {e}")

    def _update_output_resolved(self):
        """Show the resolved output directory path."""
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

    # ══════════════════════════════════════════════════════════════════════════
    # Conditions editing
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_condition_to_selected(self):
        """Apply the typed tag to all selected rows in the conditions table."""
        tag = self._cond_tag_input.text().strip()
        if not tag:
            return

        selected_rows = self._get_selected_condition_rows()
        if not selected_rows:
            QMessageBox.information(
                self, "No Selection",
                "Select rows in the Conditions or Sources table first.")
            return

        self._cond_table.blockSignals(True)
        for row in selected_rows:
            current = self._cond_table.item(row, 1)
            current_text = current.text().strip() if current else ""
            existing_tags = [t.strip() for t in current_text.split("|")
                            if t.strip()] if current_text else []
            if tag not in existing_tags:
                existing_tags.append(tag)
            self._cond_table.setItem(
                row, 1, QTableWidgetItem(" | ".join(existing_tags)))
        self._cond_table.blockSignals(False)

        self._sync_conditions_to_sources()
        self._update_output_info()
        self._mark_dirty()
        self._cond_tag_input.clear()

    def _on_cond_table_changed(self, row, col):
        """Sync conditions table edits to the sources table."""
        if col == 1:  # only sync the conditions column
            self._sync_conditions_to_sources()
            self._update_output_info()

    def _sync_conditions_to_sources(self):
        """Mirror conditions table into the read-only sources table."""
        for row in range(self._cond_table.rowCount()):
            cond_item = self._cond_table.item(row, 1)
            text = cond_item.text() if cond_item else ""
            if row < self._source_table.rowCount():
                self._source_table.setItem(row, 1, QTableWidgetItem(text))

    def _remove_condition_from_selected(self):
        """Remove a specific tag from selected rows in the conditions table."""
        tag = self._cond_tag_input.text().strip()
        if not tag:
            return

        selected_rows = self._get_selected_condition_rows()
        if not selected_rows:
            QMessageBox.information(
                self, "No Selection",
                "Select rows in the Conditions or Sources table first.")
            return

        self._cond_table.blockSignals(True)
        for row in selected_rows:
            current = self._cond_table.item(row, 1)
            current_text = current.text().strip() if current else ""
            existing_tags = [t.strip() for t in current_text.split("|")
                            if t.strip()]
            if tag in existing_tags:
                existing_tags.remove(tag)
            self._cond_table.setItem(
                row, 1, QTableWidgetItem(" | ".join(existing_tags)))
        self._cond_table.blockSignals(False)

        self._sync_conditions_to_sources()
        self._update_output_info()
        self._mark_dirty()
        self._cond_tag_input.clear()

    def _clear_conditions_from_selected(self):
        """Clear all tags from selected rows."""
        selected_rows = self._get_selected_condition_rows()
        if not selected_rows:
            QMessageBox.information(
                self, "No Selection",
                "Select rows in the Conditions or Sources table first.")
            return

        self._cond_table.blockSignals(True)
        for row in selected_rows:
            self._cond_table.setItem(row, 1, QTableWidgetItem(""))
        self._cond_table.blockSignals(False)

        self._sync_conditions_to_sources()
        self._update_output_info()
        self._mark_dirty()

    def _get_selected_condition_rows(self) -> set[int]:
        """Get selected row indices from either conditions or sources table."""
        selected_rows = set()
        for idx in self._cond_table.selectedIndexes():
            selected_rows.add(idx.row())
        if not selected_rows:
            for idx in self._source_table.selectedIndexes():
                selected_rows.add(idx.row())
        return selected_rows

    # ══════════════════════════════════════════════════════════════════════════
    # Participants editing
    # ══════════════════════════════════════════════════════════════════════════

    def _add_participant_row(self):
        row = self._part_table.rowCount()
        self._part_table.insertRow(row)
        # Pre-fill with first video name if available
        video_name = ""
        if self._discovered_sources:
            video_name = self._discovered_sources[0].name
        self._part_table.setItem(row, 0, QTableWidgetItem(video_name))
        self._part_table.setItem(row, 1, QTableWidgetItem("0"))
        self._part_table.setItem(row, 2, QTableWidgetItem(""))
        self._mark_dirty()

    def _remove_participant_row(self):
        rows = sorted(set(idx.row() for idx in self._part_table.selectedIndexes()),
                      reverse=True)
        for row in rows:
            self._part_table.removeRow(row)
        self._mark_dirty()

    def _auto_populate_participants(self):
        """Create one participant row (track_id=0, P0) per discovered video."""
        if not self._discovered_sources:
            QMessageBox.information(self, "No Sources", "Load a project first.")
            return

        if self._part_table.rowCount() > 0:
            reply = QMessageBox.question(
                self, "Overwrite?",
                "This will replace all existing participant rows.\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._part_table.blockSignals(True)
        self._part_table.setRowCount(len(self._discovered_sources))
        for i, src in enumerate(self._discovered_sources):
            self._part_table.setItem(i, 0, QTableWidgetItem(src.name))
            self._part_table.setItem(i, 1, QTableWidgetItem("0"))
            self._part_table.setItem(i, 2, QTableWidgetItem("P0"))
        self._part_table.blockSignals(False)
        self._mark_dirty()

    def _bulk_add_participant(self):
        """Add a new participant track to every discovered video."""
        if not self._discovered_sources:
            QMessageBox.information(self, "No Sources", "Load a project first.")
            return

        # Find the next track ID by looking at existing max per video
        existing_tids: dict[str, set[int]] = {}
        for row in range(self._part_table.rowCount()):
            vid_item = self._part_table.item(row, 0)
            tid_item = self._part_table.item(row, 1)
            if vid_item and tid_item:
                vid = vid_item.text().strip()
                try:
                    tid = int(tid_item.text().strip())
                except ValueError:
                    continue
                existing_tids.setdefault(vid, set()).add(tid)

        self._part_table.blockSignals(True)
        for src in self._discovered_sources:
            vid = src.name
            used = existing_tids.get(vid, set())
            next_tid = max(used) + 1 if used else 0
            row = self._part_table.rowCount()
            self._part_table.insertRow(row)
            self._part_table.setItem(row, 0, QTableWidgetItem(vid))
            self._part_table.setItem(row, 1, QTableWidgetItem(str(next_tid)))
            self._part_table.setItem(row, 2, QTableWidgetItem(f"P{next_tid}"))
        self._part_table.blockSignals(False)
        self._mark_dirty()

    # ══════════════════════════════════════════════════════════════════════════
    # Save project.yaml
    # ══════════════════════════════════════════════════════════════════════════

    def _build_project_config(self):
        """Build a ProjectConfig from the current GUI state."""
        from pipeline_config import ProjectConfig, ProjectOutputConfig

        # Pipeline path
        pipe = self._pipeline_path.text().strip() or None

        # Conditions
        conditions: dict[str, list[str]] = {}
        for row in range(self._cond_table.rowCount()):
            video_item = self._cond_table.item(row, 0)
            cond_item = self._cond_table.item(row, 1)
            if not video_item:
                continue
            video = video_item.text().strip()
            cond_text = cond_item.text().strip() if cond_item else ""
            tags = [t.strip() for t in cond_text.split("|") if t.strip()]
            if tags:
                conditions[video] = tags

        # Participants
        participants: dict[str, dict[int, str]] = {}
        for row in range(self._part_table.rowCount()):
            video_item = self._part_table.item(row, 0)
            tid_item = self._part_table.item(row, 1)
            label_item = self._part_table.item(row, 2)
            if not video_item or not tid_item or not label_item:
                continue
            video = video_item.text().strip()
            try:
                tid = int(tid_item.text().strip())
            except ValueError:
                continue
            label = label_item.text().strip() or f"P{tid}"
            if video:
                participants.setdefault(video, {})[tid] = label

        # Output
        out_dir = self._output_dir.text().strip() or None
        output_cfg = ProjectOutputConfig(directory=out_dir)

        return ProjectConfig(
            pipeline_path=pipe,
            conditions=conditions,
            participants=participants,
            output=output_cfg,
        )

    def _save_project_yaml(self):
        """Write project.yaml from current GUI state."""
        if not self._project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        from project_runner import save_project_config
        cfg = self._build_project_config()
        path = save_project_config(self._project_path, cfg)
        self._dirty = False
        self._save_btn.setText("Save project.yaml")
        self._status_label.setText(
            f"Saved: {path.name}  ({len(self._discovered_sources)} source(s))")
        self._status_label.setStyleSheet("color: #2a7a2a; font-weight: bold;")

    def _update_output_info(self):
        """Update the output info label showing what files will be generated."""
        n = len(self._discovered_sources)
        if n == 0:
            self._output_info.setText("No sources loaded")
            return

        tags = set()
        for row in range(self._cond_table.rowCount()):
            item = self._cond_table.item(row, 1)
            if item:
                for t in item.text().split("|"):
                    t = t.strip()
                    if t:
                        tags.add(t)

        lines = [f"Per-video: {n} Summary + {n} Events CSVs"]
        lines.append("Global: Global_Summary.csv + Global_Events.csv")
        if tags:
            lines.append(f"By Condition: {len(tags)} tag(s) \u2192 "
                         f"{len(tags) * 2} condition CSVs")
        else:
            lines.append("By Condition: No conditions defined")
        self._output_info.setText("\n".join(lines))

    def _mark_dirty(self, *_args):
        """Flag that unsaved changes exist and update save button text."""
        if not self._dirty:
            self._dirty = True
            self._save_btn.setText("Save project.yaml *")

    # ══════════════════════════════════════════════════════════════════════════
    # Start / Stop / Poll
    # ══════════════════════════════════════════════════════════════════════════

    def _start(self):
        if self._worker and self._worker.is_alive():
            return
        project_dir = self._project_dir.text().strip()
        if not project_dir:
            QMessageBox.critical(self, "Error", "Select a project directory first.")
            return

        # Prompt save if dirty
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Save project.yaml before running?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._save_project_yaml()

        # Build namespace from gaze tab settings (if available)
        if self._gaze_tab:
            ns = self._gaze_tab._build_namespace()
        else:
            ns = Namespace()

        ns.project = project_dir

        # Build project config from GUI state
        project_cfg = self._build_project_config()

        self._progress_q = queue.Queue()
        self._log_q = queue.Queue()
        self._frame_q = queue.Queue(maxsize=2)

        from .workers import ProjectWorker
        self._worker = ProjectWorker(
            project_dir, ns, self._progress_q, self._log_q, self._frame_q,
            project_cfg=project_cfg)
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
