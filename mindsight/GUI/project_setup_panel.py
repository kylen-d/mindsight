"""GUI/project_setup_panel.py -- project-level study setup (W3Y item 8).

Everything the retired Analyze Footage "Study setup" pane held at the
PROJECT level -- pipeline picker, Import from Inference Tuning,
participants, conditions, output root, Save project.yaml -- now lives on
the Projects tab, edited BEFORE running.  Analyze Footage keeps only the
launch surfaces; project runs read the SAVED project.yaml (single source
of truth), and per-run metadata is edited from the runs table ("Edit
run...").  The Inference Settings dialog is the sole processing
authority (anonymize included).
"""
from __future__ import annotations

from pathlib import Path

import yaml
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .conditions_section import ConditionsSection
from .participants_section import ParticipantsSection
from .widgets import CollapsibleGroupBox


class ProjectSetupPanel(QWidget):
    """Edit a project's study-level setup and save it to project.yaml."""

    saved = pyqtSignal()          # project.yaml written -- host refreshes
    message = pyqtSignal(str)     # human-readable status lines

    def __init__(self, gaze_tab=None, parent=None):
        super().__init__(parent)
        self._gaze_tab = gaze_tab
        self._project_path: Path | None = None
        self._dirty = False
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        pipe_row = QHBoxLayout()
        pipe_row.addWidget(QLabel("Pipeline:"))
        self._pipeline_path = QLineEdit()
        self._pipeline_path.setPlaceholderText("Pipeline/pipeline.yaml")
        self._pipeline_path.textChanged.connect(self._mark_dirty)
        pipe_browse = QPushButton("Browse...")
        pipe_browse.clicked.connect(self._browse_pipeline)
        pipe_row.addWidget(self._pipeline_path, 1)
        pipe_row.addWidget(pipe_browse)
        lay.addLayout(pipe_row)

        import_gaze = QPushButton("Import from Inference Tuning")
        import_gaze.setToolTip(
            "Write the current Inference Tuning settings as this project's "
            "pipeline.yaml")
        import_gaze.clicked.connect(self._import_from_gaze)
        lay.addWidget(import_gaze)

        # Headless sources model consumed by the conditions section (the
        # table itself is not shown; it carries filename -> tags rows).
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
        self._save_btn.clicked.connect(self.save)
        lay.addWidget(self._save_btn)

    # ── Populate / dirty tracking ────────────────────────────────────────────

    def open_project(self, project_path) -> None:
        """(Re)load the panel from *project_path*'s project.yaml + sources."""
        from mindsight.project.runner import (
            discover_sources,
            load_project_config,
        )
        self._project_path = Path(project_path)
        cfg = load_project_config(self._project_path)
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
        else:
            self._pipeline_path.setText("")
        self._participants.set_sources(sources)
        self._participants.populate(cfg, sources,
                                    project_path=self._project_path)
        self._conditions.populate(cfg, sources)
        self._output_dir.setText(
            cfg.output.directory
            if cfg and cfg.output and cfg.output.directory else "")
        self._dirty = False
        self._save_btn.setText("Save project.yaml")
        self._save_btn.setEnabled(False)

    def _mark_dirty(self, *_):
        if self._project_path is None:
            return
        if not self._dirty:
            self._dirty = True
            self._save_btn.setText("Save project.yaml *")
        self._save_btn.setEnabled(True)

    @property
    def dirty(self) -> bool:
        return self._dirty

    # ── Config build / save ──────────────────────────────────────────────────

    def build_config(self):
        from mindsight.pipeline_config import ProjectConfig, ProjectOutputConfig
        return ProjectConfig(
            pipeline_path=self._pipeline_path.text().strip() or None,
            conditions=self._conditions.get_conditions(),
            participants=self._participants.get_participants(),
            output=ProjectOutputConfig(
                directory=self._output_dir.text().strip() or None),
        )

    def save(self) -> None:
        if not self._project_path:
            return
        from mindsight.project.runner import save_project_config
        save_project_config(self._project_path, self.build_config())
        self._dirty = False
        self._save_btn.setText("Save project.yaml")
        self._save_btn.setEnabled(False)
        self.message.emit("Saved project.yaml")
        self.saved.emit()

    # ── Pipeline helpers ─────────────────────────────────────────────────────

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
                f"{target.name} exists. Overwrite with Inference Tuning "
                "settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.dump(yaml_dict, default_flow_style=False,
                                    sort_keys=False))
        self._pipeline_path.setText(pipe_text)
        self.message.emit(f"Wrote {target.name}")
