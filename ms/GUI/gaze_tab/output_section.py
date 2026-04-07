"""Output, logging, heatmap, anonymization, and auxiliary stream settings."""

from __future__ import annotations

from argparse import Namespace

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..widgets import _browse_btn, _hrow


class OutputSection(QWidget):
    """Save video, event log, summary CSV, heatmap, charts, anonymize, aux."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._build_ui(lay)

    def _build_ui(self, lay):
        g = QGroupBox("Output")
        f = QFormLayout(g)

        self._cb_save = QCheckBox("Save annotated video")
        f.addRow(self._cb_save)

        self._log_path = QLineEdit()
        self._log_path.setPlaceholderText("optional -- click Browse")
        lb = _browse_btn()
        lb.clicked.connect(
            lambda: self._browse_save(self._log_path, "CSV (*.csv)"))
        f.addRow("Event log:", _hrow(self._log_path, lb))

        self._summary_path = QLineEdit()
        self._summary_path.setPlaceholderText("optional -- click Browse")
        sb = _browse_btn()
        sb.clicked.connect(
            lambda: self._browse_save(self._summary_path, "CSV (*.csv)"))
        f.addRow("Summary CSV:", _hrow(self._summary_path, sb))

        heatmap_row = QWidget()
        hl = QHBoxLayout(heatmap_row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        self._cb_heatmap = QCheckBox("Heatmap")
        hl.addWidget(self._cb_heatmap)
        self._heatmap_path = QLineEdit()
        self._heatmap_path.setPlaceholderText(
            "optional -- heatmap output path")
        hl.addWidget(self._heatmap_path, 1)
        hb = _browse_btn()
        hb.clicked.connect(
            lambda: self._browse_save(
                self._heatmap_path, "Image (*.png *.jpg);;All (*)"))
        hl.addWidget(hb)
        f.addRow(heatmap_row)

        self._cb_charts = QCheckBox("Generate post-run charts")
        self._cb_charts.setToolTip(
            "Save time-series charts for each active phenomena tracker "
            "alongside the summary CSV output.")
        f.addRow(self._cb_charts)

        anon_row = QWidget()
        al = QHBoxLayout(anon_row)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(4)
        self._cb_anonymize = QCheckBox("Anonymize faces")
        al.addWidget(self._cb_anonymize)
        self._anonymize_mode = QComboBox()
        self._anonymize_mode.addItems(["blur", "black"])
        self._anonymize_mode.setEnabled(False)
        al.addWidget(self._anonymize_mode)
        self._cb_anonymize.toggled.connect(self._anonymize_mode.setEnabled)
        f.addRow(anon_row)

        self._participant_ids = QLineEdit()
        self._participant_ids.setPlaceholderText(
            "e.g. S70,S71,S72 (positional)")
        f.addRow("Participant IDs:", self._participant_ids)

        # Auxiliary Streams (collapsed by default)
        aux_grp = QGroupBox("Auxiliary Streams")
        aux_grp.setCheckable(True)
        aux_grp.setChecked(False)
        aux_lay = QVBoxLayout(aux_grp)

        self._aux_table = QTableWidget(0, 5)
        self._aux_table.setHorizontalHeaderLabels(
            ["Source", "Video Type", "Label", "Participants", "Auto-Detect"])
        self._aux_table.horizontalHeader().setStretchLastSection(True)
        self._aux_table.setMinimumHeight(100)
        aux_lay.addWidget(self._aux_table)

        aux_btn_row = _hrow()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self._aux_add_row)
        rm_btn = QPushButton("Remove Row")
        rm_btn.clicked.connect(self._aux_remove_row)
        browse_btn = QPushButton("Browse Source\u2026")
        browse_btn.clicked.connect(self._aux_browse_source)
        aux_btn_row.layout().addWidget(add_btn)
        aux_btn_row.layout().addWidget(rm_btn)
        aux_btn_row.layout().addWidget(browse_btn)
        aux_lay.addWidget(aux_btn_row)

        f.addRow(aux_grp)
        lay.addWidget(g)

    # -- Helpers --------------------------------------------------------------

    def _browse_save(self, line_edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getSaveFileName(self, "Save as", "", filt)
        if path:
            line_edit.setText(path)

    def _aux_add_row(self):
        from ms.pipeline_config import VideoType
        row = self._aux_table.rowCount()
        self._aux_table.insertRow(row)
        self._aux_table.setItem(row, 0, QTableWidgetItem(""))  # Source
        # Video Type combo box
        vtype_combo = QComboBox()
        vtype_combo.addItems([v.value for v in VideoType])
        vtype_combo.setCurrentText(VideoType.EYE_ONLY.value)
        self._aux_table.setCellWidget(row, 1, vtype_combo)
        self._aux_table.setItem(row, 2, QTableWidgetItem(""))  # Label
        self._aux_table.setItem(row, 3, QTableWidgetItem(""))  # Participants
        # Auto-Detect checkbox
        auto_cb = QCheckBox()
        auto_cb.setChecked(True)
        self._aux_table.setCellWidget(row, 4, auto_cb)

    def _aux_remove_row(self):
        row = self._aux_table.currentRow()
        if row >= 0:
            self._aux_table.removeRow(row)

    def _aux_browse_source(self):
        row = self._aux_table.currentRow()
        if row < 0:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Auxiliary Video",
            filter="Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*)")
        if path:
            self._aux_table.setItem(row, 0, QTableWidgetItem(path))

    def _aux_stream_configs(self):
        """Read the aux table into a list of AuxStreamConfig or None."""
        from ms.pipeline_config import AuxStreamConfig, VideoType
        configs = []
        for r in range(self._aux_table.rowCount()):
            source = (self._aux_table.item(r, 0).text().strip()
                      if self._aux_table.item(r, 0) else "")
            vtype_w = self._aux_table.cellWidget(r, 1)
            vtype_str = (vtype_w.currentText() if vtype_w else "custom")
            label = (self._aux_table.item(r, 2).text().strip()
                     if self._aux_table.item(r, 2) else "")
            pids_str = (self._aux_table.item(r, 3).text().strip()
                        if self._aux_table.item(r, 3) else "")
            auto_w = self._aux_table.cellWidget(r, 4)
            auto_detect = auto_w.isChecked() if auto_w else True

            if not (source and label and pids_str):
                continue

            try:
                vtype = VideoType(vtype_str)
            except ValueError:
                vtype = VideoType.CUSTOM

            participants = [p.strip() for p in pids_str.split(",")
                           if p.strip()]
            configs.append(AuxStreamConfig(
                source=source,
                video_type=vtype,
                stream_label=label,
                participants=participants,
                auto_detect_faces=auto_detect,
            ))
        return configs if configs else None

    # -- Namespace interface --------------------------------------------------

    def namespace_values(self) -> dict:
        return dict(
            save=self._cb_save.isChecked() or None,
            log=self._log_path.text().strip() or None,
            summary=self._summary_path.text().strip() or None,
            heatmap=(self._heatmap_path.text().strip()
                     if self._cb_heatmap.isChecked() else None),
            charts=True if self._cb_charts.isChecked() else None,
            anonymize=(self._anonymize_mode.currentText()
                       if self._cb_anonymize.isChecked() else None),
            anonymize_padding=0.3,
            participant_ids=(
                self._participant_ids.text().strip() or None),
            participant_csv=None,
            aux_streams=self._aux_stream_configs(),
            aux_streams_raw=None,
        )

    def apply_namespace(self, ns: Namespace):
        self._cb_save.setChecked(bool(getattr(ns, 'save', False)))
        self._log_path.setText(str(getattr(ns, 'log', '') or ''))
        self._summary_path.setText(str(getattr(ns, 'summary', '') or ''))
        self._participant_ids.setText(
            str(getattr(ns, 'participant_ids', '') or ''))
        heatmap = getattr(ns, 'heatmap', None)
        if heatmap:
            self._cb_heatmap.setChecked(True)
            self._heatmap_path.setText(str(heatmap))
        else:
            self._cb_heatmap.setChecked(False)
            self._heatmap_path.setText('')
        self._cb_charts.setChecked(bool(getattr(ns, 'charts', False)))
        anon = getattr(ns, 'anonymize', None)
        self._cb_anonymize.setChecked(anon is not None)
        if anon:
            idx = self._anonymize_mode.findText(anon)
            if idx >= 0:
                self._anonymize_mode.setCurrentIndex(idx)
