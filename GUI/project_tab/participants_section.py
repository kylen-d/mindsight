"""Participant ID table management for the project tab."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


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


class ParticipantsSection(QWidget):
    """Participants table + add/remove/auto-populate/bulk-add buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._discovered_sources = []
        self._video_delegate = _VideoComboDelegate(self)
        self._on_dirty = None  # callback set by parent
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(
            ["Video", "Track ID", "Label"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self._table.setItemDelegateForColumn(0, self._video_delegate)
        self._table.setMinimumHeight(100)
        lay.addWidget(self._table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Row")
        add_btn.clicked.connect(self._add_row)
        rm_btn = QPushButton("- Remove Row")
        rm_btn.clicked.connect(self._remove_row)
        auto_btn = QPushButton("Auto-populate")
        auto_btn.setToolTip("Create one P0 row per discovered video")
        auto_btn.clicked.connect(self._auto_populate)
        bulk_btn = QPushButton("+ Track to All")
        bulk_btn.setToolTip("Add a new participant track to every video")
        bulk_btn.clicked.connect(self._bulk_add)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        btn_row.addWidget(auto_btn)
        btn_row.addWidget(bulk_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

    # -- Public API -----------------------------------------------------------

    def set_dirty_callback(self, cb):
        self._on_dirty = cb
        self._table.cellChanged.connect(lambda *_: cb())

    def set_sources(self, sources):
        self._discovered_sources = list(sources)
        self._video_delegate.set_video_names([s.name for s in sources])

    def populate(self, project_cfg, sources, project_path=None):
        """Fill the table from project config or participant_ids.csv."""
        self._table.blockSignals(True)

        if project_cfg and project_cfg.participants:
            rows = []
            for video, mapping in sorted(project_cfg.participants.items()):
                for tid, label in sorted(mapping.items()):
                    rows.append((video, tid, label))
            self._table.setRowCount(len(rows))
            for i, (video, tid, label) in enumerate(rows):
                self._table.setItem(i, 0, QTableWidgetItem(video))
                self._table.setItem(i, 1, QTableWidgetItem(str(tid)))
                self._table.setItem(i, 2, QTableWidgetItem(label))
        else:
            from participant_ids import load_participant_csv
            csv_path = project_path / "participant_ids.csv" \
                if project_path else None
            if csv_path and csv_path.is_file():
                pid_maps = load_participant_csv(csv_path)
                rows = []
                for video, mapping in sorted(pid_maps.items()):
                    for tid, label in sorted(mapping.items()):
                        rows.append((video, tid, label))
                self._table.setRowCount(len(rows))
                for i, (video, tid, label) in enumerate(rows):
                    self._table.setItem(i, 0, QTableWidgetItem(video))
                    self._table.setItem(i, 1, QTableWidgetItem(str(tid)))
                    self._table.setItem(i, 2, QTableWidgetItem(label))
            else:
                self._table.setRowCount(len(sources))
                for i, src in enumerate(sources):
                    self._table.setItem(
                        i, 0, QTableWidgetItem(src.name))
                    self._table.setItem(i, 1, QTableWidgetItem("0"))
                    self._table.setItem(i, 2, QTableWidgetItem("P0"))

        self._table.blockSignals(False)

    def get_participants(self) -> dict[str, dict[int, str]]:
        """Return {video: {track_id: label}} from table state."""
        participants: dict[str, dict[int, str]] = {}
        for row in range(self._table.rowCount()):
            video_item = self._table.item(row, 0)
            tid_item = self._table.item(row, 1)
            label_item = self._table.item(row, 2)
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
        return participants

    # -- Button actions -------------------------------------------------------

    def _mark_dirty(self):
        if self._on_dirty:
            self._on_dirty()

    def _add_row(self):
        row = self._table.rowCount()
        self._table.insertRow(row)
        video_name = ""
        if self._discovered_sources:
            video_name = self._discovered_sources[0].name
        self._table.setItem(row, 0, QTableWidgetItem(video_name))
        self._table.setItem(row, 1, QTableWidgetItem("0"))
        self._table.setItem(row, 2, QTableWidgetItem(""))
        self._mark_dirty()

    def _remove_row(self):
        rows = sorted(
            set(idx.row() for idx in self._table.selectedIndexes()),
            reverse=True)
        for row in rows:
            self._table.removeRow(row)
        self._mark_dirty()

    def _auto_populate(self):
        if not self._discovered_sources:
            QMessageBox.information(
                self, "No Sources", "Load a project first.")
            return
        if self._table.rowCount() > 0:
            reply = QMessageBox.question(
                self, "Overwrite?",
                "This will replace all existing participant rows.\n"
                "Continue?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._discovered_sources))
        for i, src in enumerate(self._discovered_sources):
            self._table.setItem(i, 0, QTableWidgetItem(src.name))
            self._table.setItem(i, 1, QTableWidgetItem("0"))
            self._table.setItem(i, 2, QTableWidgetItem("P0"))
        self._table.blockSignals(False)
        self._mark_dirty()

    def _bulk_add(self):
        if not self._discovered_sources:
            QMessageBox.information(
                self, "No Sources", "Load a project first.")
            return
        existing_tids: dict[str, set[int]] = {}
        for row in range(self._table.rowCount()):
            vid_item = self._table.item(row, 0)
            tid_item = self._table.item(row, 1)
            if vid_item and tid_item:
                vid = vid_item.text().strip()
                try:
                    tid = int(tid_item.text().strip())
                except ValueError:
                    continue
                existing_tids.setdefault(vid, set()).add(tid)
        self._table.blockSignals(True)
        for src in self._discovered_sources:
            vid = src.name
            used = existing_tids.get(vid, set())
            next_tid = max(used) + 1 if used else 0
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(vid))
            self._table.setItem(
                row, 1, QTableWidgetItem(str(next_tid)))
            self._table.setItem(
                row, 2, QTableWidgetItem(f"P{next_tid}"))
        self._table.blockSignals(False)
        self._mark_dirty()
