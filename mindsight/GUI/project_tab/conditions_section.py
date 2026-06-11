"""Conditions table management for the project tab."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
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


class ConditionsSection(QWidget):
    """Per-video condition tags table with apply/remove/clear actions.

    Emits ``changed`` whenever the conditions data is modified so that
    the parent can sync to the sources table and update output info.
    """

    changed = pyqtSignal()

    def __init__(self, source_table=None, parent=None):
        super().__init__(parent)
        self._source_table = source_table
        self._on_dirty = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Video", "Conditions"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._table.setMinimumHeight(100)
        self._table.cellChanged.connect(self._on_cell_changed)
        lay.addWidget(self._table)

        hint = QLabel(
            "Edit cells directly, or select rows and use buttons "
            "below. Separate multiple tags with |")
        hint.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        action_row = QHBoxLayout()
        self._tag_input = QLineEdit()
        self._tag_input.setPlaceholderText("Tag to apply/remove...")
        apply_btn = QPushButton("Apply to Selected")
        apply_btn.setToolTip("Add this tag to all selected rows")
        apply_btn.clicked.connect(self._apply_to_selected)
        clear_tag_btn = QPushButton("Remove Tag")
        clear_tag_btn.setToolTip(
            "Remove this specific tag from selected rows")
        clear_tag_btn.clicked.connect(self._remove_from_selected)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setToolTip("Clear all tags from selected rows")
        clear_all_btn.clicked.connect(self._clear_selected)
        action_row.addWidget(self._tag_input, stretch=1)
        action_row.addWidget(apply_btn)
        action_row.addWidget(clear_tag_btn)
        action_row.addWidget(clear_all_btn)
        lay.addLayout(action_row)

    # -- Public API -----------------------------------------------------------

    def set_dirty_callback(self, cb):
        self._on_dirty = cb

    def populate(self, project_cfg, sources):
        """Fill the conditions table -- one row per video."""
        self._table.blockSignals(True)
        self._table.setRowCount(len(sources))
        for i, src in enumerate(sources):
            name_item = QTableWidgetItem(src.name)
            name_item.setFlags(
                name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 0, name_item)
            tags = ""
            if project_cfg and src.name in project_cfg.conditions:
                tags = " | ".join(project_cfg.conditions[src.name])
            self._table.setItem(i, 1, QTableWidgetItem(tags))
        self._table.blockSignals(False)

    def get_conditions(self) -> dict[str, list[str]]:
        """Return {video: [tag, ...]} from table state."""
        conditions: dict[str, list[str]] = {}
        for row in range(self._table.rowCount()):
            video_item = self._table.item(row, 0)
            cond_item = self._table.item(row, 1)
            if not video_item:
                continue
            video = video_item.text().strip()
            cond_text = cond_item.text().strip() if cond_item else ""
            tags = [t.strip() for t in cond_text.split("|")
                    if t.strip()]
            if tags:
                conditions[video] = tags
        return conditions

    def get_all_tags(self) -> set[str]:
        """Return all unique tags across all rows."""
        tags = set()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 1)
            if item:
                for t in item.text().split("|"):
                    t = t.strip()
                    if t:
                        tags.add(t)
        return tags

    # -- Internals ------------------------------------------------------------

    def _mark_dirty(self):
        if self._on_dirty:
            self._on_dirty()

    def _on_cell_changed(self, row, col):
        if col == 1:
            self._sync_to_sources()
            self.changed.emit()

    def _sync_to_sources(self):
        """Mirror conditions into the read-only sources table."""
        if not self._source_table:
            return
        for row in range(self._table.rowCount()):
            cond_item = self._table.item(row, 1)
            text = cond_item.text() if cond_item else ""
            if row < self._source_table.rowCount():
                self._source_table.setItem(
                    row, 1, QTableWidgetItem(text))

    def _get_selected_rows(self) -> set[int]:
        selected = set()
        for idx in self._table.selectedIndexes():
            selected.add(idx.row())
        if not selected and self._source_table:
            for idx in self._source_table.selectedIndexes():
                selected.add(idx.row())
        return selected

    def _apply_to_selected(self):
        tag = self._tag_input.text().strip()
        if not tag:
            return
        selected = self._get_selected_rows()
        if not selected:
            QMessageBox.information(
                self, "No Selection",
                "Select rows in the Conditions or Sources table "
                "first.")
            return
        self._table.blockSignals(True)
        for row in selected:
            current = self._table.item(row, 1)
            current_text = (current.text().strip()
                            if current else "")
            existing = [t.strip() for t in current_text.split("|")
                        if t.strip()] if current_text else []
            if tag not in existing:
                existing.append(tag)
            self._table.setItem(
                row, 1, QTableWidgetItem(" | ".join(existing)))
        self._table.blockSignals(False)
        self._sync_to_sources()
        self.changed.emit()
        self._mark_dirty()
        self._tag_input.clear()

    def _remove_from_selected(self):
        tag = self._tag_input.text().strip()
        if not tag:
            return
        selected = self._get_selected_rows()
        if not selected:
            QMessageBox.information(
                self, "No Selection",
                "Select rows in the Conditions or Sources table "
                "first.")
            return
        self._table.blockSignals(True)
        for row in selected:
            current = self._table.item(row, 1)
            current_text = (current.text().strip()
                            if current else "")
            existing = [t.strip() for t in current_text.split("|")
                        if t.strip()]
            if tag in existing:
                existing.remove(tag)
            self._table.setItem(
                row, 1, QTableWidgetItem(" | ".join(existing)))
        self._table.blockSignals(False)
        self._sync_to_sources()
        self.changed.emit()
        self._mark_dirty()
        self._tag_input.clear()

    def _clear_selected(self):
        selected = self._get_selected_rows()
        if not selected:
            QMessageBox.information(
                self, "No Selection",
                "Select rows in the Conditions or Sources table "
                "first.")
            return
        self._table.blockSignals(True)
        for row in selected:
            self._table.setItem(row, 1, QTableWidgetItem(""))
        self._table.blockSignals(False)
        self._sync_to_sources()
        self.changed.emit()
        self._mark_dirty()
