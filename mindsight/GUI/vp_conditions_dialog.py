"""Condition-tag editor for the VP Builder (v1.3.1 item 3c).

Edits the condition VOCABULARY (the tags a study's videos can carry) and the
per-class assignment matrix.  A class checked for no condition is active in
every video; a checked class is prompted only for videos sharing a tag.  The
dialog works on copies -- read ``result_vocabulary`` / ``result_class_tags``
after ``exec()`` accepts.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class VPConditionsDialog(QDialog):
    """Vocabulary list + class-by-condition checkbox matrix."""

    def __init__(self, classes: list[dict], conditions: list[str],
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visual Prompt Conditions")
        self.resize(520, 360)
        self._classes = [dict(c) for c in classes]
        self._vocab: list[str] = list(conditions)
        self._tags: dict[int, set] = {
            c["id"]: set(c.get("conditions") or []) for c in self._classes}
        self.result_vocabulary: list[str] = []
        self.result_class_tags: dict[int, list[str]] = {}
        self._build_ui()
        self._rebuild_table()

    # -- UI ----------------------------------------------------------------

    def _build_ui(self):
        lay = QVBoxLayout(self)
        hint = QLabel(
            "Tick the conditions each class appears in. A class with NO "
            "ticks is active in every video. Videos get their condition "
            "tags in the project setup / Build-Project wizard.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#aaa;font-size:11px;")
        lay.addWidget(hint)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add condition")
        add_btn.clicked.connect(self._add_condition)
        ren_btn = QPushButton("Rename")
        ren_btn.clicked.connect(self._rename_condition)
        del_btn = QPushButton("Remove")
        del_btn.clicked.connect(self._remove_condition)
        for b in (add_btn, ren_btn, del_btn):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self._table = QTableWidget()
        self._table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self._table, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _rebuild_table(self):
        t = self._table
        t.blockSignals(True)
        t.clear()
        t.setColumnCount(len(self._vocab))
        t.setRowCount(len(self._classes))
        t.setHorizontalHeaderLabels(self._vocab)
        t.setVerticalHeaderLabels(
            [f'[{c["id"]}] {c["name"]}' for c in self._classes])
        for row, c in enumerate(self._classes):
            for col, tag in enumerate(self._vocab):
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable
                              | Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(
                    Qt.CheckState.Checked if tag in self._tags[c["id"]]
                    else Qt.CheckState.Unchecked)
                t.setItem(row, col, item)
        t.resizeColumnsToContents()
        t.blockSignals(False)

    # -- vocabulary ops ----------------------------------------------------

    def _current_col(self) -> int:
        return self._table.currentColumn()

    def _add_condition(self):
        name, ok = QInputDialog.getText(self, "Add condition",
                                        "Condition tag:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self._vocab:
            QMessageBox.warning(self, "Duplicate",
                                f"Condition '{name}' already exists.")
            return
        self._vocab.append(name)
        self._rebuild_table()

    def _rename_condition(self):
        col = self._current_col()
        if not (0 <= col < len(self._vocab)):
            return
        old = self._vocab[col]
        new, ok = QInputDialog.getText(self, "Rename condition",
                                       "New tag:", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        new = new.strip()
        if new in self._vocab:
            QMessageBox.warning(self, "Duplicate",
                                f"Condition '{new}' already exists.")
            return
        self._vocab[col] = new
        for tags in self._tags.values():
            if old in tags:
                tags.discard(old)
                tags.add(new)
        self._rebuild_table()

    def _remove_condition(self):
        col = self._current_col()
        if not (0 <= col < len(self._vocab)):
            return
        tag = self._vocab[col]
        used = sum(1 for tags in self._tags.values() if tag in tags)
        if used:
            reply = QMessageBox.question(
                self, "Remove condition",
                f"Remove '{tag}'? {used} class(es) lose this tag.",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._vocab.pop(col)
        for tags in self._tags.values():
            tags.discard(tag)
        self._rebuild_table()

    # -- matrix ------------------------------------------------------------

    def _on_item_changed(self, item):
        row, col = item.row(), item.column()
        if not (0 <= row < len(self._classes) and 0 <= col < len(self._vocab)):
            return
        cid = self._classes[row]["id"]
        tag = self._vocab[col]
        if item.checkState() == Qt.CheckState.Checked:
            self._tags[cid].add(tag)
        else:
            self._tags[cid].discard(tag)

    # -- result ------------------------------------------------------------

    def accept(self):
        self.result_vocabulary = list(self._vocab)
        self.result_class_tags = {
            cid: [t for t in self._vocab if t in tags]
            for cid, tags in self._tags.items()}
        super().accept()
