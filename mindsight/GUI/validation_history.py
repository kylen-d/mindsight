"""
GUI/validation_history.py — Run history / compare dialog (validation suite).

One row per scored run (newest first): the key metrics plus a
"changed vs previous" column summarizing which settings differed from
the run before it (the namespace snapshots persisted by the runner are
the diff source).  Selecting a row shows the full diff below — the
"what did I turn to get this number" view the tune->validate loop needs.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from mindsight.validation import run_history

_COLUMNS = [
    ("run", None, "{}"),
    ("mean px", "endpoint_px_mean", "{:.1f}"),
    ("median px", "endpoint_px_median", "{:.1f}"),
    ("hit rate", "hit_rate", "{:.0%}"),
    ("MAE °", "mae_deg_mean", "{:.1f}"),
    ("IoU", "object_iou_mean", "{:.2f}"),
    ("changed vs prev", None, "{}"),
]


def _changed_summary(changed: dict) -> str:
    if not changed:
        return "—"
    keys = sorted(changed)
    head = ", ".join(keys[:3])
    return head + (f" (+{len(keys) - 3} more)" if len(keys) > 3 else "")


class ValidationHistoryDialog(QDialog):
    """Newest-first run table + full settings diff for the selection."""

    def __init__(self, store, set_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Validation history — {set_name}")
        self.resize(760, 480)
        # Oldest-first from the runner (diffs are vs previous); shown
        # newest-first.
        self._history = list(reversed(run_history(store.root, set_name)))

        lay = QVBoxLayout(self)
        self._table = QTableWidget(len(self._history), len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        for r, entry in enumerate(self._history):
            for c, (_title, key, fmt) in enumerate(_COLUMNS):
                if c == 0:
                    text = entry["run"]
                elif key is None:
                    text = _changed_summary(entry["changed"])
                else:
                    v = entry["score"].get(key)
                    text = fmt.format(v) if v is not None else "—"
                self._table.setItem(r, c, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()
        self._table.itemSelectionChanged.connect(self._on_select)
        lay.addWidget(self._table, 2)

        lay.addWidget(QLabel("Settings changed vs the previous run"))
        self._diff_box = QTextEdit()
        self._diff_box.setReadOnly(True)
        lay.addWidget(self._diff_box, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        if self._history:
            self._table.selectRow(0)

    def _on_select(self):
        row = self._table.currentRow()
        if not (0 <= row < len(self._history)):
            return
        changed = self._history[row]["changed"]
        if not changed:
            self._diff_box.setPlainText(
                "No settings changed (or no previous run to compare).")
            return
        lines = [f"{k}: {old!r} → {new!r}"
                 for k, (old, new) in sorted(changed.items())]
        self._diff_box.setPlainText("\n".join(lines))
