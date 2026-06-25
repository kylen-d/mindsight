"""
models_tab.py
-------------
The **Models** tab (SP3.1 Batch G, Q10) -- a read-only weights view.

It renders one row per weight the current Gaze Tuning config needs
(YOLO / MobileGaze / Gaze-LLE / VP model) with its resolved path, present/missing
state, size, and sha256 prefix -- driven by the same
``provenance.collect_weights`` table the preflight weights check uses.  There are
NO download or verify actions here: the model manager (download + manifest
verification) arrives in SP4; this tab keeps that surface warm.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Human labels for the weight dests (paper terminology, D13).
_DEST_LABEL = {
    "model": "YOLO detector",
    "mgaze_model": "MobileGaze",
    "gazelle_model": "Gaze-LLE",
    "rf_gazelle_model": "Gaze-LLE (blend)",
    "vp_model": "Visual Prompt model",
}

_COLS = ["Model", "Path", "State", "Size", "sha256"]


class ModelsTab(QWidget):
    """Read-only weights overview for the current config (SP4 stub)."""

    def __init__(self, gaze_tab=None, parent=None):
        super().__init__(parent)
        self._gaze_tab = gaze_tab
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        note = QLabel(
            "Weights required by the current Gaze Tuning config. "
            "Download & verify arrives with the model manager (SP4).")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        lay.addWidget(note)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table, 1)

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        lay.addWidget(refresh)

    def refresh(self):
        """Re-read the current config's weights and repopulate the table."""
        from mindsight.outputs import provenance
        ns = (self._gaze_tab._build_namespace()
              if self._gaze_tab is not None else Namespace())
        try:
            weights = provenance.collect_weights(ns)
        except Exception:
            weights = {}
        self._table.setRowCount(len(weights))
        for i, (dest, entry) in enumerate(sorted(weights.items())):
            present = entry.get("sha256") not in (None, "missing")
            resolved = entry.get("resolved", "")
            size = ""
            if present:
                try:
                    size = f"{Path(resolved).stat().st_size / 1e6:.1f} MB"
                except OSError:
                    size = ""
            sha = entry.get("sha256", "")
            sha = sha[:12] if present else "—"
            self._set(i, 0, _DEST_LABEL.get(dest, dest))
            self._set(i, 1, resolved)
            self._set(i, 2, "present" if present else "MISSING",
                      "#2a7a2a" if present else "#b22222")
            self._set(i, 3, size)
            self._set(i, 4, sha)

    def _set(self, row, col, text, colour=None):
        item = QTableWidgetItem(str(text))
        if colour:
            from PyQt6.QtGui import QColor
            item.setForeground(QColor(colour))
        self._table.setItem(row, col, item)
