"""
data_pane.py -- the Projects overview's per-run data preview (eyes-on
2026-07-11, user request): select a run in the overview table and see its
phenomena charts and a CSV preview side by side with the run list, without
jumping to Analyze Footage.

Read-only: charts come from the shared :mod:`csv_charts` builder and the CSV
table from :func:`run_outputs.load_csv_rows` -- the same data the Analyze
Footage output tabs show, so the two surfaces can never disagree.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

_MAX_PREVIEW_ROWS = 200


class RunDataPane(QWidget):
    """Charts + CSV preview for one run's outputs (or an empty-state hint)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._title = QLabel("Data")
        self._title.setStyleSheet("font-weight: bold;")
        lay.addWidget(self._title)

        self._hint = QLabel(
            "Select a run to preview its charts and CSVs here.")
        self._hint.setStyleSheet("color: #888; font-style: italic;")
        self._hint.setWordWrap(True)
        lay.addWidget(self._hint)

        # Charts over CSV preview, boundary draggable.
        split = QSplitter(Qt.Orientation.Vertical)
        self._split = split

        self._chart_scroll = QScrollArea()
        self._chart_scroll.setWidgetResizable(False)
        self._chart_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._chart_canvas = None
        split.addWidget(self._chart_scroll)

        csv_host = QWidget()
        csv_lay = QVBoxLayout(csv_host)
        csv_lay.setContentsMargins(0, 0, 0, 0)
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("CSV:"))
        self._csv_pick = QComboBox()
        self._csv_pick.currentIndexChanged.connect(self._load_csv)
        pick_row.addWidget(self._csv_pick, 1)
        self._csv_note = QLabel("")
        self._csv_note.setStyleSheet("color: #888; font-size: 11px;")
        pick_row.addWidget(self._csv_note)
        csv_lay.addLayout(pick_row)
        self._csv_table = QTableWidget()
        self._csv_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        csv_lay.addWidget(self._csv_table, 1)
        split.addWidget(csv_host)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        # Stretch 100 vs the trailing spacer: with the splitter hidden (hint
        # states) the labels stay TOP-aligned instead of drifting to center.
        lay.addWidget(split, 100)
        lay.addStretch(1)
        split.setVisible(False)

    # ── population ───────────────────────────────────────────────────────────

    def clear(self, hint: str = "Select a run to preview its charts and "
                                "CSVs here."):
        self._title.setText("Data")
        self._hint.setText(hint)
        self._hint.setVisible(True)
        self._split.setVisible(False)

    def set_outputs(self, run_id: str, outputs):
        """Show *outputs* (RunOutputs or None) for *run_id*."""
        if outputs is None or not outputs.csv_paths:
            self._title.setText(f"Data — {run_id}")
            self._hint.setText(
                "No outputs yet -- run this session in Analyze Footage "
                "(planned sessions need their footage first).")
            self._hint.setVisible(True)
            self._split.setVisible(False)
            return
        self._title.setText(f"Data — {run_id}")
        self._hint.setVisible(False)
        self._split.setVisible(True)
        self._render_charts(outputs)
        self._csv_pick.blockSignals(True)
        self._csv_pick.clear()
        for p in outputs.csv_paths:
            self._csv_pick.addItem(p.name, str(p))
        self._csv_pick.blockSignals(False)
        self._csv_pick.setCurrentIndex(0 if self._csv_pick.count() else -1)
        self._load_csv()

    def _render_charts(self, outputs):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from .csv_charts import build_charts_figure

        fig = build_charts_figure(outputs, panel_height=2.6)
        if fig is None:
            self._chart_scroll.setVisible(False)
            return
        if self._chart_canvas is None:
            self._chart_canvas = FigureCanvasQTAgg(fig)
            self._chart_scroll.setWidget(self._chart_canvas)
        else:
            self._chart_canvas.figure = fig
            fig.set_canvas(self._chart_canvas)
        w_in, h_in = fig.get_size_inches()
        dpi = fig.get_dpi()
        self._chart_canvas.setMinimumSize(int(w_in * dpi), int(h_in * dpi))
        self._chart_canvas.resize(int(w_in * dpi), int(h_in * dpi))
        self._chart_scroll.setVisible(True)
        self._chart_canvas.draw_idle()

    def _load_csv(self, *_):
        path = self._csv_pick.currentData()
        if not path or not Path(path).is_file():
            self._csv_table.setRowCount(0)
            self._csv_table.setColumnCount(0)
            self._csv_note.setText("")
            return
        from .run_outputs import load_csv_rows
        try:
            header, rows, total = load_csv_rows(path,
                                                max_rows=_MAX_PREVIEW_ROWS)
        except Exception as exc:  # noqa: BLE001 -- unreadable CSV stays viewable
            self._csv_note.setText(f"Could not read CSV: {exc}")
            return
        self._csv_table.setColumnCount(len(header))
        self._csv_table.setHorizontalHeaderLabels(header)
        self._csv_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row[:len(header)]):
                self._csv_table.setItem(r, c, QTableWidgetItem(str(val)))
        self._csv_note.setText(
            f"showing {len(rows)} of {total} rows" if total > len(rows)
            else f"{total} rows")
