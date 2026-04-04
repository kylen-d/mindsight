"""
GUI/live_dashboard.py — Live Qt6 dashboard with real-time charts.

Provides a ``LiveDashboardPanel`` widget that displays rolling time-series
charts for each active phenomena tracker, plus a summary bar with key metrics.

Charts use matplotlib's Qt backend (``FigureCanvasQTAgg``) — no new dependency
beyond what MindSight already requires.  Updates are throttled to ~5 Hz to
avoid excessive Qt overhead.

Supports multi-series per tracker (one line per participant/pair) with
y-axis labels, descriptive subtitles, and compact legends.
"""

from __future__ import annotations

import queue
from collections import deque

import matplotlib
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if matplotlib.get_backend().lower() != 'qtagg':
    try:
        matplotlib.use('QtAgg')
    except Exception:
        pass
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Theme colours (matching dashboard_matplotlib.py)
_BG = '#121212'
_CARD_BG = '#1e1e1e'
_TEXT = '#cccccc'
_GRID = '#2a2a2a'
_DIM = '#555555'

# Palette for per-participant series within a single chart
_SERIES_PALETTE = [
    '#ff6464', '#64ff64', '#6464ff', '#ffdc32', '#ff50ff',
    '#50ffff', '#ffa050', '#a0a0ff', '#50ffa0', '#ff80c0',
]


def _bgr_to_hex(bgr: tuple) -> str:
    return f'#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}'


def _bgr_to_rgb01(bgr: tuple) -> tuple:
    return (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)


# ══════════════════════════════════════════════════════════════════════════════
# Per-tracker multi-series chart
# ══════════════════════════════════════════════════════════════════════════════

class TrackerChartWidget(QWidget):
    """Rolling line-chart widget for a single phenomena tracker.

    Supports multiple named series (e.g. one per participant), each with
    its own colour and legend entry.  Y-axis label and chart subtitle
    are derived from the metric metadata.
    """

    WINDOW = 300  # rolling window (~10s at 30fps)

    def __init__(self, name: str, colour_bgr: tuple = (180, 180, 180),
                 parent=None):
        super().__init__(parent)
        self._name = name
        self._accent = _bgr_to_rgb01(colour_bgr)
        self._accent_hex = _bgr_to_hex(colour_bgr)

        # Per-series rolling data: series_key -> deque of (frame_no, value)
        self._series_data: dict[str, deque] = {}
        self._series_colours: dict[str, str] = {}
        self._series_labels: dict[str, str] = {}
        self._y_label: str = ''
        self._colour_idx: int = 0

        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(1)

        # Title
        self._title_lbl = QLabel(self._name.upper().replace('_', ' '))
        self._title_lbl.setFont(QFont('sans-serif', 9, QFont.Weight.Bold))
        self._title_lbl.setStyleSheet(f'color: {self._accent_hex}; padding: 2px;')
        lay.addWidget(self._title_lbl)

        # Subtitle (metric description, updated dynamically)
        self._subtitle_lbl = QLabel('')
        self._subtitle_lbl.setFont(QFont('sans-serif', 7))
        self._subtitle_lbl.setStyleSheet(f'color: {_DIM};')
        lay.addWidget(self._subtitle_lbl)

        # Matplotlib canvas
        self._fig = Figure(figsize=(4, 1.8), dpi=80, facecolor=_BG)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setMinimumHeight(100)
        lay.addWidget(self._canvas)

        # Axis styling
        self._ax.set_facecolor(_CARD_BG)
        self._ax.tick_params(colors=_TEXT, labelsize=6)
        self._ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.5)
        for spine in self._ax.spines.values():
            spine.set_color(_GRID)
        self._fig.tight_layout(pad=0.5)

        # Matplotlib line objects per series
        self._lines: dict[str, object] = {}

    def _get_series_colour(self, key: str) -> str:
        """Assign a consistent colour to a series key."""
        if key not in self._series_colours:
            if len(self._series_data) <= 1:
                # Single-series: use tracker accent colour
                self._series_colours[key] = self._accent_hex
            else:
                self._series_colours[key] = _SERIES_PALETTE[
                    self._colour_idx % len(_SERIES_PALETTE)]
                self._colour_idx += 1
        return self._series_colours[key]

    def push_metrics(self, frame_no: int, metrics: dict):
        """Push a frame's worth of per-series metric data.

        Parameters
        ----------
        metrics : dict mapping series_key -> {'value': float, 'label': str, 'y_label': str}
        """
        for key, info in metrics.items():
            if key not in self._series_data:
                self._series_data[key] = deque(maxlen=self.WINDOW)
                self._series_labels[key] = info.get('label', key)
            self._series_data[key].append((frame_no, info['value']))

            # Update y-axis label from first series that provides one
            yl = info.get('y_label', '')
            if yl and not self._y_label:
                self._y_label = yl

    def redraw(self):
        """Redraw all series with current data."""
        if not self._series_data:
            return

        ax = self._ax
        all_x_min, all_x_max = float('inf'), float('-inf')
        all_y_max = 0.1

        for key, data in self._series_data.items():
            if not data:
                continue
            x = np.array([d[0] for d in data])
            y = np.array([d[1] for d in data])
            colour = self._get_series_colour(key)
            label = self._series_labels.get(key, key)

            if key in self._lines:
                self._lines[key].set_data(x, y)
            else:
                line, = ax.plot(x, y, color=colour, linewidth=1.2, label=label)
                self._lines[key] = line

            all_x_min = min(all_x_min, x[0])
            all_x_max = max(all_x_max, x[-1])
            all_y_max = max(all_y_max, y.max())

        ax.set_xlim(all_x_min, max(all_x_max, all_x_min + 1))
        ax.set_ylim(0, all_y_max * 1.15)

        # Y-axis label
        if self._y_label:
            ax.set_ylabel(self._y_label, color=_TEXT, fontsize=7)

        # Legend (only when multiple series)
        if len(self._series_data) > 1:
            ax.legend(fontsize=6, loc='upper right', framealpha=0.5,
                      facecolor=_CARD_BG, edgecolor=_GRID, labelcolor=_TEXT)
        elif ax.get_legend():
            ax.get_legend().remove()

        # Update subtitle with series descriptions
        if len(self._series_data) == 1:
            key = next(iter(self._series_data))
            lbl = self._series_labels.get(key, '')
            if lbl and lbl != self._name:
                self._subtitle_lbl.setText(lbl)

        self._canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
# Summary bar
# ══════════════════════════════════════════════════════════════════════════════

class _SummaryBar(QWidget):
    """Horizontal bar showing key run metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f'background: {_CARD_BG}; border-radius: 4px;')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(24)

        self._labels = {}
        for key, text in [('fps', 'FPS'), ('frame', 'Frame'),
                          ('faces', 'Participants')]:
            lbl = QLabel(f'{text}: --')
            lbl.setFont(QFont('monospace', 10))
            lbl.setStyleSheet(f'color: {_TEXT};')
            lay.addWidget(lbl)
            self._labels[key] = lbl
        lay.addStretch(1)

    def update_data(self, data: dict):
        self._labels['fps'].setText(f"FPS: {data.get('fps', 0):.1f}")
        fn = data.get('frame_no', 0)
        self._labels['frame'].setText(f"Frame: {fn:,}")
        self._labels['faces'].setText(
            f"Participants: {data.get('n_faces', 0)}")


# ══════════════════════════════════════════════════════════════════════════════
# Main dashboard panel
# ══════════════════════════════════════════════════════════════════════════════

class LiveDashboardPanel(QWidget):
    """Live dashboard with rolling charts for each active tracker.

    Drains a ``dashboard_q`` at ~5 Hz and routes per-series data to
    per-tracker chart widgets arranged in a 2-column scrollable grid.
    """

    POLL_INTERVAL_MS = 200  # 5 Hz

    def __init__(self, dashboard_q: queue.Queue, parent=None):
        super().__init__(parent)
        self._q = dashboard_q
        self._charts: dict[str, TrackerChartWidget] = {}
        self._custom_widgets: dict[str, tuple] = {}
        self._tracker_colours: dict[str, tuple] = {}

        self._build_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        self._summary = _SummaryBar()
        outer.addWidget(self._summary)

        self._placeholder = QLabel('Waiting for pipeline data...')
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(f'color: {_DIM}; font-size: 14px;')
        outer.addWidget(self._placeholder)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet(f'background: {_BG};')
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(6)
        scroll.setWidget(self._grid_widget)
        outer.addWidget(scroll, 1)

        self.setStyleSheet(f'background: {_BG};')

    def start(self):
        self._poll_timer.start(self.POLL_INTERVAL_MS)

    def stop(self):
        self._poll_timer.stop()

    def reset(self):
        self._poll_timer.stop()
        for chart in self._charts.values():
            chart.setParent(None)
            chart.deleteLater()
        self._charts.clear()
        self._custom_widgets.clear()
        self._tracker_colours.clear()
        self._placeholder.show()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def register_trackers(self, trackers: list):
        """Pre-create chart widgets for known trackers."""
        for tracker in trackers:
            name = getattr(tracker, 'name', None)
            if not name or name.startswith('_'):
                continue
            colour = getattr(tracker, '_COLOUR', (180, 180, 180))
            self._tracker_colours[name] = colour

            custom = None
            if hasattr(tracker, 'dashboard_widget'):
                try:
                    custom = tracker.dashboard_widget()
                except Exception:
                    custom = None

            if custom is not None:
                self._custom_widgets[name] = (custom, tracker)
            else:
                chart = TrackerChartWidget(name, colour)
                self._charts[name] = chart

        self._relayout()

    def _relayout(self):
        all_widgets = []
        for name, chart in self._charts.items():
            all_widgets.append((name, chart))
        for name, (widget, _) in self._custom_widgets.items():
            all_widgets.append((name, widget))

        if all_widgets:
            self._placeholder.hide()

        for i, (name, widget) in enumerate(all_widgets):
            row = i // 2
            col = i % 2
            self._grid.addWidget(widget, row, col)

    def _ensure_chart(self, name: str) -> TrackerChartWidget | None:
        if name in self._charts:
            return self._charts[name]
        if name in self._custom_widgets:
            return None
        if name.startswith('_'):
            return None

        colour = self._tracker_colours.get(name, (180, 180, 180))
        chart = TrackerChartWidget(name, colour)
        self._charts[name] = chart
        self._relayout()
        return chart

    def _poll(self):
        latest = None
        count = 0
        while count < 10:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._poll_timer.stop()
                break
            latest = item
            count += 1

            frame_no = item.get('frame_no', 0)
            rich_metrics = item.get('tracker_rich_metrics', {})

            # Update tracker colours from bridge
            for name, colour in item.get('tracker_colours', {}).items():
                if name not in self._tracker_colours:
                    self._tracker_colours[name] = colour

            # Push per-series data to charts
            for name, series_dict in rich_metrics.items():
                chart = self._ensure_chart(name)
                if chart is not None:
                    chart.push_metrics(frame_no, series_dict)

            # Update custom widgets
            for name, (widget, tracker) in self._custom_widgets.items():
                if hasattr(tracker, 'dashboard_widget_update'):
                    try:
                        tracker.dashboard_widget_update(item)
                    except Exception:
                        pass

        if latest is not None:
            self._summary.update_data(latest)
            if self._placeholder.isVisible():
                self._placeholder.hide()
            for chart in self._charts.values():
                chart.redraw()
