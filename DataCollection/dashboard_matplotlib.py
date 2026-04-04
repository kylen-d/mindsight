"""
DataCollection/dashboard_matplotlib.py — Matplotlib-based dashboard renderer.

Replaces the raw OpenCV side-panel drawing with a clean, styled matplotlib
composite.  Renders dashboard panels as numpy arrays compatible with
cv2.VideoWriter.

The renderer pre-creates a matplotlib Figure once and reuses it across frames,
clearing and redrawing only when tracker data changes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle as _Rect

# BGR face colours matching the overlay (converted to RGB 0–1 for matplotlib)
_FACE_COLS_BGR = [
    (100, 100, 255), (100, 255, 100), (255, 100, 100),
    (255, 220, 50), (255, 80, 255), (80, 255, 255),
]
_FACE_COLS_RGB = [(b / 255, g / 255, r / 255) for (b, g, r) in _FACE_COLS_BGR]

# Colour palette
_BG = '#121212'
_CARD_BG = '#1e1e1e'
_CARD_BORDER = '#333333'
_HEADING = '#d0d0d0'
_DIM = '#555555'
_ACCENT_JA = '#00c8ff'
_TEXT = '#cccccc'
_VALUE = '#e0e0e0'
_OVERFLOW_COL = '#888888'

# Maximum rows rendered per card before truncation
_MAX_ROWS_PER_CARD = 8


def _bgr_to_mpl(bgr: tuple) -> tuple:
    """Convert a BGR 0–255 colour to an RGB 0–1 tuple for matplotlib."""
    return (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)


def _face_rgb(idx: int) -> tuple:
    """Return an RGB 0–1 colour for the given face index."""
    return _FACE_COLS_RGB[idx % len(_FACE_COLS_RGB)]


def _fmt_value(value, unit: str | None = None) -> str:
    """Standardize number formatting for dashboard display.

    Supported units: 'pct' (percentage), 'frames' (integer with comma sep),
    'duration' (frame count with 'f' suffix), None (pass-through).
    """
    if unit == 'pct':
        try:
            return f"{float(value):.1f}%"
        except (ValueError, TypeError):
            return str(value)
    elif unit == 'frames':
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return str(value)
    elif unit == 'duration':
        try:
            return f"{float(value):.1f}f"
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to fit within an approximate character budget."""
    if len(text) <= max_chars:
        return text
    return text[:max(0, max_chars - 3)] + '...'


class DashboardRenderer:
    """Renders side-panel dashboard as numpy arrays via matplotlib.

    All font sizes and spacing scale with the panel's pixel height so the
    dashboard remains readable from 480p webcam feeds to 4K recordings.

    Usage::

        renderer = DashboardRenderer()
        canvas_bgr = renderer.render(frame, ctx)
        # canvas_bgr is (h, frame_w + 2*panel_w, 3) uint8 BGR
    """

    # Reference resolution: sizes are authored for a 720px-tall panel.
    _REF_H = 720.0

    def __init__(self):
        self._fig: Figure | None = None

    def _scale(self, ph: int) -> float:
        """Return a multiplier that maps reference sizes to actual panel height."""
        return ph / self._REF_H

    def _ensure_fig(self, panel_w_px: int, panel_h_px: int, dpi: int):
        """Create or resize the figure for one side panel."""
        w_in = panel_w_px / dpi
        h_in = panel_h_px / dpi
        if self._fig is None:
            self._fig = Figure(figsize=(w_in, h_in), dpi=dpi,
                               facecolor=_BG)
        else:
            self._fig.set_size_inches(w_in, h_in)
            self._fig.set_dpi(dpi)

    def render(self, frame: np.ndarray, ctx: dict) -> np.ndarray:
        """Build the composite dashboard frame.

        Parameters
        ----------
        frame : annotated video frame (BGR, uint8)
        ctx   : pipeline frame context dict

        Returns
        -------
        Composite BGR uint8 array: [left_panel | video | right_panel]
        """
        h, w = frame.shape[:2]
        panel_w = max(280, int(w * 0.22))
        # Adaptive DPI: sharper text at higher resolutions
        dpi = max(100, int(h / 7.2))

        # Gather data
        fps = ctx.get('fps', 0.0)
        n_dets = ctx.get('n_dets', 0)
        all_trackers = ctx.get('all_trackers', [])
        pid_map = ctx.get('pid_map')

        # Partition tracker cards by panel attribute.
        # JA tracker provides its own card via dashboard_data() like all others.
        n_faces = len(ctx.get('persons_gaze', []))
        n_active = sum(1 for t in (all_trackers or [])
                       if hasattr(t, 'dashboard_data'))
        system_data = {
            'fps': fps,
            'n_dets': n_dets,
            'frame_no': ctx.get('frame_no'),
            'total_frames': ctx.get('total_frames'),
            'n_faces': n_faces,
            'n_trackers': n_active,
        }
        left_cards = []
        right_cards = []
        for tracker in (all_trackers or []):
            if hasattr(tracker, 'dashboard_data'):
                td = tracker.dashboard_data(pid_map=pid_map)
                if td and td.get('rows') is not None:
                    panel = getattr(tracker, 'dashboard_panel', 'right')
                    if panel == 'left':
                        left_cards.append(td)
                    else:
                        right_cards.append(td)

        # Render panels
        left_img = self._render_panel(panel_w, h, dpi, system_data,
                                      left_cards, side='left')
        right_img = self._render_panel(panel_w, h, dpi, None,
                                       right_cards, side='right')

        # Composite
        canvas = np.full((h, w + 2 * panel_w, 3), 18, dtype=np.uint8)
        canvas[:, :panel_w] = left_img[:h, :panel_w]
        canvas[:, panel_w:panel_w + w] = frame
        canvas[:, panel_w + w:] = right_img[:h, :panel_w]

        return canvas

    def _render_panel(self, pw: int, ph: int, dpi: int,
                      system_data: dict | None,
                      cards: list[dict], side: str) -> np.ndarray:
        """Render one side panel as a BGR numpy array."""
        self._ensure_fig(pw, ph, dpi)
        fig = self._fig
        fig.clf()
        fig.set_facecolor(_BG)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor(_BG)
        ax.axis('off')

        # All sizes scale with panel pixel height relative to 720px reference.
        s = self._scale(ph)
        sz = {
            'title_fs':  max(7,  12 * s),    # card title font size
            'body_fs':   max(6,  10 * s),     # body / row font size
            'line_h':    max(0.025, 0.04 * s),  # normalized line height
            'margin':    0.04,
            'gap':       max(0.012, 0.022 * s),  # gap between cards
            'pad_top':   0.015,               # text inset from card top
            'indent':    0.06,                # text indent from margin
            'stripe_w':  0.018,               # accent stripe width
            # Approximate max chars that fit in the card body area
            'max_chars': max(18, int(28 * (pw / 280))),
        }

        y = 0.97  # start near top

        # System stats (left panel only)
        if system_data is not None:
            y = self._draw_system_card(ax, y, sz, system_data)
            y -= sz['gap']

        # Two-pass card layout: measure then render
        overflow = False
        for card in cards:
            card_h = self._measure_card_h(card, sz)
            if y - card_h < 0.02:
                overflow = True
                break
            y = self._draw_tracker_card(ax, y, sz, card)
            y -= sz['gap']

        # Overflow indicator when cards don't fit
        if overflow:
            ax.text(0.5, 0.01, '...', fontsize=sz['body_fs'],
                    color=_OVERFLOW_COL, ha='center', va='bottom',
                    fontfamily='sans-serif')

        # Render to numpy
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(int(fig.get_figheight() * dpi),
                          int(fig.get_figwidth() * dpi), 4)
        # RGBA -> BGR
        return buf[:, :, [2, 1, 0]].copy()

    def _draw_system_card(self, ax, y: float, sz: dict,
                          data: dict) -> float:
        """Draw the SYSTEM stats card with enriched metrics."""
        margin, line_h = sz['margin'], sz['line_h']

        # Build rows dynamically
        rows = [f"FPS: {data['fps']:.1f}"]
        frame_no = data.get('frame_no')
        total = data.get('total_frames')
        if frame_no is not None:
            if total:
                rows.append(f"Frame: {frame_no:,} / {total:,}")
            else:
                rows.append(f"Frame: {frame_no:,}")
        rows.append(f"Detections: {data['n_dets']}")
        n_faces = data.get('n_faces', 0)
        if n_faces > 0:
            rows.append(f"Participants: {n_faces}")
        n_trackers = data.get('n_trackers', 0)
        if n_trackers > 0:
            rows.append(f"Phenomena: {n_trackers}")

        n_body = len(rows)
        card_h = line_h * (n_body + 1.5)
        self._draw_card_bg(ax, margin, y - card_h, 1 - margin, y,
                           stripe_w=sz['stripe_w'])

        y_text = y - sz['pad_top']
        ax.text(margin + 0.04, y_text, 'SYSTEM',
                fontsize=sz['title_fs'], fontweight='bold', color=_HEADING,
                verticalalignment='top', fontfamily='sans-serif')
        y_text -= line_h * 1.2

        for row in rows:
            ax.text(margin + sz['indent'], y_text, row,
                    fontsize=sz['body_fs'], color=_TEXT,
                    verticalalignment='top', fontfamily='monospace')
            y_text -= line_h

        return y - card_h

    def _measure_card_h(self, card: dict, sz: dict) -> float:
        """Pre-calculate the height a card will consume."""
        rows = card.get('rows', [])
        n_rows = min(max(len(rows), 1), _MAX_ROWS_PER_CARD)
        n_bars = sum(1 for r in rows[:n_rows]
                     if r.get('pct') is not None and r.get('pct', 0) > 0)
        # Extra row for "...and N more" if truncated
        if len(rows) > _MAX_ROWS_PER_CARD:
            n_rows += 1
        return sz['line_h'] * (n_rows + 1.5 + n_bars * 0.5) + sz['gap']

    def _draw_tracker_card(self, ax, y: float, sz: dict,
                           card: dict) -> float:
        """Draw a single tracker card."""
        margin, line_h = sz['margin'], sz['line_h']
        indent = sz['indent']
        max_chars = sz['max_chars']
        title = card.get('title', '?')
        colour_bgr = card.get('colour', (180, 180, 180))
        rows = card.get('rows', [])
        empty_text = card.get('empty_text', '--')

        # Truncate rows if there are too many
        truncated = len(rows) > _MAX_ROWS_PER_CARD
        display_rows = rows[:_MAX_ROWS_PER_CARD - 1] if truncated else rows

        accent = _bgr_to_mpl(colour_bgr)
        n_display = max(len(display_rows), 1)
        if truncated:
            n_display += 1  # room for "...and N more"
        n_bars = sum(1 for r in display_rows
                     if r.get('pct') is not None and r.get('pct', 0) > 0)
        card_h = line_h * (n_display + 1.5 + n_bars * 0.5) + sz['gap']

        self._draw_card_bg(ax, margin, y - card_h, 1 - margin, y,
                           accent_color=accent, stripe_w=sz['stripe_w'])

        clip_rect = _Rect((margin, y - card_h), 1 - 2 * margin, card_h,
                          transform=ax.transData)

        y_text = y - sz['pad_top']

        # Title with accent colour (sans-serif for readability)
        t = ax.text(margin + 0.04, y_text, title,
                    fontsize=sz['title_fs'], fontweight='bold', color=accent,
                    verticalalignment='top', fontfamily='sans-serif')
        t.set_clip_path(clip_rect)
        y_text -= line_h * 1.3

        if not display_rows and not truncated:
            t = ax.text(margin + indent, y_text, empty_text,
                        fontsize=sz['body_fs'], color=_DIM,
                        verticalalignment='top', fontfamily='sans-serif',
                        style='italic')
            t.set_clip_path(clip_rect)
        else:
            for row in display_rows:
                label = row.get('label', '')
                value = row.get('value', '')
                pct = row.get('pct')
                unit = row.get('unit')

                # Format value if unit is specified
                if value and unit:
                    value = _fmt_value(value, unit)

                if value:
                    t = ax.text(margin + indent, y_text,
                                _truncate_text(label, max_chars),
                                fontsize=sz['body_fs'], color=_TEXT,
                                verticalalignment='top',
                                fontfamily='sans-serif')
                    t.set_clip_path(clip_rect)
                    t = ax.text(1 - margin - 0.04, y_text, str(value),
                                fontsize=sz['body_fs'], color=_VALUE,
                                verticalalignment='top',
                                horizontalalignment='right',
                                fontfamily='monospace')
                    t.set_clip_path(clip_rect)
                else:
                    t = ax.text(margin + indent, y_text,
                                _truncate_text(label, max_chars),
                                fontsize=sz['body_fs'], color=_TEXT,
                                verticalalignment='top',
                                fontfamily='sans-serif')
                    t.set_clip_path(clip_rect)

                y_text -= line_h

                # Optional percentage bar — drawn below the text row
                if pct is not None and pct > 0:
                    bar_y = y_text - line_h * 0.1
                    bar_left = margin + indent
                    bar_right = 1 - margin - 0.04
                    bar_w = bar_right - bar_left
                    ax.add_patch(plt.Rectangle(
                        (bar_left, bar_y), bar_w, line_h * 0.3,
                        facecolor='#252525', edgecolor='none'))
                    ax.add_patch(plt.Rectangle(
                        (bar_left, bar_y), bar_w * min(pct, 1.0),
                        line_h * 0.3,
                        facecolor=accent, edgecolor='none', alpha=0.7))
                    y_text -= line_h * 0.5

            # Truncation indicator
            if truncated:
                remaining = len(rows) - (_MAX_ROWS_PER_CARD - 1)
                t = ax.text(margin + indent, y_text,
                            f'...and {remaining} more',
                            fontsize=sz['body_fs'], color=_DIM,
                            verticalalignment='top', fontfamily='sans-serif',
                            style='italic')
                t.set_clip_path(clip_rect)
                y_text -= line_h

        return y - card_h

    def _draw_card_bg(self, ax, x0: float, y0: float,
                      x1: float, y1: float,
                      accent_color=None, stripe_w: float = 0.018):
        """Draw a rounded-corner card background."""
        w, h = x1 - x0, y1 - y0
        patch = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.01",
            facecolor=_CARD_BG, edgecolor=_CARD_BORDER,
            linewidth=1.0)
        ax.add_patch(patch)

        # Accent stripe on the left edge
        if accent_color:
            ax.add_patch(plt.Rectangle(
                (x0 + 0.006, y0 + 0.006), stripe_w, h - 0.012,
                facecolor=accent_color, edgecolor='none',
                alpha=0.9))
