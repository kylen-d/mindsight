"""
GUI/eye_tracking_widget.py -- Custom Qt dashboard widget for eye tracking data.

Renders one "eye card" per tracked participant showing:
  - Eye outline with iris position (gaze direction indicator)
  - Pupil circle that scales with dilation
  - Eye state badge (F/S/P) from EyeMovement plugin
  - Dilation percentage readout

Painted with QPainter for fast redraws (no matplotlib overhead).
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

# Theme (matching live_dashboard.py)
_BG = QColor('#121212')
_CARD_BG = QColor('#1e1e1e')
_TEXT = QColor('#cccccc')
_DIM = QColor('#555555')

# Eye state colours
_STATE_COLOURS = {
    'fixation': QColor('#00ff00'),
    'saccade': QColor('#ff0000'),
    'pursuit': QColor('#00c8ff'),
}
_STATE_LABELS = {
    'fixation': 'F',
    'saccade': 'S',
    'pursuit': 'P',
}

# Card dimensions
_CARD_W = 120
_CARD_H = 110
_CARD_PAD = 8
_EYE_W = 72
_EYE_H = 32
_IRIS_R = 11
_PUPIL_MIN_R = 3
_PUPIL_MAX_R = 9


def _dilation_colour(dilation_pct: float) -> QColor:
    """Map dilation percentage to a green->yellow->red gradient."""
    if dilation_pct is None:
        return QColor('#888888')
    t = max(0.0, min(1.0, (dilation_pct + 10) / 40))  # -10%..+30% -> 0..1
    r = int(min(255, t * 2 * 255))
    g = int(min(255, (1 - t) * 2 * 255))
    return QColor(r, g, 40)


class EyeTrackingWidget(QWidget):
    """Combined eye-tracking dashboard widget for all participants.

    Shows one eye card per participant arranged in a horizontal flow
    that wraps to the next row when the panel width is exceeded.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(_CARD_H + _CARD_PAD * 2)
        self.setStyleSheet(f'background: #121212;')

        # Per-participant data, keyed by track_id
        self._participants: dict[int, dict] = {}
        # Sorted participant order for stable layout
        self._order: list[int] = []

    def update_participant(self, tid: int, *,
                           dilation_pct: float | None = None,
                           ratio: float | None = None,
                           baseline: float | None = None,
                           iris_offset: tuple | None = None,
                           eye_state: str | None = None,
                           label: str | None = None) -> None:
        """Push new frame data for one participant."""
        if tid not in self._participants:
            self._participants[tid] = {}
            self._order = sorted(self._participants)

        p = self._participants[tid]
        if dilation_pct is not None:
            p['dilation'] = dilation_pct
        if ratio is not None:
            p['ratio'] = ratio
        if baseline is not None:
            p['baseline'] = baseline
        if iris_offset is not None:
            p['iris_offset'] = iris_offset
        if eye_state is not None:
            p['eye_state'] = eye_state
        if label is not None:
            p['label'] = label
        else:
            p.setdefault('label', f'P{tid}')

    def refresh(self):
        """Trigger a repaint after all participants have been updated."""
        # Adjust height based on number of rows
        cols = max(1, (self.width() - _CARD_PAD) // (_CARD_W + _CARD_PAD))
        rows = max(1, math.ceil(len(self._order) / cols))
        needed_h = rows * (_CARD_H + _CARD_PAD) + _CARD_PAD
        if self.minimumHeight() != needed_h:
            self.setMinimumHeight(needed_h)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BG)

        if not self._order:
            painter.setPen(QPen(_DIM))
            painter.setFont(QFont('sans-serif', 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             'No eye data yet')
            painter.end()
            return

        cols = max(1, (self.width() - _CARD_PAD) // (_CARD_W + _CARD_PAD))

        for i, tid in enumerate(self._order):
            col = i % cols
            row = i // cols
            x = _CARD_PAD + col * (_CARD_W + _CARD_PAD)
            y = _CARD_PAD + row * (_CARD_H + _CARD_PAD)
            self._draw_card(painter, x, y, self._participants[tid])

        painter.end()

    def _draw_card(self, p: QPainter, x: int, y: int, data: dict):
        """Draw one participant eye card at (x, y)."""
        # Card background
        card_rect = QRectF(x, y, _CARD_W, _CARD_H)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(_CARD_BG))
        p.drawRoundedRect(card_rect, 6, 6)

        # -- Participant label --
        p.setPen(QPen(_TEXT))
        p.setFont(QFont('sans-serif', 9, QFont.Weight.Bold))
        label = data.get('label', '?')
        p.drawText(QRectF(x, y + 2, _CARD_W, 16),
                    Qt.AlignmentFlag.AlignHCenter, label)

        # -- Eye outline --
        eye_cx = x + _CARD_W / 2
        eye_cy = y + 42
        self._draw_eye(p, eye_cx, eye_cy, data)

        # -- Eye state badge --
        eye_state = data.get('eye_state')
        if eye_state and eye_state in _STATE_LABELS:
            badge_col = _STATE_COLOURS.get(eye_state, _DIM)
            badge_label = _STATE_LABELS[eye_state]
            p.setPen(QPen(badge_col))
            p.setFont(QFont('sans-serif', 9, QFont.Weight.Bold))
            p.drawText(QRectF(x, y + 66, _CARD_W, 14),
                        Qt.AlignmentFlag.AlignHCenter, badge_label)

        # -- Dilation readout --
        dilation = data.get('dilation')
        baseline = data.get('baseline')
        if dilation is not None:
            col = _dilation_colour(dilation)
            p.setPen(QPen(col))
            p.setFont(QFont('sans-serif', 8))
            p.drawText(QRectF(x, y + 82, _CARD_W, 14),
                        Qt.AlignmentFlag.AlignHCenter,
                        f'{dilation:+.1f}%')
        elif baseline is None:
            p.setPen(QPen(_DIM))
            p.setFont(QFont('sans-serif', 7))
            p.drawText(QRectF(x, y + 82, _CARD_W, 14),
                        Qt.AlignmentFlag.AlignHCenter, 'calibrating...')

        # -- Ratio readout --
        ratio = data.get('ratio')
        if ratio is not None:
            p.setPen(QPen(QColor('#888888')))
            p.setFont(QFont('sans-serif', 7))
            p.drawText(QRectF(x, y + 94, _CARD_W, 12),
                        Qt.AlignmentFlag.AlignHCenter,
                        f'r={ratio:.3f}')

    def _draw_eye(self, p: QPainter, cx: float, cy: float, data: dict):
        """Draw the eye outline, iris, and pupil."""
        half_w = _EYE_W / 2
        half_h = _EYE_H / 2

        # Eye outline (almond shape using two arcs)
        path = QPainterPath()
        # Left corner
        lx, ly = cx - half_w, cy
        # Right corner
        rx, ry = cx + half_w, cy

        # Upper arc
        path.moveTo(lx, ly)
        ctrl_up_y = cy - half_h * 1.8
        path.cubicTo(cx - half_w * 0.5, ctrl_up_y,
                      cx + half_w * 0.5, ctrl_up_y,
                      rx, ry)
        # Lower arc
        ctrl_dn_y = cy + half_h * 1.8
        path.cubicTo(cx + half_w * 0.5, ctrl_dn_y,
                      cx - half_w * 0.5, ctrl_dn_y,
                      lx, ly)

        # Fill eye white area
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(30, 30, 35)))
        p.drawPath(path)

        # Eye outline stroke
        p.setPen(QPen(QColor('#666666'), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # -- Iris position --
        offset = data.get('iris_offset', (0.0, 0.0))
        if offset is None:
            offset = (0.0, 0.0)
        # Clamp offset so iris stays within eye outline
        ox = max(-0.8, min(0.8, float(offset[0])))
        oy = max(-0.8, min(0.8, float(offset[1])))
        iris_cx = cx + ox * (half_w - _IRIS_R - 2)
        iris_cy = cy + oy * (half_h - 2)

        # Iris circle
        iris_col = QColor(80, 130, 60)  # dark green iris
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(iris_col))
        p.drawEllipse(QPointF(iris_cx, iris_cy), _IRIS_R, _IRIS_R)

        # Iris border
        p.setPen(QPen(QColor(50, 90, 40), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(iris_cx, iris_cy), _IRIS_R, _IRIS_R)

        # -- Pupil --
        dilation = data.get('dilation')
        ratio = data.get('ratio', 0.4)
        if ratio is not None:
            # Map ratio 0.2-0.7 to min-max pupil radius
            t = max(0.0, min(1.0, (ratio - 0.2) / 0.5))
            pupil_r = _PUPIL_MIN_R + t * (_PUPIL_MAX_R - _PUPIL_MIN_R)
        else:
            pupil_r = (_PUPIL_MIN_R + _PUPIL_MAX_R) / 2

        pupil_col = QColor(10, 10, 10)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(pupil_col))
        p.drawEllipse(QPointF(iris_cx, iris_cy), pupil_r, pupil_r)

        # Pupil highlight (small white dot for realism)
        p.setBrush(QBrush(QColor(255, 255, 255, 80)))
        p.drawEllipse(QPointF(iris_cx - pupil_r * 0.3,
                                iris_cy - pupil_r * 0.3),
                        pupil_r * 0.25, pupil_r * 0.25)
