"""
inference_settings/widgets.py -- composite controls for the Inference Settings
dialog (UP2 Batch B).

``SliderValue`` is the Q14 rate/weight control: a slider spanning the
recommended range plus an authoritative typeable number.  Typing a value
OUTSIDE the recommended range never clamps -- the slider pins/greys and the
number turns amber with an "outside the usual range" tooltip (spec Q14).
Endpoint labels flank the slider where the spec names them (e.g. "steadier --
snappier").  When a dest has no recommended range the slider is omitted and
only the (still unbounded, still typeable) number shows.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QWidget,
)

# A number field must never clamp a typed value (Q14).  The spin's own range is
# opened far past any recommended slider range so typing outside is preserved.
_WIDE_INT = 10_000_000
_WIDE_FLOAT = 1_000_000.0
_AMBER = "#d08a1d"


class SliderValue(QWidget):
    """Slider (recommended range) + authoritative typeable number.

    ``is_int`` selects a QSpinBox vs QDoubleSpinBox.  ``minimum``/``maximum``
    bound the slider only; the number accepts values beyond them and flags the
    out-of-range state instead of clamping."""

    valueChanged = pyqtSignal()

    def __init__(self, *, is_int: bool, minimum=None, maximum=None, step=None,
                 decimals=None, end_labels: tuple[str, str] | None = None,
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        self._is_int = is_int
        self._min = minimum
        self._max = maximum
        self._has_range = minimum is not None and maximum is not None
        self._step = step if step is not None else (1 if is_int else 0.1)
        self._syncing = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Authoritative number.
        if is_int:
            self._spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            self._spin.setRange(-_WIDE_INT, _WIDE_INT)
            self._spin.setSingleStep(int(self._step) or 1)
        else:
            self._spin = QDoubleSpinBox()
            dec = decimals if decimals is not None else 2
            self._spin.setDecimals(int(dec))
            self._spin.setRange(-_WIDE_FLOAT, _WIDE_FLOAT)
            self._spin.setSingleStep(float(self._step))
        if tooltip:
            self._spin.setToolTip(tooltip)
        self._spin.valueChanged.connect(self._on_spin)

        self._slider: QSlider | None = None
        if self._has_range:
            if end_labels:
                lo = QLabel(end_labels[0])
                lo.setStyleSheet("color: #888; font-size: 10px;")
                lay.addWidget(lo)
            self._slider = QSlider(Qt.Orientation.Horizontal)
            self._steps = max(1, round((float(maximum) - float(minimum))
                                       / float(self._step)))
            self._slider.setRange(0, self._steps)
            if tooltip:
                self._slider.setToolTip(tooltip)
            self._slider.valueChanged.connect(self._on_slider)
            lay.addWidget(self._slider, 1)
            if end_labels:
                hi = QLabel(end_labels[1])
                hi.setStyleSheet("color: #888; font-size: 10px;")
                lay.addWidget(hi)
        lay.addWidget(self._spin)

    # -- slider <-> value mapping --------------------------------------------

    def _val_to_tick(self, v) -> int:
        frac = (float(v) - float(self._min)) / (float(self._max) - float(self._min))
        return round(frac * self._steps)

    def _tick_to_val(self, t: int):
        v = float(self._min) + (t / self._steps) * (float(self._max) - float(self._min))
        return int(round(v)) if self._is_int else round(v, 6)

    def _in_range(self, v) -> bool:
        if not self._has_range:
            return True
        return float(self._min) <= float(v) <= float(self._max)

    # -- signal handlers ------------------------------------------------------

    def _on_spin(self, _v):
        if self._syncing:
            return
        v = self.value()
        if self._slider is not None:
            self._syncing = True
            if self._in_range(v):
                self._slider.setEnabled(True)
                self._slider.setValue(self._val_to_tick(v))
            else:
                # Pin to the nearest end + grey the slider; never clamp the
                # number (Q14).  (Eyes-on B2: it greyed but sat wherever it
                # was, misreading as a live value.)
                self._slider.setValue(
                    0 if float(v) < float(self._min) else self._steps)
                self._slider.setEnabled(False)
            self._syncing = False
        self._apply_range_style(v)
        self.valueChanged.emit()

    def _on_slider(self, tick: int):
        if self._syncing:
            return
        self._syncing = True
        self._spin.setValue(self._tick_to_val(tick))
        self._syncing = False
        self._apply_range_style(self.value())
        self.valueChanged.emit()

    def _apply_range_style(self, v):
        if self._in_range(v):
            self._spin.setStyleSheet("")
            if not self._has_range:
                return
            self._spin.setToolTip(self._spin.toolTip().replace(
                "\n(outside the usual range)", ""))
        else:
            self._spin.setStyleSheet(f"color: {_AMBER};")
            tip = self._spin.toolTip()
            if "outside the usual range" not in tip:
                self._spin.setToolTip(
                    (tip + "\n(outside the usual range)").strip())

    # -- value API ------------------------------------------------------------

    def value(self):
        return self._spin.value()

    def setValue(self, v):
        self._syncing = True
        self._spin.setValue(int(v) if self._is_int else float(v))
        if self._slider is not None:
            if self._in_range(v):
                self._slider.setEnabled(True)
                self._slider.setValue(self._val_to_tick(v))
            else:
                self._slider.setValue(
                    0 if float(v) < float(self._min) else self._steps)
                self._slider.setEnabled(False)
        self._syncing = False
        self._apply_range_style(v)

    def is_over_range(self) -> bool:
        """True when the typed value sits outside the recommended range."""
        return not self._in_range(self.value())
