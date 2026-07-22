"""W3Z item 8: VP Builder Suggest mode (FastSAM click-to-suggest).

The suggester is a plain object with an injectable model factory, so these
tests never load FastSAM.  Canvas interaction runs offscreen Qt.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── Suggester filtering (headless, fake model) ───────────────────────────────

class _FakeBoxes:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def __len__(self):
        return len(self._arr)

    @property
    def xyxy(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = _FakeBoxes(boxes) if boxes is not None else None


def _suggester(boxes):
    from mindsight.GUI.region_suggest import RegionSuggester

    calls = {}

    def fake_model(frame, *, device, points, labels, verbose):
        calls["points"] = points
        return [_FakeResult(boxes)]

    return RegionSuggester(model_factory=lambda: (fake_model, "cpu")), calls


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_suggest_orders_specific_first_and_filters():
    boxes = [
        [0, 0, 640, 480],        # whole image -> dropped (max area frac)
        [90, 90, 300, 300],      # object
        [100, 100, 150, 150],    # part (smallest, contains point)
        [101, 101, 152, 151],    # near-duplicate of the part -> deduped
        [400, 400, 420, 420],    # does not contain the click -> dropped
        [110, 110, 113, 112],    # speck -> dropped (min area frac)
    ]
    s, calls = _suggester(boxes)
    got = s.suggest(FRAME, 120, 120)
    assert calls["points"] == [[120, 120]]
    assert got == [[100, 100, 150, 150], [90, 90, 300, 300]]


def test_suggest_empty_and_none_results():
    s, _ = _suggester(None)
    assert s.suggest(FRAME, 10, 10) == []
    s, _ = _suggester([])
    assert s.suggest(FRAME, 10, 10) == []


def test_suggest_caps_candidates():
    from mindsight.GUI.region_suggest import MAX_SUGGESTIONS
    boxes = [[100 - 10 * i, 100 - 10 * i, 200 + 10 * i, 200 + 10 * i]
             for i in range(8)]
    s, _ = _suggester(boxes)
    assert len(s.suggest(FRAME, 150, 150)) == MAX_SUGGESTIONS


def test_model_loads_once():
    loads = []

    def factory():
        loads.append(1)
        return (lambda *a, **k: [_FakeResult([[10, 10, 60, 60]])], "cpu")

    from mindsight.GUI.region_suggest import RegionSuggester
    s = RegionSuggester(model_factory=factory)
    s.suggest(FRAME, 20, 20)
    s.suggest(FRAME, 20, 20)
    assert loads == [1]


def test_fastsam_path_none_until_downloaded(monkeypatch, tmp_path):
    import mindsight.weights as weights
    from mindsight.GUI import region_suggest

    monkeypatch.setattr(weights, "WEIGHTS_ROOT", tmp_path / "Weights")
    assert region_suggest.fastsam_path() is None
    dest = tmp_path / "Weights" / "SAM" / "FastSAM-s.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"x")
    assert region_suggest.fastsam_path() == dest


def test_manifest_carries_fastsam_entry():
    from mindsight import weights
    e = weights.find_entry("FastSAM-s.pt", backend="SAM")
    assert e is not None
    assert e["required"] is False
    assert e["license"] == "AGPL-3.0"
    assert e["source"] == weights.SOURCE_GITHUB


# ── Canvas suggest mode (offscreen Qt) ───────────────────────────────────────

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _click(canvas, wx, wy):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(wx, wy),
                     canvas.mapToGlobal(QPoint(int(wx), int(wy))).toPointF(),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    canvas.mousePressEvent(ev)


def _canvas_with_frame(qapp):
    from mindsight.GUI.widgets import ImageCanvas
    c = ImageCanvas()
    c.resize(640, 480)
    c.set_image_data(FRAME.copy(), [], [])
    # Fix the mapping deterministically (paintEvent normally sets these).
    c._scale, c._off_x, c._off_y = 1.0, 0, 0
    return c


def test_canvas_point_click_emits_in_suggest_mode(qapp):
    c = _canvas_with_frame(qapp)
    got = []
    c.point_clicked.connect(lambda x, y: got.append((x, y)))
    c.set_suggest_mode(True)
    _click(c, 123, 45)
    assert got == [(123, 45)]


def test_canvas_click_accepts_most_specific_suggestion(qapp):
    c = _canvas_with_frame(qapp)
    accepted, points = [], []
    c.suggestion_accepted.connect(accepted.append)
    c.point_clicked.connect(lambda x, y: points.append((x, y)))
    c.set_suggest_mode(True)
    c.set_suggestions([[50, 50, 400, 400], [100, 100, 200, 200]])
    _click(c, 150, 150)          # inside both -> smaller (index 1) wins
    assert accepted == [1] and points == []
    _click(c, 60, 60)            # only the big one
    assert accepted == [1, 0]


def test_canvas_normal_mode_untouched(qapp):
    c = _canvas_with_frame(qapp)
    got = []
    c.point_clicked.connect(lambda x, y: got.append((x, y)))
    _click(c, 123, 45)           # suggest mode off: starts a drag instead
    assert got == [] and c._drag_start is not None


def test_canvas_mode_off_clears_suggestions(qapp):
    c = _canvas_with_frame(qapp)
    c.set_suggest_mode(True)
    c.set_suggestions([[10, 10, 60, 60]])
    c.set_suggest_mode(False)
    assert c._suggestions == []


# ── Suggester introspection (v1.3.1 item 2) ──────────────────────────────────

def test_loaded_property_and_raw_count_filtered_vs_empty():
    s, _ = _suggester([[0, 0, 640, 480]])    # one box, > max area frac
    assert s.loaded is False
    assert s.suggest(FRAME, 10, 10) == []
    assert s.loaded is True
    assert s.last_raw_count == 1             # found but ALL filtered
    s2, _ = _suggester([])
    assert s2.suggest(FRAME, 10, 10) == []
    assert s2.last_raw_count == 0            # genuinely nothing found


# ── Hybrid canvas grammar (v1.3.1 item 2) ────────────────────────────────────

def _mouse(canvas, kind, wx, wy, button=None):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    button = button or Qt.MouseButton.LeftButton
    ev = QMouseEvent(kind, QPointF(wx, wy),
                     canvas.mapToGlobal(QPoint(int(wx), int(wy))).toPointF(),
                     button, button, Qt.KeyboardModifier.NoModifier)
    if kind == QMouseEvent.Type.MouseButtonPress:
        canvas.mousePressEvent(ev)
    elif kind == QMouseEvent.Type.MouseButtonRelease:
        canvas.mouseReleaseEvent(ev)
    else:
        canvas.mouseMoveEvent(ev)


def _hybrid_click(canvas, wx, wy):
    from PyQt6.QtGui import QMouseEvent
    _mouse(canvas, QMouseEvent.Type.MouseButtonPress, wx, wy)
    _mouse(canvas, QMouseEvent.Type.MouseButtonRelease, wx, wy)


def _key(canvas, key, mods=None):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, key,
                   mods or Qt.KeyboardModifier.NoModifier)
    canvas.keyPressEvent(ev)


def _hybrid_canvas(qapp, dets=()):
    c = _canvas_with_frame(qapp)
    c.set_hybrid_mode(True)
    c._dets = list(dets)
    return c


def test_hybrid_click_empty_emits_point(qapp):
    c = _hybrid_canvas(qapp)
    got = []
    c.point_clicked.connect(lambda x, y: got.append((x, y)))
    _hybrid_click(c, 123, 45)
    assert got == [(123, 45)] and c._drag_start is None


def test_hybrid_click_on_box_toggles_and_drag_draws(qapp):
    from PyQt6.QtGui import QMouseEvent
    det = {"cls_id": 0, "cls_name": "bowl", "x1": 10, "y1": 10,
           "x2": 80, "y2": 80, "conf": 1.0}
    c = _hybrid_canvas(qapp, dets=[det])
    toggled, points, crops = [], [], []
    c.box_toggled.connect(toggled.append)
    c.point_clicked.connect(lambda x, y: points.append((x, y)))
    c.crop_drawn.connect(lambda *b: crops.append(b))
    _hybrid_click(c, 40, 40)                     # inside the box
    assert toggled == [0] and points == []
    _mouse(c, QMouseEvent.Type.MouseButtonPress, 100, 100)
    _mouse(c, QMouseEvent.Type.MouseButtonRelease, 220, 240)
    assert crops == [(100, 100, 220, 240)] and toggled == [0]


def test_hybrid_click_proposal_beats_box_and_point(qapp):
    det = {"cls_id": 0, "cls_name": "bowl", "x1": 90, "y1": 90,
           "x2": 320, "y2": 320, "conf": 1.0}
    c = _hybrid_canvas(qapp, dets=[det])
    accepted, toggled, points = [], [], []
    c.suggestion_accepted.connect(accepted.append)
    c.box_toggled.connect(toggled.append)
    c.point_clicked.connect(lambda x, y: points.append((x, y)))
    c.set_suggestions([[50, 50, 400, 400], [100, 100, 200, 200]])
    _hybrid_click(c, 150, 150)   # inside det AND both proposals
    assert accepted == [1] and toggled == [] and points == []


def test_escape_and_right_click_dismiss_suggestions(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QMouseEvent
    c = _hybrid_canvas(qapp)
    cleared = []
    c.suggestions_cleared.connect(lambda: cleared.append(1))
    c.set_suggestions([[10, 10, 60, 60]])
    _key(c, Qt.Key.Key_Escape)
    assert c._suggestions == [] and cleared == [1]
    c.set_suggestions([[10, 10, 60, 60]])
    _mouse(c, QMouseEvent.Type.MouseButtonPress, 300, 300,
           button=Qt.MouseButton.RightButton)
    assert c._suggestions == [] and cleared == [1, 1]


def test_digit_and_cycle_key_signals(qapp):
    from PyQt6.QtCore import Qt
    c = _hybrid_canvas(qapp)
    digits, cycles = [], []
    c.digit_pressed.connect(digits.append)
    c.cycle_pressed.connect(cycles.append)
    _key(c, Qt.Key.Key_3)
    _key(c, Qt.Key.Key_0)
    _key(c, Qt.Key.Key_Left, Qt.KeyboardModifier.ControlModifier)
    _key(c, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    _key(c, Qt.Key.Key_Left)                 # no modifier: not a cycle
    assert digits == [3, 0] and cycles == [-1, 1]


def test_hover_highlights_proposal_under_cursor(qapp):
    from PyQt6.QtGui import QMouseEvent
    c = _hybrid_canvas(qapp)
    c.set_suggestions([[100, 100, 200, 200]])
    _mouse(c, QMouseEvent.Type.MouseMove, 150, 150)
    assert c._hover_idx == 0
    _mouse(c, QMouseEvent.Type.MouseMove, 400, 400)
    assert c._hover_idx is None
