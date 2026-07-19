"""Offscreen tests for the validation annotator dialog (W4B phase 1).

Drives the dialog through its slots with a synthetic frame provider —
no video decode, no FastSAM, no user interaction.  What is under test:
frame sampling, target/state labeling with participant auto-advance,
object boxes (drawn + suggestion-accepted), and the autosave contract
(every mutation lands on disk atomically through the store).
"""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from mindsight.validation import ValidationSet, ValidationStore  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _FakeProvider:
    count = 100

    def read(self, frame_no):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[:, :, 0] = frame_no % 255
        return frame

    def close(self):
        self.closed = True


def _make_dialog(qapp, tmp_path, frames=(0,)):
    from mindsight.GUI.validation_annotator import ValidationAnnotatorDialog
    store = ValidationStore(tmp_path)
    vset = ValidationSet(name="t", video="unused.mp4")
    for f in frames:
        vset.add_frame(f)
    dlg = ValidationAnnotatorDialog(vset, store,
                                    frame_provider=_FakeProvider())
    return dlg, vset, store


def _saved(tmp_path):
    return json.loads((tmp_path / "t.json").read_text())


def test_sample_populates_frames_and_saves(qapp, tmp_path):
    dlg, vset, _ = _make_dialog(qapp, tmp_path, frames=())
    dlg._every_spin.setValue(30)
    dlg._on_sample()
    assert vset.frames() == [0, 30, 60, 90]
    assert vset.every == 30
    assert set(_saved(tmp_path)["labels"]) == {"0", "30", "60", "90"}
    assert dlg._frame_list.count() == 4
    dlg.reject()


def test_target_click_labels_and_advances_participant(qapp, tmp_path):
    dlg, vset, _ = _make_dialog(qapp, tmp_path)
    assert dlg._pid_combo.currentText() == "P0"
    dlg._on_point(45, 67)
    assert vset.labels[0]["P0"] == {"x": 45, "y": 67}
    assert dlg._pid_combo.currentText() == "P1"      # auto-advance
    dlg._on_point(90, 20)
    assert vset.labels[0]["P1"] == {"x": 90, "y": 20}
    assert _saved(tmp_path)["labels"]["0"]["P0"] == {"x": 45, "y": 67}
    assert dlg._label_list.count() == 2
    dlg.reject()


def test_state_buttons_and_clear(qapp, tmp_path):
    dlg, vset, _ = _make_dialog(qapp, tmp_path)
    dlg._set_state_label("offscreen")
    assert vset.labels[0]["P0"] == "offscreen"
    dlg._pid_combo.setCurrentIndex(0)
    dlg._clear_label()
    assert "P0" not in vset.labels[0]
    dlg.reject()


def test_drawn_and_suggested_boxes_become_objects(qapp, tmp_path):
    dlg, vset, _ = _make_dialog(qapp, tmp_path)
    dlg._obj_name.setText("plate")
    dlg._on_box_drawn(10, 20, 60, 80)
    assert vset.objects[0] == [
        {"name": "plate", "x1": 10, "y1": 20, "x2": 60, "y2": 80}]

    dlg._canvas.set_suggestions([[5, 6, 50, 60]])
    dlg._obj_name.setText("cup")
    dlg._on_suggestion_accepted(0)
    assert vset.objects[0][1]["name"] == "cup"
    assert vset.objects[0][1]["x2"] == 50
    assert dlg._canvas._suggestions == []            # proposals consumed
    assert len(_saved(tmp_path)["objects"]["0"]) == 2
    assert dlg._obj_list.count() == 2

    dlg._obj_list.setCurrentRow(0)
    dlg._on_remove_object()
    assert [b["name"] for b in vset.objects[0]] == ["cup"]
    dlg.reject()


def test_remove_frame_and_navigation(qapp, tmp_path):
    dlg, vset, _ = _make_dialog(qapp, tmp_path, frames=(0, 10, 20))
    assert dlg._current_frame_no == 0
    dlg._step(+1)
    assert dlg._current_frame_no == 10
    dlg._on_remove_frame()
    assert vset.frames() == [0, 20]
    dlg._on_add_frame()          # spin defaults to 0 -> no-op duplicate
    assert vset.frames() == [0, 20]
    dlg.reject()


def test_tool_modes_toggle_canvas_suggest(qapp, tmp_path):
    dlg, _, _ = _make_dialog(qapp, tmp_path)
    assert dlg._canvas._suggest_mode is True         # Target tool default
    dlg._tool_draw.setChecked(True)
    assert dlg._canvas._suggest_mode is False        # drags need it off
    # Suggest tool works without the FastSAM weight when a factory is
    # injected (the dialog only hard-requires the weight for the real one).
    dlg._suggester_factory = lambda: None
    dlg._tool_suggest.setChecked(True)
    assert dlg._canvas._suggest_mode is True
    dlg.reject()
