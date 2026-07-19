"""Offscreen tests for the guided validation-set wizard (W4B rework).

The rulings under test: a guided Set -> Frames -> Label flow with
per-step gating (you cannot reach labeling with zero frames), the
labeled every-N-frames sampler with a seconds-equivalent readout, and
gaze-target-only labeling (object tools are gone).  A fake frame
provider keeps it video-free.
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
    count = 90
    fps = 30.0

    def __init__(self, video_path):
        self.video_path = video_path

    def read(self, frame_no):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[:, :, 0] = frame_no % 255
        return frame

    def close(self):
        self.closed = True


def _wizard(qapp, tmp_path, vset=None):
    from mindsight.GUI.validation_wizard import ValidationSetWizard
    store = ValidationStore(tmp_path)
    if vset is not None:
        store.save(vset)
    return ValidationSetWizard(store, vset,
                               frame_provider_factory=_FakeProvider), store


def _existing_set(tmp_path, frames=()):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    vset = ValidationSet(name="s", video=str(video))
    for f in frames:
        vset.add_frame(f)
    return vset


# ── Page gating ───────────────────────────────────────────────────────────────

def test_new_set_page_gates_on_name_and_real_video(qapp, tmp_path):
    from mindsight.GUI.validation_wizard import PAGE_FRAMES, PAGE_SET
    wiz, store = _wizard(qapp, tmp_path)
    assert wiz._stack.currentIndex() == PAGE_SET
    assert not wiz._set_next_btn.isEnabled()

    wiz._name_edit.setText("office-a")
    wiz._video_edit.setText(str(tmp_path / "nope.mp4"))
    assert not wiz._set_next_btn.isEnabled()
    assert "does not exist" in wiz._set_gate_msg.text()

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    wiz._video_edit.setText(str(video))
    assert wiz._set_next_btn.isEnabled()

    wiz._on_set_next()
    assert wiz._stack.currentIndex() == PAGE_FRAMES
    assert (tmp_path / "office-a.json").is_file()      # saved immediately
    # Labeling stays gated until frames exist.
    assert not wiz._frames_next_btn.isEnabled()
    wiz.reject()


def test_existing_set_opens_on_frames_or_label_page(qapp, tmp_path):
    from mindsight.GUI.validation_wizard import PAGE_FRAMES, PAGE_LABEL
    wiz, _ = _wizard(qapp, tmp_path, _existing_set(tmp_path))
    assert wiz._stack.currentIndex() == PAGE_FRAMES    # empty set -> sample
    wiz.reject()
    wiz2, _ = _wizard(qapp, tmp_path, _existing_set(tmp_path, frames=(0, 30)))
    assert wiz2._stack.currentIndex() == PAGE_LABEL    # has frames -> label
    wiz2.reject()


# ── Sampling page ─────────────────────────────────────────────────────────────

def test_sampling_readout_translates_units(qapp, tmp_path):
    wiz, _ = _wizard(qapp, tmp_path, _existing_set(tmp_path))
    wiz._every_spin.setValue(30)
    assert "= every 1.0 s at 30 fps" in wiz._sample_readout.text()
    assert "adds ~3 frames" in wiz._sample_readout.text()   # range(0,90,30)
    assert "90 frames" in wiz._video_info.text()
    assert "3.0 s" in wiz._video_info.text()
    wiz.reject()


def test_sampling_adds_frames_saves_and_ungates_labeling(qapp, tmp_path):
    wiz, store = _wizard(qapp, tmp_path, _existing_set(tmp_path))
    wiz._every_spin.setValue(30)
    wiz._on_sample()
    assert wiz._vset.frames() == [0, 30, 60]
    assert wiz._frames_next_btn.isEnabled()
    assert "3 frame(s)" in wiz._frames_count_label.text()
    saved = json.loads((store.root / "s.json").read_text())
    assert set(saved["labels"]) == {"0", "30", "60"}
    # Single-frame add works too.
    wiz._frame_spin.setValue(45)
    wiz._on_add_frame()
    assert 45 in wiz._vset.frames()
    wiz.reject()


# ── Labeling page (gaze targets only) ────────────────────────────────────────

def test_label_page_click_stores_digit_pid_and_advances(qapp, tmp_path):
    from mindsight.GUI.validation_wizard import PAGE_LABEL
    wiz, store = _wizard(qapp, tmp_path, _existing_set(tmp_path, frames=(0, 30)))
    assert wiz._stack.currentIndex() == PAGE_LABEL
    assert wiz._current_frame_no == 0
    assert wiz._pid_combo.currentText() == "P0"
    wiz._on_point(45, 67)
    assert wiz._vset.labels[0]["0"] == {"x": 45, "y": 67}
    assert wiz._pid_combo.currentText() == "P1"        # auto-advance
    wiz._set_state("offscreen")
    assert wiz._vset.labels[0]["1"] == "offscreen"
    assert wiz._label_list.count() == 2
    saved = json.loads((store.root / "s.json").read_text())
    assert saved["labels"]["0"]["0"] == {"x": 45, "y": 67}
    # The canvas is permanently in point-click mode; no object tools exist.
    assert wiz._canvas._suggest_mode is True
    assert not hasattr(wiz, "_obj_list")
    wiz.reject()


def test_label_page_navigation_and_remove(qapp, tmp_path):
    wiz, _ = _wizard(qapp, tmp_path, _existing_set(tmp_path, frames=(0, 30, 60)))
    wiz._step(+1)
    assert wiz._current_frame_no == 30
    wiz._on_remove_frame()
    assert wiz._vset.frames() == [0, 60]
    wiz._pid_combo.setCurrentText("S70")               # custom label as typed
    wiz._on_point(5, 6)
    assert wiz._vset.labels[wiz._current_frame_no]["S70"] == {"x": 5, "y": 6}
    wiz.reject()
