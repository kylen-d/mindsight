"""Offscreen tests for the validation-set wizard (W4C redesign).

Under test: the Build-Project-wizard house style (step list + Continue
with per-step gating), multi-video sets (add video / whole project),
participants seeded from metadata and driving the label selector,
selector reset to the first participant on frame change + auto-hop to
the next frame after the last participant, and undo.  A fake frame
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


def _existing_set(tmp_path, frames=(), name="s"):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    vset = ValidationSet(name=name, video=str(video))
    for f in frames:
        vset.add_frame(f)
    return vset


# ── Step gating ───────────────────────────────────────────────────────────────

def test_new_set_flow_gates_each_step(qapp, tmp_path):
    from mindsight.GUI.validation_wizard import (
        PAGE_FRAMES,
        PAGE_LABEL,
        PAGE_SET,
        PAGE_VIDEOS,
    )
    wiz, store = _wizard(qapp, tmp_path)
    assert wiz._stack.currentIndex() == PAGE_SET

    wiz._go_next()                                    # no name yet
    assert wiz._stack.currentIndex() == PAGE_SET
    assert "name" in wiz._gate_msg.text().lower()

    wiz._name_edit.setText("office-a")
    wiz._go_next()
    assert wiz._stack.currentIndex() == PAGE_VIDEOS
    assert (tmp_path / "office-a.json").is_file()     # saved immediately

    wiz._go_next()                                    # no videos yet
    assert wiz._stack.currentIndex() == PAGE_VIDEOS
    assert "video" in wiz._gate_msg.text().lower()

    missing = tmp_path / "nope.mp4"
    wiz._vset.add_clip(str(missing))
    wiz._go_next()                                    # video must exist
    assert wiz._stack.currentIndex() == PAGE_VIDEOS
    assert "Missing" in wiz._gate_msg.text()

    wiz._vset.remove_clip(0)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    wiz._vset.add_clip(str(video))
    wiz._go_next()
    assert wiz._stack.currentIndex() == PAGE_FRAMES

    wiz._go_next()                                    # no frames yet
    assert wiz._stack.currentIndex() == PAGE_FRAMES
    assert "frame" in wiz._gate_msg.text().lower()

    wiz._every_spin.setValue(30)
    wiz._on_sample()
    wiz._go_next()
    assert wiz._stack.currentIndex() == PAGE_LABEL
    wiz.reject()


def test_existing_set_opens_on_frames_or_label_page(qapp, tmp_path):
    from mindsight.GUI.validation_wizard import PAGE_FRAMES, PAGE_LABEL
    wiz, _ = _wizard(qapp, tmp_path, _existing_set(tmp_path))
    assert wiz._stack.currentIndex() == PAGE_FRAMES    # empty set -> sample
    assert wiz._name_edit.isReadOnly()                 # names are stable
    wiz.reject()
    wiz2, _ = _wizard(qapp, tmp_path, _existing_set(tmp_path, frames=(0, 30)))
    assert wiz2._stack.currentIndex() == PAGE_LABEL    # has frames -> label
    wiz2.reject()


# ── Videos step (multi-clip) ──────────────────────────────────────────────────

def test_videos_page_add_and_project_import(qapp, tmp_path, monkeypatch):
    proj = tmp_path / "proj" / "Inputs" / "Runs"
    for rid in ("r1", "r2"):
        (proj / rid).mkdir(parents=True)
        (proj / rid / f"{rid}.mp4").write_bytes(b"x")
    (proj / "r1" / "run.yaml").write_text(
        "participants:\n  0: S70\n  1: S71\n")

    wiz, store = _wizard(qapp, tmp_path)
    wiz._name_edit.setText("study")
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getExistingDirectory",
        staticmethod(lambda *a, **k: str(tmp_path / "proj")))
    wiz._on_import_project()                          # step-1 shortcut
    assert "2 video(s)" in wiz._proj_note.text()
    assert wiz._labels_edit.text() == "S70, S71"      # metadata labels
    wiz._go_next()
    assert len(wiz._vset.clips) == 2
    assert wiz._vset.participants == ["S70", "S71"]
    assert wiz._video_list.count() == 2

    saved = json.loads((store.root / "study.json").read_text())
    assert saved["format"] == 2 and len(saved["clips"]) == 2
    wiz.reject()


def test_sample_all_videos_covers_every_clip(qapp, tmp_path):
    vset = _existing_set(tmp_path)
    v2 = tmp_path / "second.mp4"
    v2.write_bytes(b"y")
    vset.add_clip(str(v2))
    wiz, _ = _wizard(qapp, tmp_path, vset)
    wiz._every_spin.setValue(45)
    wiz._on_sample_all()
    assert [len(c.frames()) for c in wiz._vset.clips] == [2, 2]  # 0,45 each
    assert "4 frame(s) in the set" in wiz._frames_count_label.text()
    wiz.reject()


def test_sampling_readout_translates_units(qapp, tmp_path):
    wiz, _ = _wizard(qapp, tmp_path, _existing_set(tmp_path))
    wiz._every_spin.setValue(30)
    assert "= every 1.0 s at 30 fps" in wiz._sample_readout.text()
    assert "adds ~3 frames" in wiz._sample_readout.text()   # range(0,90,30)
    assert "90 frames" in wiz._video_info.text()
    assert "3.0 s" in wiz._video_info.text()
    wiz.reject()


# ── Label step ────────────────────────────────────────────────────────────────

def test_label_click_advances_and_hops_to_next_frame(qapp, tmp_path):
    from mindsight.GUI.validation_wizard import PAGE_LABEL
    wiz, store = _wizard(qapp, tmp_path,
                         _existing_set(tmp_path, frames=(0, 30)))
    assert wiz._stack.currentIndex() == PAGE_LABEL
    assert wiz._current_frame_no == 0
    assert wiz._pid_combo.currentText() == "P0"

    wiz._on_point(45, 67)
    assert wiz._vset.labels[0]["0"] == {"x": 45, "y": 67}
    assert wiz._pid_combo.currentText() == "P1"        # auto-advance
    wiz._set_state("offscreen")
    assert wiz._vset.labels[0]["1"] == "offscreen"
    # Last participant labeled -> hopped to the NEXT frame, selector
    # reset to the first participant (user request).
    assert wiz._current_frame_no == 30
    assert wiz._pid_combo.currentText() == "P0"
    assert "1/2 frames fully labeled" in wiz._progress_label.text()

    saved = json.loads((store.root / "s.json").read_text())
    assert saved["labels"]["0"]["0"] == {"x": 45, "y": 67}
    assert wiz._canvas._suggest_mode is True           # gaze-only canvas
    assert not hasattr(wiz, "_obj_list")
    wiz.reject()


def test_pid_selector_resets_on_manual_frame_change(qapp, tmp_path):
    wiz, _ = _wizard(qapp, tmp_path, _existing_set(tmp_path, frames=(0, 30)))
    wiz._on_point(5, 6)                                # P0 -> P1 selected
    assert wiz._pid_combo.currentIndex() == 1
    wiz._step(+1)                                      # manual frame move
    assert wiz._current_frame_no == 30
    assert wiz._pid_combo.currentIndex() == 0          # reset to P0
    wiz.reject()


def test_participants_from_set_drive_selector(qapp, tmp_path):
    vset = _existing_set(tmp_path, frames=(0,))
    vset.participants = ["S70", "S71", "S72"]
    wiz, _ = _wizard(qapp, tmp_path, vset)
    labels = [wiz._pid_combo.itemText(i)
              for i in range(wiz._pid_combo.count())]
    assert labels == ["S70", "S71", "S72"]
    wiz._on_point(5, 6)
    assert wiz._vset.labels[0]["S70"] == {"x": 5, "y": 6}  # stored as typed
    wiz.reject()


def test_label_page_navigation_and_remove(qapp, tmp_path):
    wiz, _ = _wizard(qapp, tmp_path,
                     _existing_set(tmp_path, frames=(0, 30, 60)))
    wiz._step(+1)
    assert wiz._current_frame_no == 30
    wiz._on_remove_frame()
    assert wiz._vset.frames() == [0, 60]
    wiz._pid_combo.setCurrentText("S99")               # custom typed label
    wiz._on_point(5, 6)
    assert wiz._vset.labels[wiz._current_frame_no]["S99"] == {"x": 5, "y": 6}
    wiz.reject()


# ── Undo ──────────────────────────────────────────────────────────────────────

def test_undo_label_restores_previous_value(qapp, tmp_path):
    wiz, store = _wizard(qapp, tmp_path,
                         _existing_set(tmp_path, frames=(0, 30)))
    wiz._on_point(5, 6)                                # P0 point
    wiz._pid_combo.setCurrentIndex(0)
    wiz._on_point(50, 60)                              # overwrite P0
    assert wiz._vset.labels[0]["0"] == {"x": 50, "y": 60}
    wiz._undo()                                        # back to first point
    assert wiz._vset.labels[0]["0"] == {"x": 5, "y": 6}
    wiz._undo()                                        # back to unlabeled
    assert "0" not in wiz._vset.labels[0]
    assert not wiz._undo_btn.isEnabled()               # stack empty
    saved = json.loads((store.root / "s.json").read_text())
    assert saved["labels"]["0"] == {}                  # undo autosaved
    wiz.reject()


def test_undo_sample_and_remove_frame(qapp, tmp_path):
    wiz, _ = _wizard(qapp, tmp_path, _existing_set(tmp_path))
    wiz._every_spin.setValue(30)
    wiz._on_sample()
    assert wiz._vset.frames() == [0, 30, 60]
    wiz._undo()                                        # sampling undone
    assert wiz._vset.frames() == []

    wiz._on_sample()
    wiz._go_next()                                     # to Label
    wiz._on_point(5, 6)
    wiz._on_remove_frame()                             # removes frame 0
    assert 0 not in wiz._vset.frames()
    wiz._undo()                                        # frame + labels back
    assert 0 in wiz._vset.frames()
    assert wiz._vset.labels[0]["0"] == {"x": 5, "y": 6}
    wiz.reject()
