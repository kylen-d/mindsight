"""MP2 (frame extraction) + MP4 (portable VP archives)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _make_video(path: Path, frames: int = 30, w: int = 64, h: int = 48):
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         10.0, (w, h))
    for i in range(frames):
        wr.write(np.full((h, w, 3), i * 8 % 255, dtype=np.uint8))
    wr.release()
    return path


def _make_image(path: Path, shade: int = 128):
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((32, 32, 3), shade, dtype=np.uint8))
    return path


# ── MP2: extract_frames ──────────────────────────────────────────────────────

def test_extract_frames_even_spacing(tmp_path):
    from mindsight.io.video_edit import extract_frames
    video = _make_video(tmp_path / "clip.mp4", frames=30)
    out = extract_frames(video, tmp_path / "frames", count=5)
    assert len(out) == 5
    assert all(p.is_file() and p.suffix == ".jpg" for p in out)
    # Segment midpoints of 30 frames / 5 segments: 3, 9, 15, 21, 27.
    indices = [int(p.stem.rsplit("_f", 1)[1]) for p in out]
    assert indices == [3, 9, 15, 21, 27]
    # Count above total clamps to one frame per available frame.
    out2 = extract_frames(video, tmp_path / "frames2", count=100)
    assert len(out2) == 30


def test_extract_frames_bad_video(tmp_path):
    from mindsight.io.video_edit import extract_frames
    with pytest.raises(ValueError, match="cannot open"):
        extract_frames(tmp_path / "missing.mp4", tmp_path / "o")


# ── MP4: portable archives ───────────────────────────────────────────────────

def _vp_fixture(tmp_path):
    img1 = _make_image(tmp_path / "imgs_a" / "table.jpg", 100)
    img2 = _make_image(tmp_path / "imgs_b" / "table.jpg", 200)  # same name!
    classes = [{"id": 0, "name": "bowl"}, {"id": 1, "name": "spoon"}]
    refs = [
        {"image": str(img1), "annotations": [
            {"cls_id": 0, "bbox": [1, 2, 10, 12]}]},
        {"image": str(img2), "annotations": [
            {"cls_id": 1, "bbox": [3, 4, 20, 22]}]},
    ]
    return classes, refs


def test_vp_archive_round_trip(tmp_path):
    from mindsight.GUI.vp_archive import export_vp_archive, import_vp_archive
    classes, refs = _vp_fixture(tmp_path)
    zip_path = export_vp_archive(tmp_path / "kitchen.vp.zip", classes, refs)
    assert zip_path.is_file()

    # Import into a fresh location ("another machine").
    other = tmp_path / "elsewhere"
    other.mkdir()
    import shutil
    moved = Path(shutil.copy2(zip_path, other / "kitchen.vp.zip"))
    vp_json = import_vp_archive(moved)
    assert vp_json.name == "kitchen.vp.json"
    data = json.loads(vp_json.read_text())
    assert data["classes"] == classes
    assert len(data["references"]) == 2
    for ref, orig in zip(data["references"], refs):
        assert Path(ref["image"]).is_file()          # materialized image
        assert Path(ref["image"]).is_absolute()
        assert ref["annotations"] == orig["annotations"]
    # Same-named source images did not clobber each other.
    names = {Path(r["image"]).name for r in data["references"]}
    assert len(names) == 2

    # Re-import next to the same archive lands in a collision-safe folder.
    vp2 = import_vp_archive(moved)
    assert vp2 != vp_json and vp2.is_file()


def test_vp_archive_export_missing_image(tmp_path):
    from mindsight.GUI.vp_archive import export_vp_archive
    with pytest.raises(ValueError, match="not found"):
        export_vp_archive(tmp_path / "x.vp.zip",
                          [{"id": 0, "name": "a"}],
                          [{"image": str(tmp_path / "gone.jpg"),
                            "annotations": []}])


def test_vp_archive_rejects_garbage(tmp_path):
    from mindsight.GUI.vp_archive import import_vp_archive
    import zipfile
    bad = tmp_path / "bad.vp.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("readme.txt", "not a vp archive")
    with pytest.raises(ValueError, match="vp.json"):
        import_vp_archive(bad)


# ── GUI wiring ───────────────────────────────────────────────────────────────

def test_extract_dialog_extracts_and_collects(qapp, tmp_path):
    from mindsight.GUI.frame_extract_dialog import FrameExtractDialog
    v1 = _make_video(tmp_path / "a.mp4", frames=20)
    v2 = _make_video(tmp_path / "b.mp4", frames=20)
    dlg = FrameExtractDialog()
    dlg._append([v1, v2])
    dlg._append([v1])                       # duplicate ignored
    assert dlg._list.count() == 2
    dlg._count.setValue(3)
    dlg._out.setText(str(tmp_path / "out"))
    dlg._extract()
    assert len(dlg.extracted) == 6
    assert (tmp_path / "out" / "a").is_dir()
    assert (tmp_path / "out" / "b").is_dir()


def test_vp_tab_add_images_and_export(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    from mindsight.GUI.vp_builder_tab import VisualPromptBuilderTab
    tab = VisualPromptBuilderTab()
    img = _make_image(tmp_path / "ref.jpg")
    assert tab.add_images([img]) == 1
    assert tab.add_images([img]) == 0        # dedup
    # Wire one class + one annotation, then export portable.
    tab._classes = [{"id": 0, "name": "bowl"}]
    tab._images[str(img)]["annotations"] = [
        {"cls_id": 0, "cls_name": "bowl", "bbox": [1, 1, 9, 9]}]
    dest = tmp_path / "out.vp.zip"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(dest), ""))
    infos = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: infos.append(a))
    tab._export_portable()
    assert dest.is_file() and infos


# ── Start Fresh (v1.3.1 item 1) ──────────────────────────────────────────────

def _seeded_tab(tmp_path):
    """A tab with one image + class + annotation and stale session extras."""
    from mindsight.GUI.vp_builder_tab import VisualPromptBuilderTab
    tab = VisualPromptBuilderTab()
    img = _make_image(tmp_path / "ref.jpg")
    tab.add_images([img])                    # selects row 0, loads the frame
    tab._classes = [{"id": 0, "name": "bowl"}]
    tab._refresh_class_list()
    tab._images[str(img)]["annotations"] = [
        {"cls_id": 0, "cls_name": "bowl", "bbox": [1, 1, 9, 9]}]
    tab._test_dets[str(img)] = [{"cls_id": 0, "cls_name": "bowl", "x1": 0,
                                 "y1": 0, "x2": 4, "y2": 4, "conf": 0.5}]
    tab._last_saved_vp = str(tmp_path / "old.vp.json")
    tab._pending_suggestions = [[1, 1, 5, 5]]
    tab._canvas.set_suggestions([[1, 1, 5, 5]])
    return tab


def test_start_fresh_clears_everything(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    tab = _seeded_tab(tmp_path)
    suggester = object()
    tab._suggester = suggester
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    tab._start_fresh()
    assert not tab._images and not tab._classes and not tab._test_dets
    assert tab._current_path is None
    assert tab._pending_suggestions == []
    assert tab.current_vp_path() is None
    assert tab._file_list.count() == 0
    assert tab._class_list.count() == 0
    assert tab._canvas._suggestions == []
    assert tab._canvas._hybrid is True       # interaction grammar survives
    assert tab._suggester is suggester       # FastSAM cache survives


def test_start_fresh_declined_is_noop(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    tab = _seeded_tab(tmp_path)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.No)
    tab._start_fresh()
    assert len(tab._images) == 1 and len(tab._classes) == 1
    assert tab.current_vp_path() is not None
    assert tab._file_list.count() == 1


def test_start_fresh_blocked_while_test_running(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    tab = _seeded_tab(tmp_path)

    class FakeWorker:
        def is_alive(self):
            return True

    tab._vp_worker = FakeWorker()
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: pytest.fail("no confirm while worker runs"))
    tab._start_fresh()
    assert len(tab._images) == 1 and len(tab._classes) == 1
    assert "test inference" in tab._status.text().lower()


def test_load_vp_file_resets_stale_session_state(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    tab = _seeded_tab(tmp_path)
    assert tab._canvas._suggestions        # seeded proposals visible

    img = _make_image(tmp_path / "kitchen.jpg")
    vp = tmp_path / "kitchen.vp.json"
    vp.write_text(json.dumps({
        "version": 1,
        "classes": [{"id": 0, "name": "cup"}],
        "references": [{"image": str(img),
                        "annotations": [{"cls_id": 0, "bbox": [1, 1, 8, 8]}]}],
    }))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(vp), ""))
    tab._load_vp_file()
    assert [c["name"] for c in tab._classes] == ["cup"]
    assert list(tab._images) == [str(img)]
    assert tab.current_vp_path() is None     # stale save path cleared
    assert tab._pending_suggestions == []
    assert tab._canvas._suggestions == []    # stale proposals cleared


# ── Active class: keys + chip + fallback popup (v1.3.1 item 2) ───────────────

def test_class_digit_keys_and_cycle(qapp, tmp_path):
    tab = _seeded_tab(tmp_path)
    tab._classes = [{"id": 0, "name": "bowl"}, {"id": 1, "name": "spoon"},
                    {"id": 2, "name": "plate"}]
    tab._refresh_class_list()
    tab._on_class_digit(1)
    assert tab._class_list.currentRow() == 1
    assert "spoon" in tab._active_chip.text()
    tab._on_class_digit(1)                    # same digit clears
    assert tab._class_list.currentRow() == -1
    assert "none" in tab._active_chip.text()
    tab._on_class_digit(9)                    # out of range: no-op
    assert tab._class_list.currentRow() == -1
    tab._on_class_cycle(1)                    # from none -> first
    assert tab._class_list.currentRow() == 0
    tab._on_class_cycle(-1)                   # wraps backwards
    assert tab._class_list.currentRow() == 2


def test_accept_without_class_pops_menu(qapp, tmp_path, monkeypatch):
    from mindsight.GUI.vp_builder_tab import VisualPromptBuilderTab
    tab = _seeded_tab(tmp_path)
    img = next(iter(tab._images))
    tab._class_list.setCurrentRow(-1)
    tab._pending_suggestions = [[2, 2, 20, 20]]
    tab._canvas.set_suggestions([[2, 2, 20, 20]])
    monkeypatch.setattr(VisualPromptBuilderTab, "_popup_class_choice",
                        lambda self: None)    # menu dismissed
    tab._on_suggestion_accepted(0)
    assert len(tab._images[img]["annotations"]) == 1      # unchanged
    assert tab._pending_suggestions == [[2, 2, 20, 20]]   # proposal kept
    monkeypatch.setattr(VisualPromptBuilderTab, "_popup_class_choice",
                        lambda self: self._classes[0])
    tab._on_suggestion_accepted(0)
    anns = tab._images[img]["annotations"]
    assert len(anns) == 2 and anns[-1]["bbox"] == [2, 2, 20, 20]
    assert tab._pending_suggestions == []


def test_popup_class_choice_lists_classes_and_selects(qapp, tmp_path,
                                                      monkeypatch):
    from PyQt6.QtWidgets import QMenu
    tab = _seeded_tab(tmp_path)
    tab._class_list.setCurrentRow(-1)
    captured = {}

    def fake_exec(menu, *_a):
        captured["texts"] = [a.text() for a in menu.actions()
                             if not a.isSeparator()]
        return menu.actions()[0]              # pick the first class
    monkeypatch.setattr(QMenu, "exec", fake_exec)
    assert tab._popup_class_choice() == tab._classes[0]
    assert tab._class_list.currentRow() == 0  # popup also sets active class
    assert captured["texts"] == ["[0] bowl", "New class…"]


def test_suggest_point_checkbox_gate_and_messages(qapp, tmp_path, monkeypatch):
    import time

    import mindsight.GUI.region_suggest as region_suggest
    tab = _seeded_tab(tmp_path)
    tab._suggest_chk.setChecked(False)
    tab._on_suggest_point(5, 5)
    assert not tab._suggest_busy              # checkbox off: inert

    monkeypatch.setattr(region_suggest, "fastsam_path", lambda: "present")
    tab._suggest_chk.setChecked(True)

    class FakeSuggester:
        loaded = False
        last_raw_count = 0

        def __init__(self, result, raw):
            self._result, self._raw = result, raw

        def suggest(self, frame, x, y):
            self.last_raw_count = self._raw
            self.loaded = True
            return self._result

    def run_round(suggester, x, arm_msg=None):
        tab._suggester = suggester
        tab._on_suggest_point(x, 5)
        assert tab._suggest_busy
        if arm_msg is not None:
            assert arm_msg in tab._status.text()
        for _ in range(100):
            tab._poll_suggest()
            if not tab._suggest_busy:
                return
            time.sleep(0.02)
        pytest.fail("suggest round never completed")

    run_round(FakeSuggester([[1, 1, 5, 5]], 3), 5,
              arm_msg="Loading FastSAM")       # first use: load message
    assert tab._pending_suggestions == [[1, 1, 5, 5]]
    assert "proposal(s)" in tab._status.text()

    run_round(FakeSuggester([], 2), 6)         # found but all filtered
    assert "too large or too small" in tab._status.text()

    run_round(FakeSuggester([], 0), 7)         # genuinely nothing
    assert "No region found" in tab._status.text()
