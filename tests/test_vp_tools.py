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
