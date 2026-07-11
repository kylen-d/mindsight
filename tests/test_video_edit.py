"""UP4: ffmpeg-backed video editing engine + Crop & Adjust dialog."""

from pathlib import Path

import pytest

pytest.importorskip("imageio_ffmpeg")


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _make_video(path: Path, frames: int = 12, w: int = 64, h: int = 48,
                fps: float = 10.0):
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    for i in range(frames):
        wr.write(np.full((h, w, 3), i * 15 % 255, dtype=np.uint8))
    wr.release()
    return path


def _dims_fps(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def test_ffmpeg_exe_resolves_and_falls_back(monkeypatch):
    import shutil as _shutil

    from mindsight.io import video_edit
    exe = video_edit.ffmpeg_exe()
    assert exe and Path(exe).exists()
    # Without a system ffmpeg, the bundled imageio-ffmpeg binary answers.
    monkeypatch.setattr(_shutil, "which", lambda *_: None)
    fallback = video_edit.ffmpeg_exe()
    assert fallback and Path(fallback).exists()
    assert "imageio_ffmpeg" in fallback


def test_crop_video_crops_and_resamples(tmp_path):
    from mindsight.io.video_edit import crop_video
    src = _make_video(tmp_path / "src.mp4")
    out = crop_video(src, tmp_path / "out.mp4", rect=(4, 4, 33, 25),
                     fps=5.0)
    w, h, fps = _dims_fps(out)
    assert (w, h) == (32, 24)          # odd sizes rounded down to even
    assert fps == pytest.approx(5.0, abs=0.2)


def test_crop_video_argument_errors(tmp_path):
    from mindsight.io.video_edit import crop_video
    src = _make_video(tmp_path / "src.mp4")
    with pytest.raises(ValueError, match="nothing to do"):
        crop_video(src, tmp_path / "o.mp4")
    with pytest.raises(ValueError, match="not found"):
        crop_video(tmp_path / "missing.mp4", tmp_path / "o.mp4", fps=5)
    with pytest.raises(ValueError, match="empty crop"):
        crop_video(src, tmp_path / "o.mp4", rect=(0, 0, 0, 10))


def test_apply_edit_backup_and_overwrite(tmp_path):
    from mindsight.io.video_edit import apply_edit
    src = _make_video(tmp_path / "run" / "dyad01.mp4")
    original_bytes = src.read_bytes()

    backup = apply_edit(src, rect=(0, 0, 32, 24))
    assert backup == tmp_path / "run" / "original" / "dyad01.mp4"
    assert backup.read_bytes() == original_bytes     # untouched original
    assert _dims_fps(src)[:2] == (32, 24)            # same path, new content

    # Second edit backs up without clobbering the first backup.
    backup2 = apply_edit(src, fps=5.0)
    assert backup2 == tmp_path / "run" / "original" / "dyad01_2.mp4"

    # Overwrite mode leaves no new backup.
    before = {p.name for p in (tmp_path / "run" / "original").iterdir()}
    assert apply_edit(src, rect=(0, 0, 16, 12), overwrite=True) is None
    after = {p.name for p in (tmp_path / "run" / "original").iterdir()}
    assert before == after


def test_backup_folder_invisible_to_discovery(tmp_path):
    """The original/ backup keeps the one-video-per-run-folder rule intact."""
    from mindsight.io.video_edit import apply_edit
    from mindsight.project.project import Project
    from mindsight.project.runner import create_project
    from mindsight.project.staging import stage_run

    proj = create_project(tmp_path, "CropStudy")
    video = _make_video(tmp_path / "vid" / "s1.mp4")
    stage_run(proj, video, {"participants": {0: "S70"}})
    spec = Project.open(str(proj)).runs()[0]

    apply_edit(spec.source, rect=(0, 0, 32, 24))
    runs = Project.open(str(proj)).runs()
    assert len(runs) == 1                             # no ambiguity error
    assert Path(runs[0].source) == Path(spec.source)  # same primary path
    assert (Path(spec.source).parent / "original" / "s1.mp4").is_file()


def test_canvas_coordinate_mapping(qapp):
    from PyQt6.QtCore import QRect

    import numpy as np

    from mindsight.GUI.crop_dialog import _CropCanvas
    canvas = _CropCanvas()
    frame = np.zeros((360, 1280, 3), dtype=np.uint8)   # wide video
    canvas.set_frame(frame)
    box = canvas._display_box()
    assert box is not None
    # Full display box maps back to (roughly) the full video.
    vrect = canvas._display_to_video(QRect(box))
    assert vrect is not None
    x, y, w, h = vrect
    assert (x, y) == (0, 0)
    assert abs(w - 1280) <= 4 and abs(h - 360) <= 4
    # Round trip: video rect -> display -> video stays close.
    disp = canvas._video_to_display((320, 90, 640, 180))
    x, y, w, h = canvas._display_to_video(disp)
    assert abs(x - 320) <= 4 and abs(y - 90) <= 4
    assert abs(w - 640) <= 4 and abs(h - 180) <= 4


def test_crop_dialog_batch_apply(qapp, tmp_path, monkeypatch):
    """Two project videos; crop one, fps the other; apply non-destructively."""
    from PyQt6.QtWidgets import QMessageBox

    from mindsight.GUI.crop_dialog import CropVideosDialog
    from mindsight.project.runner import create_project
    from mindsight.project.staging import stage_run

    proj = create_project(tmp_path, "BatchStudy")
    v1 = _make_video(tmp_path / "vid" / "a.mp4")
    v2 = _make_video(tmp_path / "vid" / "b.mp4")
    stage_run(proj, v1, None)
    stage_run(proj, v2, None)

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    infos = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: infos.append(a))
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))

    dlg = CropVideosDialog(proj)
    assert len(dlg._videos) == 2
    assert not dlg._apply_btn.isEnabled()             # nothing pending yet
    # Video 1: queue a crop (programmatically, as the drag handler would).
    dlg._rect_changed((0, 0, 32, 24))
    # Video 2: queue an fps change.
    dlg._show_video(1)
    dlg._fps_check.setChecked(True)
    dlg._fps_spin.setValue(5.0)
    assert len(dlg.pending()) == 2
    assert "(2)" in dlg._apply_btn.text()
    dlg._apply()

    assert dlg.applied == 2 and not warned and infos
    a_run = proj / "Inputs" / "Runs" / "a"
    b_run = proj / "Inputs" / "Runs" / "b"
    assert _dims_fps(a_run / "a.mp4")[:2] == (32, 24)
    assert (a_run / "original" / "a.mp4").is_file()
    assert _dims_fps(b_run / "b.mp4")[2] == pytest.approx(5.0, abs=0.2)
    assert (b_run / "original" / "b.mp4").is_file()
    assert not dlg.pending()                          # edits consumed


# ── LP1: auto-crop ───────────────────────────────────────────────────────────

def test_union_rect_pads_and_clamps():
    from mindsight.GUI.auto_crop import union_rect
    # Union of two boxes + 100px pad, clamped to the frame.
    rect = union_rect([(200, 300, 400, 500), (350, 150, 600, 450)],
                      100, 1280, 720)
    assert rect == (100, 50, 600, 550)
    # Clamping at the edges.
    assert union_rect([(5, 5, 30, 30)], 100, 640, 360) == (0, 0, 130, 130)
    # Nothing to fit.
    assert union_rect([], 100, 640, 360) is None
    assert union_rect([(10, 10, 12, 12)], 0, 640, 360) is None  # degenerate


def test_union_rect_per_side_and_negative_padding():
    from mindsight.GUI.auto_crop import union_rect
    # (left, top, right, bottom) applied independently.
    rect = union_rect([(200, 300, 400, 500)], (10, 20, 30, 40), 1280, 720)
    assert rect == (190, 280, 240, 260)
    # Negative padding crops INSIDE the detections (eyes-on D3).
    rect = union_rect([(200, 300, 400, 500)], (-50, -50, -50, -50), 1280, 720)
    assert rect == (250, 350, 100, 100)
    # Over-negative padding collapses -> degenerate -> None.
    assert union_rect([(200, 300, 400, 500)], (-150, 0, -150, 0),
                      1280, 720) is None


class _FakeBoxes:
    def __init__(self, xyxy):
        import numpy as np
        self.xyxy = np.array(xyxy, dtype=float)


class _FakeResult:
    def __init__(self, xyxy):
        self.boxes = _FakeBoxes(xyxy)


class _FakeDetector:
    """Mimics the ultralytics __call__ -> [result] interface."""

    def __init__(self, xyxy):
        self._xyxy = xyxy
        self.calls = 0

    def __call__(self, frame, **kw):
        self.calls += 1
        return [_FakeResult(self._xyxy)]


def test_detect_boxes_extracts_xyxy():
    from mindsight.GUI.auto_crop import detect_boxes
    det = _FakeDetector([(10, 20, 110, 220)])
    boxes = detect_boxes(det, object())
    assert boxes == [(10.0, 20.0, 110.0, 220.0)]


def test_auto_crop_all_places_reviewable_rects(qapp, tmp_path, monkeypatch):
    """LP1 batch flow: detect on every middle frame, pre-place the rects as
    pending edits, leave the user in the review loop (nothing re-encoded)."""
    from PyQt6.QtWidgets import QMessageBox

    from mindsight.GUI import crop_dialog as cd
    from mindsight.project.runner import create_project
    from mindsight.project.staging import stage_run

    proj = create_project(tmp_path, "AutoStudy")
    stage_run(proj, _make_video(tmp_path / "vid" / "a.mp4"), None)
    stage_run(proj, _make_video(tmp_path / "vid" / "b.mp4"), None)

    fake = _FakeDetector([(10, 10, 30, 25), (20, 12, 40, 30)])
    import mindsight.GUI.auto_crop as ac
    monkeypatch.setattr(ac, "load_landmark_detector",
                        lambda *a, **k: fake)
    infos = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: infos.append(a[2]))

    dlg = cd.CropVideosDialog(proj)
    dlg._auto_pad.setValue(5)
    dlg._auto_classes.setText("person, dining table")
    dlg._auto_crop_all()
    # Union (10,10)-(40,30) + 5px pad, clamped to the 64x48 frame.
    expected = (5, 5, 40, 30)
    pending = dlg.pending()
    assert set(pending) == {"a", "b"}
    assert all(e["rect"] == expected for e in pending.values())
    assert fake.calls == 2
    assert infos and "2 of 2" in infos[-1]
    assert "(2)" in dlg._apply_btn.text()
    # Nothing was re-encoded: sources untouched (review-first flow).
    assert _dims_fps(proj / "Inputs" / "Runs" / "a" / "a.mp4")[:2] == (64, 48)


def test_auto_crop_requires_object_names(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from mindsight.GUI import crop_dialog as cd
    from mindsight.project.runner import create_project
    from mindsight.project.staging import stage_run

    proj = create_project(tmp_path, "EmptyNames")
    stage_run(proj, _make_video(tmp_path / "vid" / "a.mp4"), None)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    dlg = cd.CropVideosDialog(proj)
    dlg._auto_classes.setText("  ")
    dlg._auto_crop_current()
    assert warned and "object" in warned[-1]
    assert not dlg.pending()
