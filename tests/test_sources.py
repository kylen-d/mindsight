"""Unit coverage for mindsight.io.sources source normalization (UP1 D1).

``open_video_source`` must hand cv2 an ``int`` for a digit-only camera-index
string (cv2.VideoCapture("0") does not open a webcam) while leaving file-path
strings untouched.  No real capture is opened -- cv2.VideoCapture is stubbed.
"""

import numpy as np
import pytest

import mindsight.io.sources as sources


class _FakeCapture:
    def __init__(self, arg, *, opened=True, frames=True):
        self.arg = arg
        self._opened = opened
        self._frames = frames
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._frames:
            return True, np.zeros((4, 4, 3), np.uint8)
        return False, None

    def release(self):
        self.released = True

    def get(self, _prop):
        return 30.0


def test_digit_string_opens_as_int_camera_index(monkeypatch):
    calls = []

    def fake_vc(arg):
        calls.append(arg)
        return _FakeCapture(arg)

    monkeypatch.setattr(sources.cv2, "VideoCapture", fake_vc)
    cap, fps = sources.open_video_source("0")
    assert calls == [0]
    assert isinstance(calls[0], int)
    assert cap.arg == 0


def test_path_string_passed_through_unchanged(monkeypatch):
    calls = []

    def fake_vc(arg):
        calls.append(arg)
        return _FakeCapture(arg)

    monkeypatch.setattr(sources.cv2, "VideoCapture", fake_vc)
    sources.open_video_source("test_data/trimmed.mp4")
    assert calls == ["test_data/trimmed.mp4"]
    assert isinstance(calls[0], str)


def test_int_source_passed_through_unchanged(monkeypatch):
    """An already-int source (CLI/GazeWorker convert upstream) is untouched."""
    calls = []

    def fake_vc(arg):
        calls.append(arg)
        return _FakeCapture(arg)

    monkeypatch.setattr(sources.cv2, "VideoCapture", fake_vc)
    sources.open_video_source(0)
    assert calls == [0]
    assert isinstance(calls[0], int)


def test_unopenable_source_raises_runtimeerror(monkeypatch):
    monkeypatch.setattr(sources.cv2, "VideoCapture",
                        lambda arg: _FakeCapture(arg, opened=False))
    with pytest.raises(RuntimeError):
        sources.open_video_source("3")


def test_camera_open_failure_is_actionable(monkeypatch):
    """A camera (int) that won't open gets a permission-aware message and the
    capture is released."""
    cap = _FakeCapture(0, opened=False)
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda arg: cap)
    with pytest.raises(RuntimeError, match="camera 0"):
        sources.open_video_source("0")
    assert "Privacy" in _last_error(monkeypatch, sources, "0")
    assert cap.released


def test_camera_opens_but_no_frames_fails_loudly(monkeypatch):
    """The macOS 'opened but streams nothing' case raises instead of silently
    ending with zero frames."""
    cap = _FakeCapture(0, opened=True, frames=False)
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda arg: cap)
    with pytest.raises(RuntimeError, match="no video"):
        sources.open_video_source(0)
    assert cap.released


def test_file_source_is_not_warmup_probed(monkeypatch):
    """A file source that opens is returned WITHOUT a warm-up read (no frame
    consumed), so file/smoke behavior stays byte-identical."""
    cap = _FakeCapture("clip.mp4", opened=True, frames=True)
    reads = []
    cap.read = lambda: (reads.append(1), (True, None))[1]
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda arg: cap)
    got, fps = sources.open_video_source("clip.mp4")
    assert got is cap and fps == 30.0 and reads == []


def _last_error(monkeypatch, sources, src):
    try:
        sources.open_video_source(src)
    except RuntimeError as exc:
        return str(exc)
    return ""
