"""Unit coverage for mindsight.io.sources source normalization (UP1 D1).

``open_video_source`` must hand cv2 an ``int`` for a digit-only camera-index
string (cv2.VideoCapture("0") does not open a webcam) while leaving file-path
strings untouched.  No real capture is opened -- cv2.VideoCapture is stubbed.
"""

import mindsight.io.sources as sources


class _FakeCapture:
    def __init__(self, arg):
        self.arg = arg

    def isOpened(self):
        return True

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
    class _Closed(_FakeCapture):
        def isOpened(self):
            return False

    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda arg: _Closed(arg))
    import pytest
    with pytest.raises(RuntimeError):
        sources.open_video_source("3")
