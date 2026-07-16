"""RetinaFace configurability (v1.1 W2.4).

--face-conf / --face-input-size reach the uniface constructor; the defaults
are the library defaults so an unconfigured build behaves exactly as 1.0.
"""

import sys
import types

from mindsight.cli_flags import parse_cli
from mindsight.ObjectDetection.model_factory import create_face_detector


class _RecorderRetinaFace:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecorderRetinaFace.instances.append(self)


def _with_fake_uniface(monkeypatch):
    fake = types.ModuleType("uniface")
    fake.RetinaFace = _RecorderRetinaFace
    monkeypatch.setitem(sys.modules, "uniface", fake)
    _RecorderRetinaFace.instances.clear()


def test_defaults_match_library_defaults(monkeypatch):
    _with_fake_uniface(monkeypatch)
    create_face_detector()
    kwargs = _RecorderRetinaFace.instances[-1].kwargs
    assert kwargs == {"conf_thresh": 0.5, "input_size": (640, 640)}


def test_flags_reach_constructor(monkeypatch):
    _with_fake_uniface(monkeypatch)
    ns = parse_cli(["--face-conf", "0.3", "--face-input-size", "960"])
    create_face_detector(conf_thresh=ns.face_conf,
                         input_size=ns.face_input_size)
    kwargs = _RecorderRetinaFace.instances[-1].kwargs
    assert kwargs == {"conf_thresh": 0.3, "input_size": (960, 960)}


def test_flag_defaults():
    ns = parse_cli([])
    assert ns.face_conf == 0.5
    assert ns.face_input_size == 640
    assert ns.face_model is None


def test_face_model_selects_backbone(monkeypatch):
    _with_fake_uniface(monkeypatch)
    constants = types.ModuleType("uniface.constants")
    constants.RetinaFaceWeights = lambda v: f"enum:{v}"
    monkeypatch.setitem(sys.modules, "uniface.constants", constants)
    create_face_detector(model_name="r34")
    kwargs = _RecorderRetinaFace.instances[-1].kwargs
    assert kwargs["model_name"] == "enum:retinaface_r34"


def test_face_model_none_keeps_library_default(monkeypatch):
    _with_fake_uniface(monkeypatch)
    create_face_detector(model_name=None)
    assert "model_name" not in _RecorderRetinaFace.instances[-1].kwargs
