"""In/out-of-frame gating on the blend path (v1.1 W3.1).

``--rf-inout-gate T`` (default 0.0 = fully inert, byte-identical to 1.0.0):
* accept VETO: a fresh heatmap whose in-frame score is below T is not
  accepted into the belief map (protects belief + length latch);
* trust attenuation: blend trust is scaled by the cached in/out score;
* variant auto-detect: when the gate is on, the name untyped, and the
  checkpoint carries inout_head params, the ``_inout`` architecture is
  constructed (same weights, plus the in/out output).
"""

from argparse import Namespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mindsight.PostProcessing.RayForming.gazelle_provider import (  # noqa: E402
    _resolve_gazelle_name,
)
from mindsight.PostProcessing.RayForming.ray_config import (  # noqa: E402
    RayFormingConfig,
)
from mindsight.PostProcessing.RayForming.ray_pipeline import (  # noqa: E402
    RawGaze,
    run_ray_forming,
)


# ── Gating math through run_ray_forming ────────────────────────────────────

class _StubProvider:
    """Provider double with a scripted cache state and trust."""

    def __init__(self, heatmap, age, inout, wanted, trust):
        self._state = (heatmap, age, inout, wanted)
        self._trust = trust
        self.heatmap_cache = self

    def get(self, tid):
        return self._state

    def observe_face(self, **kwargs):
        pass

    def likelihood(self, tid):
        return self._trust

    def pop_length_refresh(self, tid):
        return None                        # W3Y channel: nothing pending


class _RecordingBlender:
    def __init__(self):
        self.calls = []

    def update(self, *, track_id, pitch, yaw, gaze_conf, origin, face_width,
               frame_h, frame_w, gazelle_hm, accept_heatmap, trust, dt):
        self.calls.append({'accept': accept_heatmap, 'trust': trust,
                           'hm': gazelle_hm})
        return np.asarray(origin, float) + np.array([50.0, 0.0])


def _run(cfg, provider, blender):
    raw = [RawGaze(origin=np.array([100.0, 100.0]), pitch=0.5, yaw=0.4,
                   confidence=0.2, face_width=80.0, track_id=0,
                   face_bbox=(60, 60, 140, 140))]
    return run_ray_forming(raw, objects=[], face_objs=[], frame_h=480,
                           frame_w=640, cfg=cfg, gazelle_provider=provider,
                           gazelle_blender=blender)


HM = np.full((64, 64), 0.5, dtype=np.float32)


def test_gate_zero_is_inert():
    cfg = RayFormingConfig(forward_gaze_threshold=0.0)   # default gate 0.0
    blender = _RecordingBlender()
    _run(cfg, _StubProvider(HM, 0, 0.05, True, trust=0.8), blender)
    call = blender.calls[0]
    assert call['accept'] is True          # low inout ignored at gate 0
    assert call['trust'] == pytest.approx(0.8)


def test_gate_vetoes_low_inout_accepts():
    cfg = RayFormingConfig(forward_gaze_threshold=0.0, rf_inout_gate=0.5)
    blender = _RecordingBlender()
    result = _run(cfg, _StubProvider(HM, 0, 0.2, True, trust=0.8), blender)
    call = blender.calls[0]
    assert call['accept'] is False         # 0.2 < 0.5 gate
    assert call['hm'] is None
    assert call['trust'] == pytest.approx(0.8 * 0.2)   # attenuated
    assert result.blend_info[0]['accepted'] is False
    assert result.blend_info[0]['trust'] == pytest.approx(0.16)


def test_gate_passes_high_inout_and_attenuates_mildly():
    cfg = RayFormingConfig(forward_gaze_threshold=0.0, rf_inout_gate=0.5)
    blender = _RecordingBlender()
    _run(cfg, _StubProvider(HM, 0, 0.9, True, trust=0.8), blender)
    call = blender.calls[0]
    assert call['accept'] is True          # 0.9 >= 0.5
    assert call['trust'] == pytest.approx(0.8 * 0.9)


def test_stale_heatmap_never_accepts_regardless_of_gate():
    cfg = RayFormingConfig(forward_gaze_threshold=0.0, rf_inout_gate=0.5)
    blender = _RecordingBlender()
    _run(cfg, _StubProvider(HM, age=3, inout=0.9, wanted=True, trust=0.8),
         blender)
    assert blender.calls[0]['accept'] is False


# ── Variant auto-detect ─────────────────────────────────────────────────────

def _ckpt(tmp_path, with_head: bool):
    sd = {"backbone.x": torch.zeros(1)}
    if with_head:
        sd["inout_head.0.weight"] = torch.zeros(1)
        sd["inout_token.weight"] = torch.zeros(1)
    path = tmp_path / "ckpt.pt"
    torch.save(sd, path)
    return path


def test_autodetect_upgrades_default_name_when_gated(tmp_path):
    ns = Namespace(rf_gazelle_name="gazelle_dinov2_vitb14", rf_inout_gate=0.5,
                   _explicit_cli=frozenset())
    assert _resolve_gazelle_name(ns, _ckpt(tmp_path, True)) == \
        "gazelle_dinov2_vitb14_inout"


def test_autodetect_inert_at_gate_zero(tmp_path):
    ns = Namespace(rf_gazelle_name="gazelle_dinov2_vitb14", rf_inout_gate=0.0,
                   _explicit_cli=frozenset())
    assert _resolve_gazelle_name(ns, _ckpt(tmp_path, True)) == \
        "gazelle_dinov2_vitb14"


def test_autodetect_respects_explicit_name(tmp_path):
    ns = Namespace(rf_gazelle_name="gazelle_dinov2_vitb14", rf_inout_gate=0.5,
                   _explicit_cli=frozenset({"rf_gazelle_name"}))
    assert _resolve_gazelle_name(ns, _ckpt(tmp_path, True)) == \
        "gazelle_dinov2_vitb14"


def test_autodetect_no_head_stays_plain(tmp_path):
    ns = Namespace(rf_gazelle_name="gazelle_dinov2_vitb14", rf_inout_gate=0.5,
                   _explicit_cli=frozenset())
    assert _resolve_gazelle_name(ns, _ckpt(tmp_path, False)) == \
        "gazelle_dinov2_vitb14"


def test_autodetect_handles_vitl_names(tmp_path):
    ns = Namespace(rf_gazelle_name="gazelle_dinov2_vitl14", rf_inout_gate=0.5,
                   _explicit_cli=frozenset())
    assert _resolve_gazelle_name(ns, _ckpt(tmp_path, True)) == \
        "gazelle_dinov2_vitl14_inout"
    # already-inout names never double-append
    ns2 = Namespace(rf_gazelle_name="gazelle_dinov2_vitl14_inout",
                    rf_inout_gate=0.5, _explicit_cli=frozenset())
    assert _resolve_gazelle_name(ns2, _ckpt(tmp_path, True)) == \
        "gazelle_dinov2_vitl14_inout"
