"""W3Z items 3a/3b: --rf-len-gain and --rf-endpoint-extract.

Both default to the historical behavior (gain 1.0, centroid extraction) --
the smoke/CSV goldens pin that.  The eval decomposition that motivated them:
84% of eval rays measured too short (pred 197 px vs true 233 px along-ray).
"""
from __future__ import annotations

import numpy as np

from mindsight.PostProcessing.RayForming.gazelle_blender import (
    GazeLLEBlender,
    _heatmap_centroid_pixel,
    _heatmap_topp_pixel,
)
from mindsight.PostProcessing.RayForming.ray_config import RayFormingConfig

ORIGIN = np.array([0.0, 0.0])


def _update(b, tid=0, hm=None, accept=False):
    return b.update(track_id=tid, pitch=0.1, yaw=0.2, gaze_conf=0.5,
                    origin=ORIGIN, face_width=50.0, frame_h=480, frame_w=640,
                    gazelle_hm=hm, accept_heatmap=accept, trust=0.0, dt=1 / 30)


# ── 3a: length gain ──────────────────────────────────────────────────────────

def test_gain_scales_endpoint_reach():
    base = GazeLLEBlender(RayFormingConfig(rf_len_gain=1.0))
    gained = GazeLLEBlender(RayFormingConfig(rf_len_gain=1.10))
    e1 = _update(base)
    e2 = _update(gained)
    # trust 0, no latch -> pure PY path; endpoint distance scales by 1.10.
    np.testing.assert_allclose(np.linalg.norm(e2 - ORIGIN),
                               1.10 * np.linalg.norm(e1 - ORIGIN), rtol=1e-9)


def test_gain_one_is_byte_inert():
    a = GazeLLEBlender(RayFormingConfig())
    b = GazeLLEBlender(RayFormingConfig(rf_len_gain=1.0))
    np.testing.assert_array_equal(_update(a), _update(b))


def test_gain_applies_to_latched_length_too():
    cfg = RayFormingConfig(rf_len_gain=2.0, rf_len_slew=0)
    b = GazeLLEBlender(cfg)
    b._latched_lle_length[0] = 300.0
    b._latch_age_s[0] = 0.0
    e = _update(b)
    ref = GazeLLEBlender(RayFormingConfig(rf_len_slew=0))
    ref._latched_lle_length[0] = 300.0
    ref._latch_age_s[0] = 0.0
    e_ref = _update(ref)
    np.testing.assert_allclose(np.linalg.norm(e - ORIGIN),
                               2.0 * np.linalg.norm(e_ref - ORIGIN), rtol=1e-9)


# ── 3b: top-p extraction ─────────────────────────────────────────────────────

def _bimodal_hm():
    """A sharp far peak plus a diffuse near blob of equal total mass."""
    hm = np.zeros((64, 64), dtype=np.float32)
    hm[50, 50] = 10.0                       # dominant sharp mode (far)
    hm[8:14, 8:14] += 10.0 / 36.0           # diffuse equal-mass blob (near)
    return hm


def test_topp_tracks_the_dominant_mode():
    hm = _bimodal_hm()
    cen = _heatmap_centroid_pixel(hm, 480, 640)
    top = _heatmap_topp_pixel(hm, 480, 640)
    # Full centroid lands between the modes; top-p sits on the sharp peak.
    peak_px = np.array([50 / 64 * 640, 50 / 64 * 480])
    assert np.linalg.norm(top - peak_px) < 12.0
    assert np.linalg.norm(cen - peak_px) > 100.0


def test_topp_equals_centroid_on_unimodal_maps():
    hm = np.zeros((64, 64), dtype=np.float32)
    hm[24, 32] = 1.0
    np.testing.assert_allclose(_heatmap_topp_pixel(hm, 480, 640),
                               _heatmap_centroid_pixel(hm, 480, 640))


def test_topp_empty_map_falls_back_to_center():
    hm = np.zeros((64, 64), dtype=np.float32)
    np.testing.assert_allclose(_heatmap_topp_pixel(hm, 480, 640),
                               [320.0, 240.0])


def test_extract_mode_reaches_the_latch():
    hm = _bimodal_hm()
    for mode in ("centroid", "topp"):
        b = GazeLLEBlender(RayFormingConfig(rf_endpoint_extract=mode,
                                            rf_len_slew=0))
        _update(b, hm=hm, accept=True)
        expect = (_heatmap_topp_pixel if mode == "topp"
                  else _heatmap_centroid_pixel)(hm, 480, 640)
        np.testing.assert_allclose(b._latched_lle_length[0],
                                   float(np.linalg.norm(expect - ORIGIN)))


def test_refresh_length_honors_extract_mode():
    hm = _bimodal_hm()
    b = GazeLLEBlender(RayFormingConfig(rf_endpoint_extract="topp",
                                        rf_len_slew=0))
    b._latched_lle_length[0] = 100.0
    b._latch_age_s[0] = 0.0
    assert b.refresh_length(track_id=0, gazelle_hm=hm, origin=ORIGIN,
                            frame_h=480, frame_w=640)
    expect = _heatmap_topp_pixel(hm, 480, 640)
    np.testing.assert_allclose(b._latched_lle_length[0],
                               float(np.linalg.norm(expect - ORIGIN)))


# ── Flag plumbing ────────────────────────────────────────────────────────────

def test_flags_reach_schema_and_ray_config():
    from mindsight.cli_flags import parse_cli
    from mindsight.config import PipelineConfig

    ns = parse_cli([])
    cfg = PipelineConfig.from_namespace(ns)
    assert cfg.rayforming.rf_len_gain == 1.0
    assert cfg.rayforming.rf_endpoint_extract == "centroid"
    rc = RayFormingConfig.from_namespace(ns)
    assert rc.rf_len_gain == 1.0 and rc.rf_endpoint_extract == "centroid"

    ns = parse_cli(["--rf-len-gain", "1.1", "--rf-endpoint-extract", "topp"])
    cfg = PipelineConfig.from_namespace(ns)
    assert cfg.rayforming.rf_len_gain == 1.1
    assert cfg.rayforming.rf_endpoint_extract == "topp"
