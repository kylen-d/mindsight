"""W3Z length-slew: --rf-len-slew.

Default 0 (the W3Z flip to 5 was reverted same-day: slewing the latch
while the hold decay pulls the target toward the PY baseline reads as
BOUNCE on real footage -- a rework should slew the effective target
instead).  With N > 0 a refresh that re-latches an EXISTING length no
longer snaps; the latch slews old -> new linearly over the next N
``update()`` calls.  First-ever latches still snap (nothing to slew from),
and the age clock resets at slew START so the len_hold_tau decay does not
double-count the transition.
"""
from __future__ import annotations

import numpy as np

from mindsight.PostProcessing.RayForming.gazelle_blender import GazeLLEBlender
from mindsight.PostProcessing.RayForming.ray_config import RayFormingConfig

ORIGIN = np.array([0.0, 0.0])


def _peak_hm(gx: int, gy: int) -> np.ndarray:
    hm = np.zeros((64, 64), dtype=np.float32)
    hm[gy, gx] = 1.0
    return hm


def _blender(slew=0, latched=None, tid=0):
    b = GazeLLEBlender(RayFormingConfig(rf_len_slew=slew))
    if latched is not None:
        b._latched_lle_length[tid] = latched
        b._latch_age_s[tid] = 3.0
    return b


def _update(b, tid=0, hm=None, accept=False):
    b.update(track_id=tid, pitch=0.1, yaw=0.2, gaze_conf=0.5,
             origin=ORIGIN, face_width=50.0, frame_h=480, frame_w=640,
             gazelle_hm=hm, accept_heatmap=accept, trust=0.5, dt=1 / 30)


# Peak at grid (32, 24) on 640x480 -> pixel (320, 180) -> length from (0,0).
_TARGET = float(np.hypot(320.0, 180.0))


def test_slew_zero_snaps_instantly():
    b = _blender(slew=0, latched=100.0)
    assert b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                            origin=ORIGIN, frame_h=480, frame_w=640)
    assert b._latched_lle_length[0] == _TARGET
    assert b._latch_age_s[0] == 0.0
    assert b._len_slew == {}


def test_slew_trajectory_linear_arrival_in_k_updates():
    k = 5
    b = _blender(slew=k, latched=100.0)
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    # Latch untouched until an update advances the slew.
    assert b._latched_lle_length[0] == 100.0
    assert b._latch_age_s[0] == 0.0            # age resets at slew START
    seen = []
    for _ in range(k):
        _update(b)
        seen.append(b._latched_lle_length[0])
    expect = [100.0 + (_TARGET - 100.0) * (i / k) for i in range(1, k + 1)]
    np.testing.assert_allclose(seen, expect, rtol=1e-9)
    assert seen[-1] == _TARGET                 # exact arrival, slew consumed
    assert b._len_slew == {}
    assert all(b2 > a for a, b2 in zip(seen, seen[1:]))   # monotonic


def test_further_updates_after_arrival_hold_the_target():
    b = _blender(slew=2, latched=100.0)
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    for _ in range(4):
        _update(b)
    assert b._latched_lle_length[0] == _TARGET


def test_mid_slew_refresh_restarts_from_interpolated_value():
    k = 4
    b = _blender(slew=k, latched=100.0)
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    _update(b)                                  # 1/4 of the way
    mid = b._latched_lle_length[0]
    assert 100.0 < mid < _TARGET
    # New refresh back toward a shorter length restarts from `mid`.
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(4, 3),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    start, target, done = b._len_slew[0]
    assert start == mid and done == 0
    assert target < mid                         # heading back down
    for _ in range(k):
        _update(b)
    assert b._latched_lle_length[0] == target


def test_first_latch_snaps_even_with_slew_on():
    b = _blender(slew=5)                        # no latch yet
    _update(b, hm=_peak_hm(32, 24), accept=True)
    assert b._latched_lle_length[0] == _TARGET
    assert b._len_slew == {}


def test_accept_path_relatch_slews_too():
    k = 5
    b = _blender(slew=k, latched=100.0)
    # The accepting update itself advances one step (start then advance).
    _update(b, hm=_peak_hm(32, 24), accept=True)
    np.testing.assert_allclose(
        b._latched_lle_length[0], 100.0 + (_TARGET - 100.0) / k)
    assert b._latch_age_s[0] == 0.0


def test_hold_decay_ages_normally_during_slew():
    b = _blender(slew=10, latched=100.0)
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    for _ in range(3):
        _update(b)                              # non-accept: age accrues
    assert 0.0 < b._latch_age_s[0] < 0.2        # 3 * dt, from the slew reset


def test_prune_clears_slew_state():
    b = _blender(slew=5, latched=100.0)
    _update(b)                                  # prune walks _beliefs
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    assert b._len_slew
    b.prune(set())
    assert b._len_slew == {}


def test_flag_reaches_schema_with_default_off():
    from mindsight.cli_flags import parse_cli
    from mindsight.config import PipelineConfig

    ns = parse_cli([])
    assert PipelineConfig.from_namespace(ns).rayforming.rf_len_slew == 0  # flip reverted
    assert RayFormingConfig.from_namespace(ns).rf_len_slew == 0

    ns = parse_cli(["--rf-len-slew", "5"])       # opt-in
    assert PipelineConfig.from_namespace(ns).rayforming.rf_len_slew == 5
    assert RayFormingConfig.from_namespace(ns).rf_len_slew == 5
