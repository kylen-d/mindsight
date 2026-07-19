"""W4B length-slew rework: --rf-len-slew slews the EFFECTIVE target.

History: the W3Z version slewed the LATCH while the len_hold_tau decay
kept pulling the target toward the PY baseline -- two mechanisms
fighting at the refresh cadence read as ~3Hz length BOUNCE on real
footage (user eyes-on; the default-5 flip was reverted in w3z-8).

The rework (ruled): a re-latch snaps the latch and resets the age clock
as before, but the OUTPUT length target ramps linearly from the reach
the ray last showed to the new latch over N ``update()`` calls, with
the hold decay PAUSED for the duration (age frozen at 0; decay resumes
after arrival).  Monotone approach, no interaction with the decay.
Default stays 0 (snap) -- goldens byte-identical.
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


def _eff(b, tid=0) -> float:
    """Pre-gain effective length target recorded by the last update."""
    return b._last_eff_len[tid]


# Peak at grid (32, 24) on 640x480 -> pixel (320, 180) -> length from (0,0).
_TARGET = float(np.hypot(320.0, 180.0))


def test_slew_zero_snaps_instantly():
    b = _blender(slew=0, latched=100.0)
    assert b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                            origin=ORIGIN, frame_h=480, frame_w=640)
    assert b._latched_lle_length[0] == _TARGET
    assert b._latch_age_s[0] == 0.0
    assert b._len_slew == {}


def test_effective_target_ramps_monotonically_and_arrives():
    k = 5
    b = _blender(slew=k, latched=100.0)
    _update(b)                                  # record current effective
    start_eff = _eff(b)
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    # W4B rework: the LATCH snaps immediately; the ramp lives in the
    # effective target, not the latch.
    assert b._latched_lle_length[0] == _TARGET
    assert b._latch_age_s[0] == 0.0
    seen = []
    for _ in range(k):
        _update(b)
        seen.append(_eff(b))
    expect = [start_eff + (_TARGET - start_eff) * (i / k)
              for i in range(1, k + 1)]
    np.testing.assert_allclose(seen, expect, rtol=1e-9)
    assert seen[-1] == _TARGET                  # exact arrival
    assert b._len_slew == {}                    # slew consumed
    assert all(y > x for x, y in zip(seen, seen[1:]))     # MONOTONE


def test_hold_decay_paused_during_slew_resumes_after():
    k = 4
    b = _blender(slew=k, latched=100.0)
    _update(b)
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    for _ in range(k):
        _update(b)                              # non-accept updates
    # Decay was PAUSED: age never accrued while slewing.
    assert b._latch_age_s[0] == 0.0
    assert _eff(b) == _TARGET
    # After arrival the decay resumes normally.
    _update(b)
    assert b._latch_age_s[0] > 0.0
    assert _eff(b) < _TARGET                    # decaying toward PY baseline


def test_mid_slew_relatch_restarts_from_shown_value():
    k = 4
    b = _blender(slew=k, latched=100.0)
    _update(b)
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    _update(b)                                  # 1/4 of the way
    mid = _eff(b)
    assert mid < _TARGET
    # A shorter re-latch mid-slew ramps DOWN from `mid` -- no jump.
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(4, 3),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    new_target = b._latched_lle_length[0]
    start, done = b._len_slew[0]
    assert start == mid and done == 0
    assert new_target < mid
    seen = []
    for _ in range(k):
        _update(b)
        seen.append(_eff(b))
    assert seen[-1] == new_target
    assert all(y < x for x, y in zip([mid] + seen, seen))  # monotone down


def test_first_latch_snaps_even_with_slew_on():
    b = _blender(slew=5)                        # no latch yet
    _update(b, hm=_peak_hm(32, 24), accept=True)
    assert b._latched_lle_length[0] == _TARGET
    assert b._len_slew == {}


def test_accept_path_relatch_slews_too():
    k = 5
    b = _blender(slew=k, latched=100.0)
    _update(b)                                  # record shown effective
    start_eff = _eff(b)
    _update(b, hm=_peak_hm(32, 24), accept=True)
    # The accepting update itself advances the ramp one step.
    np.testing.assert_allclose(
        _eff(b), start_eff + (_TARGET - start_eff) / k)
    assert b._latched_lle_length[0] == _TARGET
    assert b._latch_age_s[0] == 0.0


def test_prune_clears_slew_state():
    b = _blender(slew=5, latched=100.0)
    _update(b)                                  # prune walks _beliefs
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                     origin=ORIGIN, frame_h=480, frame_w=640)
    assert b._len_slew
    b.prune(set())
    assert b._len_slew == {}
    assert b._last_eff_len == {}


def test_flag_reaches_schema_with_default_off():
    from mindsight.cli_flags import parse_cli
    from mindsight.config import PipelineConfig

    ns = parse_cli([])
    assert PipelineConfig.from_namespace(ns).rayforming.rf_len_slew == 0
    assert RayFormingConfig.from_namespace(ns).rf_len_slew == 0

    ns = parse_cli(["--rf-len-slew", "5"])       # opt-in
    assert PipelineConfig.from_namespace(ns).rayforming.rf_len_slew == 5
    assert RayFormingConfig.from_namespace(ns).rf_len_slew == 5
