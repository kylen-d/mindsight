"""W3X fire-decision knobs: --rf-onset-samples / --rf-onset-gap / --rf-reuse-eps.

All three default off (0), reproducing the 1.0.0 fire behavior exactly --
the existing scheduler/provider tests pin that.  These tests pin the
opt-in behaviors: earlier bootstrap eligibility, the relaxed global gap
for never-anchored faces, and perceptual refire suppression.
"""
from __future__ import annotations

import numpy as np
import pytest

from mindsight.PostProcessing.RayForming.gazelle_provider import (
    GazelleProvider,
    GazelleReuseGate,
)
from mindsight.PostProcessing.RayForming.heatmap_cache import HeatmapCache
from mindsight.PostProcessing.RayForming.inference_scheduler import InferenceScheduler
from mindsight.PostProcessing.RayForming.py_history import PYHistoryBuffer

PYD = np.array([1.0, 0.0])


# ── PYHistoryBuffer.min_stable ────────────────────────────────────────────────

def test_min_stable_none_keeps_half_buffer_rule():
    buf = PYHistoryBuffer(size=10)
    for _ in range(4):
        buf.push(PYD)
    assert buf.unstable is True
    buf.push(PYD)
    assert buf.unstable is False


def test_min_stable_override_clears_earlier():
    buf = PYHistoryBuffer(size=10, min_stable=2)
    buf.push(PYD)
    assert buf.unstable is True
    buf.push(PYD)
    assert buf.unstable is False


def test_min_stable_below_two_rejected():
    with pytest.raises(ValueError):
        PYHistoryBuffer(size=10, min_stable=1)


# ── Scheduler onset_samples ──────────────────────────────────────────────────

def test_onset_samples_allows_earlier_first_fire():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10,
                             min_call_gap=1, onset_samples=2)
    for _ in range(2):
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    should_fire, wanting = sch.tick()
    assert should_fire is True
    assert wanting == {0}


def test_onset_samples_zero_keeps_default_warmup():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10,
                             min_call_gap=1)
    for _ in range(2):
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    should_fire, _ = sch.tick()
    assert should_fire is False


def test_onset_samples_clamped_to_two():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10,
                             min_call_gap=1, onset_samples=1)
    sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    assert sch.tick()[0] is False          # one sample can never qualify
    sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    assert sch.tick()[0] is True


# ── Scheduler onset_gap ──────────────────────────────────────────────────────

def _drive_first_fire(sch, tid=0, frames=10):
    for _ in range(frames):
        sch.observe(track_id=tid, py_dir=PYD, py_conf=1.0)
    should_fire, wanting = sch.tick()
    assert should_fire is True
    sch.record_accepted(wanting)
    sch.advance_frame()


def test_onset_gap_relaxes_global_gap_for_new_face():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10,
                             min_call_gap=30, onset_gap=2, onset_samples=2)
    _drive_first_fire(sch, tid=0)
    # A new face appears right after track 0's fire; it becomes eligible
    # after 2 samples and should fire once 2 frames have passed since the
    # global call (onset_gap) -- not 30 (min_call_gap).
    sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    sch.observe(track_id=1, py_dir=PYD, py_conf=1.0)
    assert sch.tick()[0] is False          # new face has only 1 sample
    sch.advance_frame()
    sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    sch.observe(track_id=1, py_dir=PYD, py_conf=1.0)
    should_fire, wanting = sch.tick()      # 2 frames since global call
    assert should_fire is True
    assert wanting == {1}                  # track 0 still inside min-refresh


def test_onset_gap_ignored_when_all_wanting_are_latched():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10,
                             min_call_gap=30, onset_gap=2)
    _drive_first_fire(sch, tid=0)
    for _ in range(6):                     # well past onset_gap, below 30
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
        should_fire, _ = sch.tick()
        assert should_fire is False
        sch.advance_frame()


# ── GazelleReuseGate unit behavior ───────────────────────────────────────────

def _frame(value=100, w=64, h=48):
    return np.full((h, w, 3), value, dtype=np.uint8)


BBOX = (10, 10, 30, 30)


def _primed_gate(eps=1.0):
    gate = GazelleReuseGate(eps)
    cache = HeatmapCache()
    cache.update(0, np.ones((64, 64), dtype=np.float32), inout_score=0.7)
    gate.record_fire(_frame(), {0: BBOX})
    return gate, cache


def test_reuse_identical_scene_reanchors_cache():
    gate, cache = _primed_gate()
    cache.age_all({0})                     # simulate frames passing
    assert cache.get(0)[1] == 1            # age advanced
    assert gate.try_reuse(_frame(), {0}, {0: BBOX}, cache) is True
    hm, age, inout, wanted = cache.get(0)
    assert age == 0 and wanted is True and inout == 0.7
    assert gate.hits == 1


def test_reuse_misses_on_changed_scene():
    gate, cache = _primed_gate()
    assert gate.try_reuse(_frame(180), {0}, {0: BBOX}, cache) is False
    assert gate.misses == 1


def test_reuse_misses_on_moved_bbox():
    gate, cache = _primed_gate()
    moved = (40, 10, 60, 30)               # zero IoU vs fire-time bbox
    assert gate.try_reuse(_frame(), {0}, {0: moved}, cache) is False


def test_reuse_misses_without_cached_heatmap():
    gate, cache = _primed_gate()
    assert gate.try_reuse(_frame(), {0, 1}, {0: BBOX, 1: BBOX}, cache) is False


def test_reuse_misses_before_first_fire():
    gate = GazelleReuseGate(1.0)
    assert gate.try_reuse(_frame(), {0}, {0: BBOX}, HeatmapCache()) is False


def test_prune_drops_dead_tracks():
    gate, _cache = _primed_gate()
    gate.prune(set())
    assert gate._bboxes == {}


# ── Provider-level integration: static scene skips the forward pass ─────────

class _CountingEngine:
    def __init__(self):
        self.calls = 0

    def raw_heatmaps(self, frame, bboxes):
        self.calls += 1
        return np.ones((len(bboxes), 64, 64), dtype=np.float32)


def _provider(reuse_eps):
    return GazelleProvider(_CountingEngine(), v_threshold=0.02,
                           d_threshold=0.10, min_call_gap=1,
                           onset_samples=2, reuse_eps=reuse_eps)


def _step_fixating(provider, frame):
    provider.observe_face(track_id=0, py_dir=PYD, py_conf=1.0)
    provider.step(frame, [BBOX], [0])


def test_provider_reuses_on_static_scene():
    p = _provider(reuse_eps=1.0)
    frame = _frame()
    for _ in range(8):
        _step_fixating(p, frame)
    assert p.engine.calls == 1             # first real fire, rest reused
    assert p.reuse_gate.hits >= 1


def test_provider_refires_when_scene_changes():
    p = _provider(reuse_eps=1.0)
    for i in range(8):
        _step_fixating(p, _frame(40 + 25 * i))
    assert p.engine.calls > 1


def test_provider_eps_zero_disables_gate():
    p = _provider(reuse_eps=0.0)
    assert p.reuse_gate is None
    frame = _frame()
    for _ in range(8):
        _step_fixating(p, frame)
    assert p.engine.calls > 1


def test_provider_reset_clears_reuse_gate():
    p = _provider(reuse_eps=1.0)
    frame = _frame()
    for _ in range(4):
        _step_fixating(p, frame)
    p.reset()
    assert p.reuse_gate is not None and p.reuse_gate.hits == 0
    for _ in range(4):
        _step_fixating(p, frame)
    assert p.engine.calls >= 2             # fresh gate: first post-reset fire is real


# ── Flag plumbing ────────────────────────────────────────────────────────────

def test_flags_reach_schema_and_provider_defaults():
    from mindsight.cli_flags import parse_cli
    from mindsight.config import PipelineConfig

    ns = parse_cli([])
    cfg = PipelineConfig.from_namespace(ns)
    assert cfg.rayforming.rf_reuse_eps == 0.0
    assert cfg.rayforming.rf_onset_samples == 0
    assert cfg.rayforming.rf_onset_gap == 0

    ns = parse_cli(["--rf-reuse-eps", "1.5",
                    "--rf-onset-samples", "3",
                    "--rf-onset-gap", "5"])
    cfg = PipelineConfig.from_namespace(ns)
    assert cfg.rayforming.rf_reuse_eps == 1.5
    assert cfg.rayforming.rf_onset_samples == 3
    assert cfg.rayforming.rf_onset_gap == 5
