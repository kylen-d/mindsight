"""W3Y cheap length-refresh channel: --rf-len-refresh-gap.

Default off (0), reproducing the existing blend behavior exactly -- the
smoke/CSV goldens pin that.  These tests pin the opt-in behavior: a
counter-gated second engine whose heatmaps refresh ray LENGTH only --
never the belief map, never direction, and never a latch that the
fixation-gated fp32 channel has not established first.
"""
from __future__ import annotations

import numpy as np

from mindsight.PostProcessing.RayForming.gazelle_blender import GazeLLEBlender
from mindsight.PostProcessing.RayForming.gazelle_provider import GazelleProvider
from mindsight.PostProcessing.RayForming.inference_scheduler import InferenceScheduler
from mindsight.PostProcessing.RayForming.ray_config import RayFormingConfig
from mindsight.PostProcessing.RayForming.ray_pipeline import RawGaze, run_ray_forming

PYD = np.array([1.0, 0.0])
BBOX = (10, 10, 30, 30)


def _frame(w=64, h=48):
    return np.full((h, w, 3), 100, dtype=np.uint8)


def _peak_hm(gx: int, gy: int) -> np.ndarray:
    hm = np.zeros((64, 64), dtype=np.float32)
    hm[gy, gx] = 1.0
    return hm


# ── Scheduler channel ────────────────────────────────────────────────────────

def test_length_channel_off_by_default():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10,
                             min_call_gap=1)
    for _ in range(5):
        assert sch.tick_length_refresh() is False
        sch.advance_frame()


def test_length_channel_fires_on_counter():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10,
                             min_call_gap=1, length_refresh_gap=3)
    assert sch.tick_length_refresh() is True        # first fire immediate
    sch.record_length_refresh()
    fired_at = []
    for i in range(7):
        sch.advance_frame()
        if sch.tick_length_refresh():
            fired_at.append(i)
            sch.record_length_refresh()
    assert fired_at == [2, 5]                       # every 3rd frame


def test_full_accept_resets_length_counter():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10,
                             min_call_gap=1, length_refresh_gap=3)
    sch.record_length_refresh()
    sch.advance_frame()
    sch.advance_frame()
    sch.record_accepted({0})                        # fp32 just re-latched
    sch.advance_frame()
    assert sch.tick_length_refresh() is False       # countdown restarted


# ── Blender: length-only application ─────────────────────────────────────────

def _blender_with_latch(tid=0, latched=200.0, age=3.0):
    b = GazeLLEBlender(RayFormingConfig())
    b._latched_lle_length[tid] = latched
    b._latch_age_s[tid] = age
    return b


def test_refresh_length_updates_existing_latch_only():
    b = _blender_with_latch()
    origin = np.array([0.0, 0.0])
    # Peak at grid (32, 24) on a 640x480 frame -> pixel (320, 180).
    ok = b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                          origin=origin, frame_h=480, frame_w=640)
    assert ok is True
    assert b._latched_lle_length[0] == np.hypot(320.0, 180.0)
    assert b._latch_age_s[0] == 0.0


def test_refresh_length_never_creates_a_latch():
    b = GazeLLEBlender(RayFormingConfig())
    ok = b.refresh_length(track_id=0, gazelle_hm=_peak_hm(32, 24),
                          origin=np.array([0.0, 0.0]),
                          frame_h=480, frame_w=640)
    assert ok is False
    assert 0 not in b._latched_lle_length


def test_refresh_length_leaves_belief_and_direction_alone():
    b = _blender_with_latch()
    # Establish belief state via one non-accepted update.
    b.update(track_id=0, pitch=0.1, yaw=0.2, gaze_conf=0.5,
             origin=np.array([100.0, 100.0]), face_width=50.0,
             frame_h=480, frame_w=640, gazelle_hm=None,
             accept_heatmap=False, trust=0.5, dt=1 / 30)
    belief_before = b._beliefs[0].copy()
    b.refresh_length(track_id=0, gazelle_hm=_peak_hm(10, 10),
                     origin=np.array([100.0, 100.0]),
                     frame_h=480, frame_w=640)
    np.testing.assert_array_equal(b._beliefs[0], belief_before)


# ── Provider: the cheap engine only ever feeds length refreshes ──────────────

class _CountingEngine:
    def __init__(self, inout=None):
        self.calls = 0
        self.fired_at: list[int] = []
        self.step_no = 0                    # set by the test loop
        self._inout = inout
        self._last_inout = None

    def raw_heatmaps(self, frame, bboxes):
        self.calls += 1
        self.fired_at.append(self.step_no)
        self._last_inout = (
            np.full(len(bboxes), self._inout, dtype=np.float32)
            if self._inout is not None else None)
        return np.ones((len(bboxes), 64, 64), dtype=np.float32)


def _length_provider(gap=3, main=None, length=None):
    main = main or _CountingEngine()
    length = length or _CountingEngine()
    return GazelleProvider(main, v_threshold=0.02, d_threshold=0.10,
                           min_call_gap=1, onset_samples=2,
                           length_engine=length, length_refresh_gap=gap), \
        main, length


def test_length_engine_feeds_side_dict_not_cache():
    p, main, length = _length_provider(gap=2)
    for _ in range(6):                     # no fixation -> main quiet
        p.step(_frame(), [BBOX], [0])
    assert main.calls == 0
    assert length.calls >= 2
    assert p.heatmap_cache.get(0)[0] is None        # cache untouched
    pending = p.pop_length_refresh(0)
    assert pending is not None
    hm, inout = pending
    assert hm.shape == (64, 64) and inout == 1.0
    assert p.pop_length_refresh(0) is None          # consumed exactly once


def test_length_pass_suppressed_on_full_fire_frames():
    p, main, length = _length_provider(gap=1)
    for i in range(12):
        main.step_no = length.step_no = i
        p.observe_face(track_id=0, py_dir=PYD, py_conf=1.0)
        p.step(_frame(), [BBOX], [0])
    assert main.calls >= 2
    assert length.calls >= 1
    assert not set(main.fired_at) & set(length.fired_at)


def test_gap_zero_keeps_channel_inert():
    p = GazelleProvider(_CountingEngine(), v_threshold=0.02,
                        d_threshold=0.10, min_call_gap=1,
                        length_engine=_CountingEngine(),
                        length_refresh_gap=0)
    assert p._length_engine is None
    for _ in range(6):
        p.step(_frame(), [BBOX], [0])
    assert p.pop_length_refresh(0) is None


def test_reset_and_prune_clear_pending_refreshes():
    p, _main, _length = _length_provider(gap=1)
    p.step(_frame(), [BBOX], [0])
    assert p._length_refresh
    p.reset()
    assert p._length_refresh == {}
    p.step(_frame(), [BBOX], [0])
    p.step(_frame(), [BBOX], [1])          # track 0 gone -> pruned
    assert 0 not in p._length_refresh


# ── Pipeline: length-only application + in/out veto ──────────────────────────

def _run_once(provider, blender, cfg, tid=0):
    rg = RawGaze(origin=np.array([100.0, 100.0]), pitch=0.3, yaw=0.3,
                 confidence=0.02,            # below PY_CONF_FLOOR: main quiet
                 face_width=50.0, track_id=tid, face_bbox=BBOX)
    return run_ray_forming([rg], [], [], 480, 640, cfg,
                           gazelle_provider=provider,
                           gazelle_blender=blender)


def test_pipeline_applies_length_refresh_to_latch():
    cfg = RayFormingConfig()
    p, _main, _length = _length_provider(gap=1)
    b = _blender_with_latch(latched=500.0)
    p.step(_frame(), [BBOX], [0])          # length pass fires
    _run_once(p, b, cfg)
    assert b._latched_lle_length[0] != 500.0
    assert b._latch_age_s[0] < 1.0         # re-latched this frame


def test_pipeline_inout_veto_blocks_length_refresh():
    cfg = RayFormingConfig(rf_inout_gate=0.5)
    length = _CountingEngine(inout=0.1)    # off-screen: below the gate
    p, _main, _length = _length_provider(gap=1, length=length)
    b = _blender_with_latch(latched=500.0, age=3.0)
    p.step(_frame(), [BBOX], [0])
    _run_once(p, b, cfg)
    assert b._latched_lle_length[0] == 500.0        # veto held
    assert p.pop_length_refresh(0) is None          # still consumed


# ── Flag plumbing ────────────────────────────────────────────────────────────

def test_flag_reaches_schema_with_flipped_default():
    from mindsight.cli_flags import parse_cli
    from mindsight.config import PipelineConfig

    ns = parse_cli([])
    cfg = PipelineConfig.from_namespace(ns)
    assert cfg.rayforming.rf_len_refresh_gap == 10   # W3Y flip default

    ns = parse_cli(["--rf-len-refresh-gap", "0"])    # escape hatch
    cfg = PipelineConfig.from_namespace(ns)
    assert cfg.rayforming.rf_len_refresh_gap == 0
    assert RayFormingConfig.from_namespace(ns).rf_len_refresh_gap == 0
