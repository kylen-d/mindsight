"""Focused unit tests for InferenceScheduler."""
from __future__ import annotations

import numpy as np

from ms.PostProcessing.RayForming.inference_scheduler import InferenceScheduler


PYD = np.array([1.0, 0.0])  # canonical constant direction for fixation tests


def test_fresh_face_wants_inference_once_stable():
    """A newly-tracked face should want its first inference as soon as
    the buffer is stable and PY is fixating (bootstrap exception)."""
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10, min_call_gap=1)
    for i in range(10):  # populate buffer to stable
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    should_fire, wanting = sch.tick()
    assert should_fire is True
    assert wanting == {0}


def test_pursuit_face_does_not_want_inference():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10, min_call_gap=1)
    for i in range(15):
        theta = 0.05 * i    # fast ramp
        d = np.array([np.cos(theta), np.sin(theta)])
        sch.observe(track_id=0, py_dir=d, py_conf=1.0)
    should_fire, wanting = sch.tick()
    assert should_fire is False
    assert wanting == set()


def test_multiple_faces_batched_when_any_wants():
    """One fixating face and one moving face -> should_fire=True, but
    wanting_tids includes only the fixating one."""
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10, min_call_gap=1)
    for i in range(15):
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)              # fixating
        theta = 0.05 * i
        sch.observe(track_id=1, py_dir=np.array([np.cos(theta), np.sin(theta)]),
                    py_conf=1.0)                                       # moving
    should_fire, wanting = sch.tick()
    assert should_fire is True
    assert wanting == {0}


def test_global_min_call_gap_enforced():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10, min_call_gap=5)
    for _ in range(15):
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    # First fire.
    should_fire, wanting = sch.tick()
    assert should_fire is True
    sch.record_accepted(wanting)
    sch.advance_frame()
    # Immediately next frame: gap not elapsed.
    sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    should_fire2, _ = sch.tick()
    assert should_fire2 is False


def test_py_conf_floor_blocks_inference():
    """A face with low PY confidence should never want inference, even
    when fixating."""
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10, min_call_gap=1)
    for _ in range(15):
        sch.observe(track_id=0, py_dir=PYD, py_conf=0.2)   # below PY_CONF_FLOOR (0.5)
    should_fire, wanting = sch.tick()
    assert should_fire is False
    assert wanting == set()


def test_per_face_min_refresh_enforced():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10, min_call_gap=1)
    for _ in range(15):
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    should_fire, wanting = sch.tick()
    sch.record_accepted(wanting)
    sch.advance_frame()
    assert 0 in wanting
    # Now observe a few more frames -- fixation continues, but per-face
    # refresh interval blocks re-request.
    for _ in range(3):
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
        _, wanting2 = sch.tick()
        assert 0 not in wanting2, f"per-face min_refresh violated: {wanting2}"
        sch.advance_frame()


def test_unobserved_tracked_face_excluded_from_fire_decision():
    """A face that is tracked but NOT observed this frame (ReID grace
    period) must not drive should_fire with stale likelihoods."""
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10, min_call_gap=1)
    for _ in range(10):
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    # Face is fixating and would fire this frame...
    should_fire, wanting = sch.tick()
    assert should_fire is True and wanting == {0}
    sch.record_accepted(wanting)
    sch.advance_frame()
    # ...but the next frames it is NOT observed (missed detection).
    # advance past min_call_gap and MIN_FACE_REFRESH with no observations.
    for _ in range(10):
        sch.advance_frame()
    should_fire2, wanting2 = sch.tick()
    assert should_fire2 is False and wanting2 == set(), \
        f"unobserved face drove fire decision with stale data: {wanting2}"


def test_forget_removes_face_state():
    sch = InferenceScheduler(v_threshold=0.02, d_threshold=0.10, min_call_gap=1)
    for _ in range(15):
        sch.observe(track_id=0, py_dir=PYD, py_conf=1.0)
    sch.forget(inactive_tids={0})
    should_fire, wanting = sch.tick()
    # No live faces -> nothing to want.
    assert should_fire is False
    assert wanting == set()
