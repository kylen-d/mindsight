"""Unit tests for the SP3.1 Batch C per-video state-reset seams (Q4/D9).

Two new seams let ``project.runner.iter_project_runs`` give each video fresh
cross-video state without reloading model weights:

* ``GazelleProvider.reset()`` -- rebuilds the fixation scheduler + heatmap
  cache while keeping the loaded Gaze-LLE engine.
* ``factory.rebuild_plugin_instances(ns)`` -- silently re-instantiates the
  phenomena + object-detection plugin instances via their ``from_args``.

The end-to-end acceptance is the Batch C determinism gate (2-copy blend
project); these tests pin the seams' unit behaviour.
"""

from types import SimpleNamespace

import numpy as np

from mindsight.PostProcessing.RayForming.gazelle_provider import GazelleProvider


def _make_provider():
    """A provider over a sentinel engine (reset never touches the engine)."""
    engine = object()
    return engine, GazelleProvider(
        engine, v_threshold=0.04, d_threshold=0.15, min_call_gap=10)


def test_reset_clears_scheduler_and_cache_state():
    _engine, prov = _make_provider()

    # Seed per-track scheduler state + a cached heatmap for a track.
    prov.observe_face(track_id=0, py_dir=np.array([1.0, 0.0]), py_conf=0.5)
    prov.observe_face(track_id=1, py_dir=np.array([0.0, 1.0]), py_conf=0.5)
    prov.heatmap_cache.update(0, np.zeros((64, 64), dtype=np.float32),
                              inout_score=1.0, wanted=True)

    assert prov._scheduler.tracked_tids == {0, 1}
    assert prov.heatmap_cache.track_ids == {0}

    prov.reset()

    # Scheduler + cache are pristine again.
    assert prov._scheduler.tracked_tids == set()
    assert prov.heatmap_cache.track_ids == set()
    assert prov.likelihood(0) == 0.0
    # Global call-gap counter restored so the next video can fire immediately.
    assert prov._scheduler._frames_since_last_global_call >= 10 ** 9


def test_reset_rebuilds_fresh_objects_keeping_engine_and_thresholds():
    engine, prov = _make_provider()
    old_sched = prov._scheduler
    old_cache = prov.heatmap_cache

    prov.reset()

    # Genuinely new sub-objects, not just cleared in place.
    assert prov._scheduler is not old_sched
    assert prov.heatmap_cache is not old_cache
    # Engine (the loaded weights) is preserved -- never reloaded.
    assert prov.engine is engine
    # Thresholds survive the rebuild.
    assert prov._scheduler.v_threshold == 0.04
    assert prov._scheduler.d_threshold == 0.15
    assert prov._scheduler.min_call_gap == 10


def test_rebuild_plugin_instances_empty_namespace():
    from mindsight.factory import rebuild_plugin_instances

    # No plugin flags set -> nothing activates -> (None, None), matching the
    # 14-tuple convention build_from_namespace uses.
    active, detection = rebuild_plugin_instances(SimpleNamespace())
    assert active is None
    assert detection is None


def test_rebuild_plugin_instances_gives_fresh_instances():
    from mindsight.factory import rebuild_plugin_instances

    # gaze_boost is a lightweight object-detection plugin (no model load); its
    # from_args activates on the gaze_boost flag.
    ns = SimpleNamespace(gaze_boost=True)
    active1, detection1 = rebuild_plugin_instances(ns)
    active2, detection2 = rebuild_plugin_instances(ns)

    assert active1 is None and active2 is None
    assert detection1 is not None and detection2 is not None
    assert len(detection1) == 1 and len(detection2) == 1
    # Each rebuild yields a genuinely fresh instance (per-video isolation).
    assert detection1[0] is not detection2[0]
    assert type(detection1[0]) is type(detection2[0])
