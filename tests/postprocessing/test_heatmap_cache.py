"""Focused unit tests for HeatmapCache -- pins the wanted-flag lifecycle.

The wanted flag gates whether the GazeLLEBlender applies a cached
heatmap to its belief map.  A lifecycle regression here (e.g. dropping
the age_all-clears-wanted step, or reordering the provider's
age_all/update calls) would silently make every cached face "wanted
forever" with no crash -- exactly the class of bug worth a cheap pin.
"""
from __future__ import annotations

import numpy as np

from ms.PostProcessing.RayForming.heatmap_cache import HeatmapCache


def _hm():
    return np.zeros((64, 64), dtype=np.float32)


def test_update_then_get_returns_wanted_true_at_age_zero():
    c = HeatmapCache()
    c.update(0, _hm(), inout_score=0.8, wanted=True)
    hm, age, inout, wanted = c.get(0)
    assert hm is not None
    assert age == 0
    assert inout == 0.8
    assert wanted is True


def test_age_all_clears_wanted_and_increments_age():
    c = HeatmapCache()
    c.update(0, _hm(), wanted=True)
    c.age_all({0})
    hm, age, inout, wanted = c.get(0)
    assert hm is not None
    assert age == 1
    assert wanted is False


def test_provider_call_order_leaves_fresh_fire_wanted():
    """The provider ages FIRST, then updates fresh fires -- a fresh
    heatmap must survive its fire frame with wanted=True."""
    c = HeatmapCache()
    c.update(0, _hm(), wanted=True)     # frame N fire
    # Frame N+1: age first (stale entry loses wanted), then fresh fire.
    c.age_all({0})
    c.update(0, _hm(), wanted=True)
    hm, age, inout, wanted = c.get(0)
    assert age == 0 and wanted is True


def test_miss_returns_none_tuple():
    c = HeatmapCache()
    assert c.get(99) == (None, -1, 0.0, False)


def test_legacy_positional_update_defaults_wanted_true():
    c = HeatmapCache()
    c.update(1, _hm(), 0.5)   # legacy 3-arg positional call
    _, _, inout, wanted = c.get(1)
    assert inout == 0.5
    assert wanted is True


def test_unwanted_entry_stays_unwanted():
    """A track included in the batch but NOT flagged by the scheduler
    caches with wanted=False and never flips to True by aging."""
    c = HeatmapCache()
    c.update(0, _hm(), wanted=False)
    _, age, _, wanted = c.get(0)
    assert age == 0 and wanted is False
    c.age_all({0})
    _, age, _, wanted = c.get(0)
    assert age == 1 and wanted is False


def test_prune_inactive_removes_all_state():
    c = HeatmapCache()
    c.update(0, _hm(), wanted=True)
    c.age_all(active_tids=set())   # track 0 no longer active
    assert c.get(0) == (None, -1, 0.0, False)
    assert c.track_ids == set()
