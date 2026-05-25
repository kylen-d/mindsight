"""Pins the three-way min_call_gap resolution shared across CLI, config,
provider, and GUI load paths.  Regression guard for the int(None) crash
when --min-call-gap defaults to None in argparse."""
from __future__ import annotations

from argparse import Namespace

from ms.PostProcessing.RayForming.ray_config import resolve_min_call_gap


def test_explicit_min_call_gap_wins():
    ns = Namespace(min_call_gap=15, rf_gazelle_interval=99)
    assert resolve_min_call_gap(ns) == 15


def test_legacy_interval_used_when_min_call_gap_none():
    ns = Namespace(min_call_gap=None, rf_gazelle_interval=12)
    assert resolve_min_call_gap(ns) == 12


def test_default_when_both_none():
    ns = Namespace(min_call_gap=None, rf_gazelle_interval=None)
    assert resolve_min_call_gap(ns) == 30


def test_default_when_attrs_absent():
    ns = Namespace()   # neither attr present
    assert resolve_min_call_gap(ns) == 30


def test_returns_int_not_none():
    # The bug: int(None) crash.  Guard that the result is always a usable int.
    for ns in (Namespace(min_call_gap=None, rf_gazelle_interval=None),
               Namespace(min_call_gap=None, rf_gazelle_interval=7),
               Namespace(min_call_gap=3, rf_gazelle_interval=None)):
        result = resolve_min_call_gap(ns)
        assert isinstance(result, int)
