"""Sweep engine tests (W4C item 1) — GUI-free.

Covers combo expansion (order, cap, rejects), override isolation on the
prepared namespaces, the time estimate, manifest allocate/save/latest,
and winner pick.  The dialog's worker loop is tested offscreen in
test_validation_autotune.py.
"""

import copy
import json
from argparse import Namespace

import pytest

from mindsight.validation import (
    COMBO_CAP,
    CURATED_KNOBS,
    ValidationSet,
    ValidationSetError,
    allocate_sweep_path,
    estimate_seconds,
    expand_combos,
    latest_sweep,
    new_sweep_manifest,
    pick_winner,
    prepare_sweep_namespace,
    save_sweep,
)


# ── expand_combos ────────────────────────────────────────────────────────────

def test_expand_single_knob_order():
    combos = expand_combos([("rf_len_gain", [1.0, 1.1, 1.2])])
    assert combos == [{"rf_len_gain": 1.0}, {"rf_len_gain": 1.1},
                      {"rf_len_gain": 1.2}]


def test_expand_two_knobs_cartesian_first_outermost():
    combos = expand_combos([("rf_len_gain", [1.0, 1.1]),
                            ("min_call_gap", [30, 45, 60])])
    assert len(combos) == 6
    assert combos[0] == {"rf_len_gain": 1.0, "min_call_gap": 30}
    assert combos[2] == {"rf_len_gain": 1.0, "min_call_gap": 60}
    assert combos[3] == {"rf_len_gain": 1.1, "min_call_gap": 30}


def test_expand_at_cap_passes_over_cap_raises():
    at_cap = expand_combos([("a", list(range(4))), ("b", list(range(3)))])
    assert len(at_cap) == COMBO_CAP
    with pytest.raises(ValidationSetError, match="cap"):
        expand_combos([("a", list(range(4))), ("b", list(range(4)))])


@pytest.mark.parametrize("knobs,msg", [
    ([], "at least one"),
    ([("a", [1]), ("b", [1]), ("c", [1])], "at most two"),
    ([("a", [1, 2]), ("a", [3])], "twice"),
    ([("a", [])], "no values"),
    ([("source", ["x"])], "cannot be swept"),
])
def test_expand_rejects(knobs, msg):
    with pytest.raises(ValidationSetError, match=msg):
        expand_combos(knobs)


def test_curated_knobs_are_sweepable():
    for dest, _label, cast in CURATED_KNOBS:
        assert expand_combos([(dest, [cast(1)])]) == [{dest: cast(1)}]


# ── estimate ─────────────────────────────────────────────────────────────────

def test_estimate_seconds():
    assert estimate_seconds(6, 900, 15.0) == pytest.approx(360.0)
    assert estimate_seconds(6, 900, None) is None
    assert estimate_seconds(6, 900, 0) is None
    assert estimate_seconds(6, 0, 15.0) is None


# ── prepare_sweep_namespace ──────────────────────────────────────────────────

def _base_ns():
    return Namespace(source="0", summary=None, log=None, save="out.mp4",
                     heatmap=None, charts=None, no_dashboard=False,
                     save_detections=False, rf_len_gain=1.1,
                     min_call_gap=30)


def _vset(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    vset = ValidationSet(name="s", video=str(video))
    vset.set_label(10, "0", {"x": 1, "y": 1})
    return vset


def test_prepare_applies_overrides_and_isolates_base(tmp_path):
    base = _base_ns()
    frozen = copy.deepcopy(vars(base))
    vset = _vset(tmp_path)
    ns_a = prepare_sweep_namespace(base, vset, tmp_path / "run-001",
                                   {"rf_len_gain": 1.3})
    ns_b = prepare_sweep_namespace(base, vset, tmp_path / "run-002",
                                   {"rf_len_gain": 1.5, "min_call_gap": 60})
    assert vars(base) == frozen                      # base untouched
    assert ns_a.rf_len_gain == 1.3 and ns_a.min_call_gap == 30
    assert ns_b.rf_len_gain == 1.5 and ns_b.min_call_gap == 60
    # Ordinary run-target rewrite still happened.
    assert ns_a.source == vset.video
    assert ns_a.save is None and ns_a.save_detections is True
    assert str(tmp_path / "run-001") in ns_a.summary


def test_prepare_rejects_runner_owned_override(tmp_path):
    with pytest.raises(ValidationSetError, match="cannot be swept"):
        prepare_sweep_namespace(_base_ns(), _vset(tmp_path),
                                tmp_path / "run-001", {"summary": "x"})


# ── manifest ─────────────────────────────────────────────────────────────────

def test_manifest_allocate_save_latest_roundtrip(tmp_path):
    p1 = allocate_sweep_path(tmp_path, "My Set")
    assert p1.name == "sweep-001.json"
    m1 = new_sweep_manifest("My Set", [("rf_len_gain", [1.0, 1.1])])
    m1["results"].append({"overrides": {"rf_len_gain": 1.0},
                          "run": "run-001",
                          "score": {"endpoint_px_mean": 50.0},
                          "error": None})
    save_sweep(p1, m1)
    assert latest_sweep(tmp_path, "My Set") == json.loads(p1.read_text())

    p2 = allocate_sweep_path(tmp_path, "My Set")
    assert p2.name == "sweep-002.json"
    m2 = new_sweep_manifest("My Set", [("conf", [0.3])])
    save_sweep(p2, m2)
    assert latest_sweep(tmp_path, "My Set")["knobs"] == [["conf", [0.3]]]

    # Unreadable newest file falls back to the previous one.
    p2.write_text("{broken")
    assert latest_sweep(tmp_path, "My Set")["set"] == "My Set"
    assert latest_sweep(tmp_path, "My Set")["results"][0]["run"] == "run-001"


def test_latest_sweep_none_for_unswept_set(tmp_path):
    assert latest_sweep(tmp_path, "nope") is None


# ── winner ───────────────────────────────────────────────────────────────────

def test_pick_winner_min_mean_skipping_failures():
    results = [
        {"overrides": {}, "run": "run-001", "score": None, "error": "boom"},
        {"overrides": {}, "run": "run-002",
         "score": {"endpoint_px_mean": 61.5}, "error": None},
        {"overrides": {}, "run": "run-003",
         "score": {"endpoint_px_mean": 58.8}, "error": None},
        {"overrides": {}, "run": "run-004", "score": {}, "error": None},
    ]
    assert pick_winner(results) == 2
    assert pick_winner(results[:1]) is None
    assert pick_winner([]) is None
