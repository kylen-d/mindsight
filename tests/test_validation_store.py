"""Tests for the validation-set store (v1.1 W4B, validation suite phase 1).

The on-disk contract matters most: a set file must remain a valid
eval-harness labels file (``labels`` in exactly the
``eval_data/{stem}_labels.json`` shape) so ``scripts/eval_gaze.py``
scores validation sets unchanged.
"""

import json

import pytest

from mindsight.validation import (
    LABEL_STATES,
    ValidationSet,
    ValidationSetError,
    ValidationStore,
    validation_root,
)


def _sample_set() -> ValidationSet:
    vset = ValidationSet(name="office-a", video="/tmp/clip.mp4", every=10)
    vset.set_label(120, "P0", {"x": 451, "y": 475})
    vset.set_label(120, "P1", "offscreen")
    vset.set_label(130, "P0", {"x": 10.7, "y": 20.2})     # coerced to int
    vset.add_object(120, "plate", (100, 200, 180, 260))
    return vset


# ── Data model ────────────────────────────────────────────────────────────────

def test_round_trip_preserves_everything(tmp_path):
    store = ValidationStore(tmp_path)
    path = store.save(_sample_set())
    loaded = store.load("office-a")
    assert path.name == "office-a.json"
    assert loaded.video == "/tmp/clip.mp4" and loaded.every == 10
    assert loaded.labels[120]["P0"] == {"x": 451, "y": 475}
    assert loaded.labels[120]["P1"] == "offscreen"
    assert loaded.labels[130]["P0"] == {"x": 10, "y": 20}
    assert loaded.objects[120] == [
        {"name": "plate", "x1": 100, "y1": 200, "x2": 180, "y2": 260}]
    assert loaded.frames() == [120, 130]
    assert loaded.point_label_count() == 2


def test_eval_harness_label_compatibility(tmp_path):
    """The saved file IS an eval labels file: string frame keys under
    'labels', values are {pid: {x, y} | offscreen/uncertain/skip} — the
    exact shape scripts/eval_gaze.py::_load_labels returns."""
    store = ValidationStore(tmp_path)
    data = json.loads(store.save(_sample_set()).read_text())
    labels = data["labels"]
    assert set(labels) == {"120", "130"}          # string keys like the harness
    assert labels["120"]["P0"] == {"x": 451, "y": 475}
    assert labels["120"]["P1"] in LABEL_STATES
    # The scoring branches: dict -> distance, state string -> special-case.
    for per_frame in labels.values():
        for v in per_frame.values():
            assert isinstance(v, dict) or v in LABEL_STATES


def test_label_and_object_validation():
    vset = ValidationSet(name="x")
    with pytest.raises(ValidationSetError, match="Unknown label state"):
        vset.set_label(1, "P0", "nonsense")
    with pytest.raises(ValidationSetError, match="point"):
        vset.set_label(1, "P0", {"x": 1})
    with pytest.raises(ValidationSetError, match="Degenerate"):
        vset.add_object(1, "plate", (50, 50, 50, 90))


def test_editing_helpers():
    vset = _sample_set()
    vset.clear_label(120, "P1")
    assert "P1" not in vset.labels[120]
    vset.remove_object(120, 0)
    assert 120 not in vset.objects
    vset.remove_frame(120)
    assert vset.frames() == [130]
    vset.add_frame(500)
    assert vset.frames() == [130, 500]


# ── Store behavior ────────────────────────────────────────────────────────────

def test_list_sets_skips_unreadable(tmp_path):
    store = ValidationStore(tmp_path)
    store.save(_sample_set())
    (tmp_path / "junk.json").write_text("{not json")
    (tmp_path / "other.json").write_text(json.dumps({"no_labels": True}))
    sets = store.list_sets()
    assert [s["name"] for s in sets] == ["office-a"]
    assert sets[0]["frames"] == 2 and sets[0]["points"] == 2


def test_slug_and_delete(tmp_path):
    store = ValidationStore(tmp_path)
    vset = _sample_set()
    vset.name = "Lab cam #2 (wide)"
    path = store.save(vset)
    assert path.name == "Lab-cam-2-wide.json"
    store.delete("Lab cam #2 (wide)")
    assert not path.exists()
    with pytest.raises(ValidationSetError):
        ValidationStore(tmp_path).path_for("###")


def test_validation_root_project_vs_state(tmp_path, monkeypatch):
    assert validation_root(tmp_path / "proj") == tmp_path / "proj" / "validation"
    monkeypatch.setenv("MINDSIGHT_STATE_DIR", str(tmp_path / "state"))
    assert validation_root() == tmp_path / "state" / "validation"
