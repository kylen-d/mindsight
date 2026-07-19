"""Tests for the pipeline `validation:` metadata block (W4B phase 5).

The load-bearing property is the ruled canonical_hash CARVE-OUT:
embedding or editing validation metadata must never move the hash (and
so never force a resume-ledger reprocess), while real config changes
still must.  Plus: YAML loader pass-through, section shape enforcement,
and the embed helper's leave-everything-else-alone contract.
"""

import yaml

import pytest

from mindsight.config import PipelineConfig, ValidationSection
from mindsight.config_compat import load_yaml
from mindsight.validation import (
    ValidationSet,
    ValidationSetError,
    embed_validation_summary,
    validation_summary_block,
)

_BLOCK = {
    "set_name": "office-a",
    "n_frames": 24,
    "date": "2026-07-18",
    "metrics": {"endpoint_px_mean": 65.2, "hit_rate": 0.7},
    "settings_hash": "abcd1234abcd1234",
}


# ── The carve-out ─────────────────────────────────────────────────────────────

def test_canonical_hash_ignores_validation_block():
    base = PipelineConfig()
    with_block = PipelineConfig(validation=_BLOCK)
    edited = PipelineConfig(validation={**_BLOCK, "n_frames": 999})
    assert base.canonical_hash() == with_block.canonical_hash()
    assert with_block.canonical_hash() == edited.canonical_hash()
    # Real config still moves it.
    real_change = PipelineConfig(detection={"conf": 0.5})
    assert real_change.canonical_hash() != base.canonical_hash()


def test_validation_section_shape_enforced():
    ValidationSection(**_BLOCK)                       # valid
    with pytest.raises(Exception):
        ValidationSection(set_name="x", bogus_field=1)
    with pytest.raises(Exception):
        PipelineConfig(validation={"bogus_field": 1})


# ── YAML loader pass-through ──────────────────────────────────────────────────

def test_load_yaml_carries_validation_block(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({
        "gaze": {"ray_length": 2.0},
        "validation": dict(_BLOCK),
    }))
    cfg = load_yaml(path)
    assert cfg.validation.set_name == "office-a"
    assert cfg.validation.metrics["hit_rate"] == 0.7
    assert cfg.gaze.ray_length == 2.0
    # And the block still does not influence the hash through the loader.
    path2 = tmp_path / "pipeline2.yaml"
    path2.write_text(yaml.dump({"gaze": {"ray_length": 2.0}}))
    assert load_yaml(path2).canonical_hash() == cfg.canonical_hash()


def test_load_yaml_without_block_defaults(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump({"gaze": {"ray_length": 2.0}}))
    cfg = load_yaml(path)
    assert cfg.validation.set_name is None
    assert cfg.validation.metrics is None


# ── Block builder + embed helper ──────────────────────────────────────────────

def test_validation_summary_block_shape():
    vset = ValidationSet(name="s", video="v.mp4")
    vset.set_label(10, "0", {"x": 1, "y": 2})
    vset.set_label(20, "0", {"x": 3, "y": 4})
    score = {"scored_points": 2, "endpoint_px_mean": 65.2, "hit_rate": 0.7,
             "offscreen_auc": None, "hit_radius_px": 80}
    block = validation_summary_block(vset, score,
                                     settings={"conf": 0.5},
                                     date="2026-07-18")
    ValidationSection(**block)                        # accepted by schema
    assert block["set_name"] == "s" and block["n_frames"] == 2
    assert block["metrics"] == {"scored_points": 2,
                                "endpoint_px_mean": 65.2, "hit_rate": 0.7}
    assert len(block["settings_hash"]) == 16
    # Same settings -> same hash; different -> different.
    b2 = validation_summary_block(vset, score, settings={"conf": 0.5})
    assert b2["settings_hash"] == block["settings_hash"]
    b3 = validation_summary_block(vset, score, settings={"conf": 0.9})
    assert b3["settings_hash"] != block["settings_hash"]


def test_embed_preserves_other_yaml_keys(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(yaml.dump({"gaze": {"ray_length": 2.0},
                               "phenomena": ["mutual_gaze"]}))
    embed_validation_summary(path, dict(_BLOCK))
    data = yaml.safe_load(path.read_text())
    assert data["gaze"] == {"ray_length": 2.0}
    assert data["phenomena"] == ["mutual_gaze"]
    assert data["validation"]["set_name"] == "office-a"
    # Re-embed replaces, not duplicates.
    embed_validation_summary(path, {**_BLOCK, "n_frames": 30})
    data = yaml.safe_load(path.read_text())
    assert data["validation"]["n_frames"] == 30
    # And the result loads cleanly through the real loader.
    assert load_yaml(path).validation.n_frames == 30


def test_embed_rejects_bad_block_and_bad_file(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(yaml.dump({"gaze": {}}))
    with pytest.raises(Exception):
        embed_validation_summary(path, {"bogus_field": 1})
    listy = tmp_path / "l.yaml"
    listy.write_text(yaml.dump(["not", "a", "mapping"]))
    with pytest.raises(ValidationSetError, match="not a pipeline mapping"):
        embed_validation_summary(listy, dict(_BLOCK))
