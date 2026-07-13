"""Bundled-resource resolution (wheel installs) -- mindsight/resources.py.

A checkout has no ``mindsight/_bundled/`` tree, so every lookup must behave
exactly as before (the PROJECT_ROOT file, or None/error).  When the staging
tree exists (wheel installs, built via scripts/sync_bundled_resources.py) it
backfills only what PROJECT_ROOT lacks -- an explicit PROJECT_ROOT copy
always wins so installs can override shipped resources.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mindsight import config_compat, constants, resources, weights

REPO = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = REPO / "scripts" / "sync_bundled_resources.py"


# ── resources.py primitives ──────────────────────────────────────────────────

def test_bundled_path_none_without_staging_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "_BUNDLED", tmp_path / "absent")
    assert resources.bundled_path("weights_manifest.json") is None


def test_resource_path_prefers_project_root(monkeypatch, tmp_path):
    root = tmp_path / "root"
    bundled = tmp_path / "bundled"
    for d in (root, bundled):
        d.mkdir()
        (d / "thing.txt").write_text(d.name)
    monkeypatch.setattr(constants, "PROJECT_ROOT", root)
    monkeypatch.setattr(resources, "_BUNDLED", bundled)
    assert resources.resource_path("thing.txt").read_text() == "root"


def test_resource_path_falls_back_to_bundled(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "thing.txt").write_text("bundled")
    monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path / "empty")
    monkeypatch.setattr(resources, "_BUNDLED", bundled)
    assert resources.resource_path("thing.txt").read_text() == "bundled"


def test_resource_path_none_when_neither(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path / "empty")
    monkeypatch.setattr(resources, "_BUNDLED", tmp_path / "absent")
    assert resources.resource_path("thing.txt") is None


# ── call-site fallbacks ──────────────────────────────────────────────────────

def test_known_good_preset_bundled_fallback(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    (bundled / "configs").mkdir(parents=True)
    preset = bundled / "configs" / "pipeline_known_good.yaml"
    preset.write_text("pipeline: {}\n")
    monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path / "empty")
    monkeypatch.setattr(resources, "_BUNDLED", bundled)
    assert config_compat.known_good_preset_path() == preset


def test_known_good_preset_checkout_still_wins(monkeypatch, tmp_path):
    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    checkout_preset = root / "configs" / "pipeline_known_good.yaml"
    checkout_preset.write_text("pipeline: {}\n")
    bundled = tmp_path / "bundled"
    (bundled / "configs").mkdir(parents=True)
    (bundled / "configs" / "pipeline_known_good.yaml").write_text("x: 1\n")
    monkeypatch.setattr(constants, "PROJECT_ROOT", root)
    monkeypatch.setattr(resources, "_BUNDLED", bundled)
    assert config_compat.known_good_preset_path() == checkout_preset


def test_load_manifest_bundled_fallback(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "weights_manifest.json").write_text(json.dumps({"weights": []}))
    monkeypatch.setattr(weights, "MANIFEST_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(resources, "_BUNDLED", bundled)
    assert weights.load_manifest() == {"weights": []}


def test_load_manifest_still_errors_when_neither(monkeypatch, tmp_path):
    monkeypatch.setattr(weights, "MANIFEST_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(resources, "_BUNDLED", tmp_path / "no-bundle")
    with pytest.raises(weights.WeightsError, match="manifest not found"):
        weights.load_manifest()


def test_docs_root_bundled_fallback(monkeypatch, tmp_path):
    from mindsight.GUI import about_tab

    bundled = tmp_path / "bundled"
    (bundled / "docs").mkdir(parents=True)
    monkeypatch.setattr(about_tab, "repo_root", lambda: tmp_path / "empty")
    monkeypatch.setattr(resources, "_BUNDLED", bundled)
    assert about_tab.docs_root() == bundled / "docs"


# ── the staging script itself ────────────────────────────────────────────────

def test_sync_script_stages_census_and_cleans(tmp_path):
    target = tmp_path / "staged"
    run = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--target", str(target)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "PASS" in run.stdout
    for rel in ("docs/index.md", "docs/studies/run-a-study-tutorial.md",
                "configs/pipeline_known_good.yaml", "weights_manifest.json"):
        assert (target / rel).is_file(), f"census file missing: {rel}"
    clean = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--target", str(target), "--clean"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert clean.returncode == 0
    assert not target.exists()
