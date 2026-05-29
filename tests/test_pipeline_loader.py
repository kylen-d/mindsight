"""Tests for pipeline_loader.py -- YAML pipeline config loading."""

from argparse import Namespace
from pathlib import Path

import pytest

from ms.pipeline_loader import _flatten, _is_default, load_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── _flatten helper ──────────────────────────────────────────────────────────

class TestFlatten:
    """Tests for the internal _flatten helper."""

    def test_flat_dict_unchanged(self):
        """A flat dict gets keys as-is."""
        result = _flatten({'a': 1, 'b': 2})
        assert result == {'a': 1, 'b': 2}

    def test_nested_dict(self):
        """Nested keys are dot-separated."""
        result = _flatten({'detection': {'conf': 0.5, 'model': 'yolo'}})
        assert result == {'detection.conf': 0.5, 'detection.model': 'yolo'}

    def test_deeply_nested(self):
        """Multiple levels of nesting."""
        result = _flatten({'a': {'b': {'c': 1}}})
        assert result == {'a.b.c': 1}

    def test_empty_dict(self):
        """Empty dict returns empty."""
        assert _flatten({}) == {}

    def test_mixed_nesting(self):
        """Mix of flat and nested keys."""
        result = _flatten({'source': 'video.mp4', 'gaze': {'ray_length': 1.5}})
        assert result['source'] == 'video.mp4'
        assert result['gaze.ray_length'] == 1.5


# ── _is_default helper ──────────────────────────────────────────────────────

class TestIsDefault:
    """Tests for the _is_default heuristic."""

    def test_none_is_default(self):
        ns = Namespace(val=None)
        assert _is_default(ns, 'val') is True

    def test_false_is_default(self):
        ns = Namespace(val=False)
        assert _is_default(ns, 'val') is True

    def test_zero_int_is_default(self):
        ns = Namespace(val=0)
        assert _is_default(ns, 'val') is True

    def test_zero_float_is_default(self):
        ns = Namespace(val=0.0)
        assert _is_default(ns, 'val') is True

    def test_empty_string_is_default(self):
        ns = Namespace(val='')
        assert _is_default(ns, 'val') is True

    def test_empty_list_is_default(self):
        ns = Namespace(val=[])
        assert _is_default(ns, 'val') is True

    def test_nonzero_is_not_default(self):
        ns = Namespace(val=42)
        assert _is_default(ns, 'val') is False

    def test_true_is_not_default(self):
        ns = Namespace(val=True)
        assert _is_default(ns, 'val') is False

    def test_nonempty_string_is_not_default(self):
        ns = Namespace(val='hello')
        assert _is_default(ns, 'val') is False

    def test_missing_attr_is_default(self):
        ns = Namespace()
        assert _is_default(ns, 'nonexistent') is True


# ── load_pipeline ────────────────────────────────────────────────────────────

class TestLoadPipeline:
    """Tests for load_pipeline with temp YAML files."""

    def test_file_not_found(self, tmp_path):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "nonexistent.yaml")

    def test_empty_yaml(self, tmp_path):
        """Empty YAML file returns a namespace without errors."""
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        ns = load_pipeline(cfg_file)
        assert isinstance(ns, Namespace)

    def test_source_key(self, tmp_path):
        """Top-level 'source' key is mapped to ns.source."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text("source: video.mp4\n")
        ns = load_pipeline(cfg_file)
        assert ns.source == "video.mp4"

    def test_nested_detection_keys(self, tmp_path):
        """Nested detection section maps to flat attributes."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "detection:\n"
            "  conf: 0.6\n"
            "  detect_scale: 0.5\n"
        )
        ns = load_pipeline(cfg_file)
        assert ns.conf == 0.6
        assert ns.detect_scale == 0.5

    def test_nested_gaze_keys(self, tmp_path):
        """Nested gaze section maps to flat attributes."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "gaze:\n"
            "  ray_length: 2.0\n"
            "  snap_dist: 200.0\n"
        )
        ns = load_pipeline(cfg_file)
        assert ns.ray_length == 2.0
        assert ns.snap_dist == 200.0

    def test_output_keys(self, tmp_path):
        """Output section maps correctly."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "output:\n"
            "  save_video: output.mp4\n"
            "  log_csv: log.csv\n"
        )
        ns = load_pipeline(cfg_file)
        assert ns.save == "output.mp4"
        assert ns.log == "log.csv"

    def test_existing_namespace_updated(self, tmp_path):
        """Passing an existing namespace updates it."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text("source: cam.mp4\n")
        ns = Namespace(existing_attr="keep_me")
        result = load_pipeline(cfg_file, ns)
        assert result is ns
        assert result.source == "cam.mp4"
        assert result.existing_attr == "keep_me"

    def test_cli_override_preserves_nondefault(self, tmp_path):
        """CLI-set values (non-default) are not overwritten by YAML."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "detection:\n"
            "  conf: 0.8\n"
        )
        ns = Namespace(conf=0.5)  # Non-default (0.5 != 0)
        load_pipeline(cfg_file, ns)
        # 0.5 is non-default, so YAML's 0.8 should not overwrite
        assert ns.conf == 0.5

    def test_cli_override_default_is_overwritten(self, tmp_path):
        """CLI values at their default (e.g. 0) are overwritten by YAML."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "detection:\n"
            "  conf: 0.7\n"
        )
        ns = Namespace(conf=0)  # Default-like value
        load_pipeline(cfg_file, ns)
        assert ns.conf == 0.7


# ── Phenomena list parsing ───────────────────────────────────────────────────

class TestPhenomenaListParsing:
    """Tests for phenomena list parsing from YAML."""

    def test_simple_toggle(self, tmp_path):
        """A simple string in the phenomena list enables the tracker."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "phenomena:\n"
            "  - mutual_gaze\n"
        )
        ns = load_pipeline(cfg_file)
        assert ns.mutual_gaze is True

    def test_multiple_toggles(self, tmp_path):
        """Multiple phenomena toggles are all enabled."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "phenomena:\n"
            "  - mutual_gaze\n"
            "  - gaze_aversion\n"
            "  - scanpath\n"
        )
        ns = load_pipeline(cfg_file)
        assert ns.mutual_gaze is True
        assert ns.gaze_aversion is True
        assert ns.scanpath is True

    def test_toggle_with_params(self, tmp_path):
        """Phenomenon with params sets both the toggle and the params."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "phenomena:\n"
            "  - joint_attention:\n"
            "      ja_window: 45\n"
            "      ja_quorum: 0.8\n"
        )
        ns = load_pipeline(cfg_file)
        assert ns.joint_attention is True
        assert ns.ja_window == 45
        assert ns.ja_quorum == 0.8

    def test_unknown_phenomenon_ignored(self, tmp_path):
        """Unknown phenomenon names are silently ignored."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "phenomena:\n"
            "  - nonexistent_tracker\n"
        )
        ns = load_pipeline(cfg_file)
        assert not hasattr(ns, 'nonexistent_tracker')

    def test_phenomena_as_dict_not_list(self, tmp_path):
        """If phenomena is a dict (not a list), the list parsing is skipped."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "phenomena:\n"
            "  ja_window: 30\n"
        )
        ns = load_pipeline(cfg_file)
        # When phenomena is a dict, it goes through _YAML_MAP not the list path
        # The _YAML_MAP has 'phenomena.ja_window' -> 'ja_window'
        assert ns.ja_window == 30

    def test_gaze_aversion_with_params(self, tmp_path):
        """Gaze aversion with custom aversion_window param."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "phenomena:\n"
            "  - gaze_aversion:\n"
            "      aversion_window: 120\n"
            "      aversion_conf: 0.7\n"
        )
        ns = load_pipeline(cfg_file)
        assert ns.gaze_aversion is True
        assert ns.aversion_window == 120
        assert ns.aversion_conf == 0.7


# ── Explicit-flag precedence (CLI route) ─────────────────────────────────────

class TestExplicitFlagPrecedence:
    """Tests for the ``_explicit_cli`` precedence path.

    When the namespace carries ``_explicit_cli`` (the exact set of dests the
    user typed on the CLI, attached by ms.cli._args), YAML overwrites every
    dest NOT in that set -- the truthy-default _is_default heuristic that used
    to silently drop YAML values is bypassed.
    """

    def test_empty_explicit_set_applies_all_yaml(self, tmp_path):
        """With _explicit_cli=frozenset() (nothing typed), YAML wins even for
        keys whose truthy parser default previously blocked it."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "detection:\n"
            "  conf: 0.05\n"
            "  detect_scale: 0.75\n"
            "gaze:\n"
            "  ray_length: 1.5\n"
            "  adaptive_ray: extend\n"
        )
        # Parser-default-like namespace: every attr truthy, so legacy
        # _is_default would DROP all four YAML values.
        ns = Namespace(conf=0.35, detect_scale=1.0, ray_length=1.0,
                       adaptive_ray='off')
        ns._explicit_cli = frozenset()
        load_pipeline(cfg_file, ns)
        assert ns.conf == 0.05
        assert ns.detect_scale == 0.75
        assert ns.ray_length == 1.5
        assert ns.adaptive_ray == 'extend'

    def test_explicit_dest_not_overwritten(self, tmp_path):
        """A dest listed in _explicit_cli keeps its CLI value; others take
        their YAML values."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "detection:\n"
            "  conf: 0.05\n"
            "gaze:\n"
            "  ray_length: 1.5\n"
        )
        ns = Namespace(conf=0.9, ray_length=1.0)  # user typed --conf 0.9
        ns._explicit_cli = frozenset({'conf'})
        load_pipeline(cfg_file, ns)
        assert ns.conf == 0.9      # explicit -- YAML does NOT override
        assert ns.ray_length == 1.5  # not explicit -- YAML applies

    def test_explicit_phenomena_toggle_and_params(self, tmp_path):
        """Phenomena toggles/params honor the explicit set too."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text(
            "phenomena:\n"
            "  - joint_attention:\n"
            "      ja_window: 45\n"
        )
        ns = Namespace(joint_attention=False, ja_window=0)
        ns._explicit_cli = frozenset()
        load_pipeline(cfg_file, ns)
        assert ns.joint_attention is True
        assert ns.ja_window == 45

    def test_without_explicit_legacy_heuristic(self, tmp_path):
        """No _explicit_cli attr => legacy _is_default behavior (byte-for-byte:
        a truthy non-default value blocks the YAML value)."""
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text("detection:\n  conf: 0.05\n")
        ns = Namespace(conf=0.35)  # truthy => _is_default False => blocked
        load_pipeline(cfg_file, ns)
        assert ns.conf == 0.35  # legacy loader drops the YAML value


# ── End-to-end: real _args parse + YAML merge ────────────────────────────────

def test_args_yaml_honored_over_unset_flags_source_wins():
    """Parse real argv through ms.cli._args, then apply the repo pipeline YAML
    exactly as main() does.  The explicitly-typed --source must survive while
    unset flags take their YAML values (the whole point of Fix 1)."""
    from ms.cli import _args

    argv = ["--pipeline", "test_pipeline.yaml", "--source", "x.mp4"]
    ns = _args(argv)
    # _explicit_cli reflects exactly what was typed.
    assert ns._explicit_cli == frozenset({"pipeline", "source"})

    load_pipeline(REPO_ROOT / ns.pipeline, ns)

    assert ns.source == "x.mp4"        # explicit -- NOT overwritten by YAML
    assert ns.conf == 0.05             # unset flag -- YAML wins
    assert ns.ray_length == 1.5        # unset flag -- YAML wins
    assert ns.adaptive_ray == "extend"  # truthy-default dest, now honored
