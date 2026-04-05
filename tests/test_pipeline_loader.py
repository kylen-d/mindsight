"""Tests for pipeline_loader.py -- YAML pipeline config loading."""

from argparse import Namespace

import pytest

from ms.pipeline_loader import _flatten, _is_default, load_pipeline

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
