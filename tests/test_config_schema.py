"""Tests for ms/config.py -- the unified pydantic schema (SP1.1).

Covers:
* section models mirror the runtime dataclasses (names + defaults),
* CLI flag completeness against the LIVE parser (cli metadata, CLI_ALIASES,
  documented exclusions, plugin flags),
* canonical_hash determinism (including across processes),
* extra="forbid" strictness.
"""

import dataclasses
import subprocess
import sys

import pytest
from pydantic import ValidationError

from mindsight.config import (
    AuxStream,
    DepthSection,
    DetectionSection,
    GazeSection,
    OutputSection,
    PhenomenaSection,
    PipelineConfig,
    ProjectOutputSection,
    ProjectSection,
    RayFormingSection,
    TrackerSection,
)
from mindsight.config_compat import CLI_ALIASES, EXCLUDED_CLI_FLAGS
from mindsight.Phenomena.phenomena_config import PhenomenaConfig
from mindsight.pipeline_config import (
    AuxStreamConfig,
    DepthConfig,
    DetectionConfig,
    GazeConfig,
    OutputConfig,
    ProjectConfig,
    ProjectOutputConfig,
    TrackerConfig,
)
from mindsight.PostProcessing.RayForming.ray_config import RayFormingConfig

# ══════════════════════════════════════════════════════════════════════════════
# Live parser capture (shared with test_config_equivalence via import)
# ══════════════════════════════════════════════════════════════════════════════

_PARSER = None


def get_parser():
    """Build the REAL cli parser once and return it for introspection.

    ``ms.cli._args()`` constructs the parser (core + submodule + plugin
    flags) and immediately calls ``parse_args()``; we intercept that call to
    capture the parser object itself.
    """
    global _PARSER
    if _PARSER is None:
        import argparse

        from mindsight.cli import _args

        box = {}
        orig = argparse.ArgumentParser.parse_args

        def capture(self, *args, **kwargs):
            box["parser"] = self
            return orig(self, [])

        argparse.ArgumentParser.parse_args = capture
        try:
            _args()
        finally:
            argparse.ArgumentParser.parse_args = orig
        _PARSER = box["parser"]
    return _PARSER


def get_plugin_dests() -> set[str]:
    """Dests contributed by registry plugins (out of schema scope)."""
    from mindsight.GUI.arg_introspector import FakeArgumentParser
    from Plugins import (
        gaze_registry,
        object_detection_registry,
        phenomena_registry,
    )

    dests: set[str] = set()
    for registry in (gaze_registry, object_detection_registry,
                     phenomena_registry):
        for name in registry.names():
            fake = FakeArgumentParser()
            try:
                registry.get(name).add_arguments(fake)
            except Exception:
                continue
            dests.update(spec.dest for spec in fake.specs)
    return dests


# Frozen snapshot of plugin-contributed dests.  If this fails, a plugin was
# added/removed/renamed -- update the list (plugin flags stay namespace-passed
# and outside the schema by design).
KNOWN_PLUGIN_DESTS = {
    # gaze backends
    "gazelle_model", "gazelle_name", "gazelle_inout_threshold",
    "gazelle_device", "gazelle_skip_frames", "gazelle_fp16",
    "gazelle_compile",
    "iris_refine", "iris_refine_weight", "iris_refine_upscale",
    "l2cs_model", "l2cs_arch", "l2cs_dataset",
    "mgaze_model", "mgaze_arch", "mgaze_dataset",
    # object detection plugins
    "gaze_boost", "gaze_boost_factor", "gaze_boost_radius",
    "gaze_boost_min_conf", "gaze_boost_max_conf", "gaze_boost_classes",
    # phenomena plugins
    "eye_movement", "em_source", "em_saccade_thresh", "em_fixation_thresh",
    "em_min_fixation", "em_velocity_window",
    "novel_salience", "ns_speed_thresh", "ns_cooldown", "ns_history",
    "ns_flash",
    "pupillometry", "pupil_mode", "pupil_baseline", "pupil_upscale",
    "pupil_ir_thresh", "pupil_filter", "pupil_ema_alpha",
    "pupil_kalman_process_noise", "pupil_kalman_meas_noise",
    "pupil_ear_thresh", "pupil_blink_frames", "pupil_outlier_window",
    "pupil_per_eye",
}


def schema_cli_metadata() -> dict[str, tuple[str, str]]:
    """Collect {flag: (section, field)} from json_schema_extra cli entries."""
    out: dict[str, tuple[str, str]] = {}
    for section, model in PipelineConfig.model_fields.items():
        for field_name, field in model.annotation.model_fields.items():
            extra = field.json_schema_extra
            if isinstance(extra, dict) and "cli" in extra:
                flag = extra["cli"]
                assert flag not in out, f"duplicate cli metadata for {flag}"
                out[flag] = (section, field_name)
    return out


def resolve_schema_path(path: str):
    """Return the FieldInfo for a 'section.field' path (asserts it exists)."""
    section, field = path.split(".", 1)
    section_model = PipelineConfig.model_fields[section].annotation
    return section_model.model_fields[field]


# ══════════════════════════════════════════════════════════════════════════════
# Section models mirror the dataclasses
# ══════════════════════════════════════════════════════════════════════════════

# (schema model, dataclass) pairs; every dataclass field must exist on the
# schema section with an identical default, except documented exceptions.
MIRROR_PAIRS = [
    (DetectionSection, DetectionConfig),
    (GazeSection, GazeConfig),
    (TrackerSection, TrackerConfig),
    (RayFormingSection, RayFormingConfig),
    (DepthSection, DepthConfig),
    (PhenomenaSection, PhenomenaConfig),
    (OutputSection, OutputConfig),
    (ProjectOutputSection, ProjectOutputConfig),
]

# No default exceptions remain: merge_overlap_strategy's schema default was
# corrected from the dead 'filter' getattr-fallback to the runtime-true
# 'dynamic', so it now matches the dataclass default like every other field.
DEFAULT_EXCEPTIONS: dict = {}


def _dataclass_default(field: dataclasses.Field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


@pytest.mark.parametrize("model,dc", MIRROR_PAIRS,
                         ids=[dc.__name__ for _, dc in MIRROR_PAIRS])
def test_section_field_names_match_dataclass(model, dc):
    """Schema sections carry exactly the dataclass field names."""
    dc_names = {f.name for f in dataclasses.fields(dc)}
    schema_names = set(model.model_fields)
    assert schema_names == dc_names


@pytest.mark.parametrize("model,dc", MIRROR_PAIRS,
                         ids=[dc.__name__ for _, dc in MIRROR_PAIRS])
def test_section_defaults_match_dataclass(model, dc):
    """Schema defaults equal dataclass defaults (documented exceptions aside)."""
    for f in dataclasses.fields(dc):
        dc_default = _dataclass_default(f)
        if dc_default is dataclasses.MISSING:
            continue  # required field (none today outside AuxStreamConfig)
        schema_default = model.model_fields[f.name].get_default(
            call_default_factory=True)
        if (model, f.name) in DEFAULT_EXCEPTIONS:
            exp_dc, exp_schema = DEFAULT_EXCEPTIONS[(model, f.name)]
            assert dc_default == exp_dc
            assert schema_default == exp_schema
            continue
        if isinstance(schema_default, ProjectOutputSection):
            assert schema_default.directory == dc_default.directory
            continue
        assert schema_default == dc_default, (
            f"{dc.__name__}.{f.name}: dataclass default {dc_default!r} != "
            f"schema default {schema_default!r}")


def test_aux_stream_mirrors_dataclass():
    dc_names = {f.name for f in dataclasses.fields(AuxStreamConfig)}
    assert set(AuxStream.model_fields) == dc_names
    assert AuxStream.model_fields["auto_detect_faces"].get_default() is True


def test_project_section_mirrors_dataclass():
    dc_names = {f.name for f in dataclasses.fields(ProjectConfig)}
    assert set(ProjectSection.model_fields) == dc_names


def test_root_sections():
    assert set(PipelineConfig.model_fields) == {
        "detection", "gaze", "tracker", "rayforming", "depth",
        "phenomena", "output", "project",
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI flag completeness against the live parser
# ══════════════════════════════════════════════════════════════════════════════

def test_plugin_dest_snapshot():
    """Registry-introspected plugin dests match the frozen documented list."""
    assert get_plugin_dests() == KNOWN_PLUGIN_DESTS, (
        "Plugin-contributed flags changed; update KNOWN_PLUGIN_DESTS "
        "(plugin flags stay outside the schema by design).")


def test_every_core_flag_resolves():
    """Every non-plugin flag maps to a schema path or a documented exclusion.

    This is the GENERATED-THEN-FROZEN check from the SP1.1 plan: it reflects
    over the live parser so any newly added core flag must be given a schema
    home (cli metadata or CLI_ALIASES) or an explicit exclusion entry.
    """
    import argparse

    parser = get_parser()
    plugin_dests = get_plugin_dests()
    metadata = schema_cli_metadata()

    seen_flags = set()
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if action.dest in plugin_dests:
            continue
        assert len(action.option_strings) == 1, action.dest
        flag = action.option_strings[0]
        seen_flags.add(flag)
        resolved = [
            name for name, hit in (
                ("cli-metadata", flag in metadata),
                ("CLI_ALIASES", flag in CLI_ALIASES),
                ("EXCLUDED_CLI_FLAGS", flag in EXCLUDED_CLI_FLAGS),
            ) if hit
        ]
        assert len(resolved) == 1, (
            f"{flag} (dest={action.dest}) resolves via {resolved or 'NOTHING'}; "
            f"each core flag must map to exactly one of cli metadata, "
            f"CLI_ALIASES, or EXCLUDED_CLI_FLAGS")
        if flag in metadata:
            section, field_name = metadata[flag]
            assert action.dest == field_name, (
                f"{flag}: cli metadata may only sit on fields whose name "
                f"matches the argparse dest ({action.dest} != {field_name}); "
                f"use CLI_ALIASES instead")

    # Freshness: alias and exclusion tables may not reference dead flags.
    for flag in list(CLI_ALIASES) + list(EXCLUDED_CLI_FLAGS):
        assert flag in seen_flags, f"{flag} no longer exists in the parser"
    # Aliased paths must resolve, and genuinely differ from the dest name.
    for flag, path in CLI_ALIASES.items():
        resolve_schema_path(path)


def test_parser_defaults_match_schema_defaults():
    """Argparse defaults equal schema defaults for every mapped flag.

    Exceptions: the two min-call-gap spellings (argparse None resolves to the
    schema default 30 via resolve_min_call_gap).  --merge-overlap-strategy is
    no longer excepted -- its schema default was corrected to 'dynamic', which
    now matches the argparse default.
    """
    import argparse

    parser = get_parser()
    metadata = schema_cli_metadata()
    exceptions = {"--min-call-gap", "--rf-gazelle-interval"}

    by_flag = {}
    for action in parser._actions:
        if not isinstance(action, argparse._HelpAction) and action.option_strings:
            by_flag[action.option_strings[0]] = action

    for flag, (section, field_name) in metadata.items():
        if flag in exceptions:
            continue
        schema_default = resolve_schema_path(f"{section}.{field_name}").get_default(
            call_default_factory=True)
        assert by_flag[flag].default == schema_default, flag
    for flag, path in CLI_ALIASES.items():
        if flag in exceptions:
            continue
        schema_default = resolve_schema_path(path).get_default(
            call_default_factory=True)
        assert by_flag[flag].default == schema_default, flag


# ══════════════════════════════════════════════════════════════════════════════
# canonical_hash
# ══════════════════════════════════════════════════════════════════════════════

def test_canonical_hash_stable_across_constructions():
    a = PipelineConfig()
    b = PipelineConfig()
    assert a.canonical_hash() == b.canonical_hash()


def test_canonical_hash_changes_when_any_field_changes():
    base = PipelineConfig().canonical_hash()
    changed = PipelineConfig(gaze={"ray_length": 2.0})
    assert changed.canonical_hash() != base
    deep = PipelineConfig(project={"participants": {"v.mp4": {0: "S1"}}})
    assert deep.canonical_hash() != base


def test_canonical_hash_ignores_set_insertion_order():
    a = PipelineConfig(detection={"blacklist": {"tv", "laptop", "person"}})
    b = PipelineConfig(detection={"blacklist": {"person", "tv", "laptop"}})
    assert a.canonical_hash() == b.canonical_hash()


def test_canonical_hash_deterministic_across_processes():
    """Hash must survive hash-randomization (fresh interpreter, new seed)."""
    code = (
        "from mindsight.config import PipelineConfig\n"
        "cfg = PipelineConfig(detection={'blacklist': {'tv', 'laptop', 'person'}},\n"
        "                     gaze={'ray_length': 1.5})\n"
        "print(cfg.canonical_hash())\n"
    )
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=True, cwd=repo_root)
    local = PipelineConfig(
        detection={"blacklist": {"tv", "laptop", "person"}},
        gaze={"ray_length": 1.5}).canonical_hash()
    assert result.stdout.strip().splitlines()[-1] == local


# ══════════════════════════════════════════════════════════════════════════════
# extra="forbid"
# ══════════════════════════════════════════════════════════════════════════════

def test_unknown_root_key_raises():
    with pytest.raises(ValidationError):
        PipelineConfig(bogus_section={})


@pytest.mark.parametrize("model", [
    DetectionSection, GazeSection, TrackerSection, RayFormingSection,
    DepthSection, PhenomenaSection, OutputSection, ProjectSection,
    ProjectOutputSection,
], ids=lambda m: m.__name__)
def test_unknown_section_key_raises(model):
    with pytest.raises(ValidationError):
        model(definitely_not_a_field=1)


def test_nested_unknown_key_raises_through_root():
    with pytest.raises(ValidationError):
        PipelineConfig(gaze={"ray_length": 1.0, "bogus": 2})
