"""Tests for mindsight/config.py -- the unified pydantic schema (SP1.1).

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
    """Dests contributed by registry plugins, plus core-backend flags that live
    outside both the schema and the registries (MGaze since SP1.6)."""
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

    # Core gaze backend (MGaze) -- resolved by gaze_factory, not the registry.
    from mindsight.GazeTracking.Backends.MGaze.MGaze_Tracking import MGazePlugin
    fake = FakeArgumentParser()
    MGazePlugin.add_arguments(fake)
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
    # core backend (MGaze)
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


# ══════════════════════════════════════════════════════════════════════════════
# UI metadata (SP3.1 Batch F, D6)
# ══════════════════════════════════════════════════════════════════════════════

def _iter_ui():
    """Yield (path, field, ui) for every schema field; ui is dict or None."""
    for section, section_field in PipelineConfig.model_fields.items():
        for fname, field in section_field.annotation.model_fields.items():
            extra = field.json_schema_extra
            assert isinstance(extra, dict) and "ui" in extra, (
                f"{section}.{fname} has no ui metadata")
            yield f"{section}.{fname}", field, extra["ui"]


def test_ui_metadata_completeness():
    """Every schema field carries a ui entry (dict or explicit None)."""
    for path, _field, ui in _iter_ui():
        assert ui is None or isinstance(ui, dict), (
            f"{path}: ui must be a dict or None, got {type(ui)}")


def test_ui_metadata_does_not_move_canonical_hash():
    """Attaching ui metadata is inert to canonical_hash (hashes VALUES, not
    json_schema_extra).  Pins the default + a perturbed config.

    Re-pinned 2026-07-16 (v1.1 W2.2): adding the tracker.mgaze_reuse_eps
    schema FIELD legitimately moves the hash (it hashes values, and there is
    one more value).  Consequence: resume ledgers written before v1.1 report
    a config-hash mismatch and reprocess -- expected, noted in the changelog.
    Re-pinned again 2026-07-16 (v1.1 W3X): rayforming.rf_reuse_eps /
    rf_onset_samples / rf_onset_gap fields added (same one-time-reprocess
    consequence, still pre-release on v1.1-dev).
    """
    assert PipelineConfig().canonical_hash() == (
        "3c3b9a2bc676572770c409453196d862beceb1b96b191271faccbf78042fc351")
    assert PipelineConfig(gaze={"ray_length": 1.5}).canonical_hash() == (
        "10cc159eaf3fbeddd023c4464b370d15374bb74bb0d83471d69deb7951b417fb")


def test_ui_mirror_rule_targets_are_hidden():
    """Every PATH_MIRRORS target is ui:None -- the canonical owner renders,
    the mirror never appears twice on the generated surface (D6(a))."""
    from mindsight.config_compat import PATH_MIRRORS
    ui_by_path = {p: ui for p, _f, ui in _iter_ui()}
    for _canonical, mirrors in PATH_MIRRORS.items():
        for mirror in mirrors:
            assert ui_by_path[mirror] is None, (
                f"PATH_MIRRORS target {mirror} must be ui:None "
                f"(got {ui_by_path[mirror]!r})")


def test_ui_dict_fields_resolve_to_a_flag():
    """Every ui:dict field's cli/alias flag exists (D6 completeness)."""
    alias_paths = set(CLI_ALIASES.values())
    for path, field, ui in _iter_ui():
        if ui is None:
            continue
        extra = field.json_schema_extra
        has_cli = "cli" in extra
        has_alias = path in alias_paths
        assert has_cli or has_alias, (
            f"{path} is ui:dict but has no CLI flag (cli metadata or "
            f"CLI_ALIASES entry)")


def test_ui_toggle_group_integrity():
    """Every toggle_group named in the schema has exactly one owner field
    carrying an off_value (T10 / D6(d))."""
    owners: dict[str, list[str]] = {}
    for path, _field, ui in _iter_ui():
        if ui is None:
            continue
        tg = ui.get("toggle_group")
        if tg is None:
            continue
        assert "off_value" in ui, (
            f"{path} owns toggle_group {tg!r} but has no off_value")
        owners.setdefault(tg, []).append(path)
    for tg, members in owners.items():
        assert len(members) == 1, (
            f"toggle_group {tg!r} must have exactly one owner, got {members}")


def test_ui_advanced_tier_census():
    """The set of advanced-tier fields matches the committed golden -- tier
    changes are deliberate, reviewed diffs (D6 advanced-tier census)."""
    import json
    from pathlib import Path
    golden_path = (Path(__file__).resolve().parents[1] / "tests" / "data"
                   / "ui_advanced_tier_golden.json")
    golden = json.loads(golden_path.read_text())
    advanced = sorted(
        path for path, _f, ui in _iter_ui()
        if ui is not None and ui.get("advanced"))
    assert advanced == golden, (
        "advanced-tier census diverged from ui_advanced_tier_golden.json\n"
        f"added:   {sorted(set(advanced) - set(golden))}\n"
        f"removed: {sorted(set(golden) - set(advanced))}")
