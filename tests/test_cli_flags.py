"""SP1.3 step 2: prove the schema-generated parser reproduces the legacy one.

While the swap has not landed yet, this holds a three-way equivalence:
``ms.cli_flags.build_parser()`` == the frozen JSON/help golden == (the still
live) ``ms.cli._args`` parser (the latter via test_cli_parser_golden).  It also
welds the FlagSpec table to the schema/alias/exclusion tables and pins
``parse_cli`` behavior (defaults, nargs/const, the min_call_gap None, and the
explicit-flag set).
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_cli_parser_spec import format_help_pinned, parser_spec  # noqa: E402

from mindsight.cli_flags import (  # noqa: E402
    _FROM_SCHEMA,
    CORE_FLAGS,
    build_parser,
    parse_cli,
)


def _norm_action(a):
    a = dict(a)
    a["type"] = None if a["type"] == "str" else a["type"]
    return a


def _golden():
    return json.loads((DATA_DIR / "cli_parser_golden.json").read_text())


def test_generated_parser_spec_matches_golden():
    spec = parser_spec(build_parser())
    golden = _golden()
    assert spec["prog"] == golden["prog"]
    assert len(spec["actions"]) == len(golden["actions"]) == 150
    got = [_norm_action(a) for a in spec["actions"]]
    want = [_norm_action(a) for a in golden["actions"]]
    assert [a["dest"] for a in got] == [a["dest"] for a in want]
    for g, w in zip(got, want):
        assert g == w, f"generated action {w['dest']!r} diverged from golden"


def test_generated_help_matches_golden():
    got = format_help_pinned(build_parser())
    want = (DATA_DIR / "cli_help_golden.txt").read_text()
    assert got == want


def test_flagspec_table_welds_to_schema_and_tables():
    from mindsight.config import PipelineConfig
    from mindsight.config_compat import CLI_ALIASES, EXCLUDED_CLI_FLAGS

    assert len(CORE_FLAGS) == 107

    schema_cli: set[str] = set()

    def walk(model):
        for f in model.model_fields.values():
            extra = f.json_schema_extra or {}
            if isinstance(extra, dict) and "cli" in extra:
                schema_cli.add(extra["cli"])
            ann = f.annotation
            if hasattr(ann, "model_fields"):
                walk(ann)

    walk(PipelineConfig)

    schema_flags = {s.flag for s in CORE_FLAGS
                    if s.schema_path is not None and s.flag in schema_cli}
    alias_flags = {s.flag for s in CORE_FLAGS if s.flag in CLI_ALIASES}
    excluded_flags = {s.flag for s in CORE_FLAGS if s.schema_path is None}

    # every schema-backed FlagSpec is either a schema-cli flag or a known alias
    for s in CORE_FLAGS:
        if s.schema_path is not None:
            assert s.flag in schema_cli or s.flag in CLI_ALIASES, s.flag

    assert schema_flags == schema_cli and len(schema_flags) == 71
    assert alias_flags == set(CLI_ALIASES) and len(alias_flags) == 12
    assert excluded_flags == set(EXCLUDED_CLI_FLAGS) and len(excluded_flags) == 24


def test_only_two_default_overrides():
    overrides = {s.flag for s in CORE_FLAGS
                 if s.schema_path is not None and s.default is not _FROM_SCHEMA}
    assert overrides == {"--min-call-gap", "--rf-gazelle-interval"}
    for s in CORE_FLAGS:
        if s.flag in overrides:
            assert s.default is None


def test_parse_cli_explicit_flag_tracking():
    ns = parse_cli([])
    assert ns._explicit_cli == frozenset()
    ns = parse_cli(["--conf", "0.5"])
    assert ns.conf == 0.5
    assert ns._explicit_cli == frozenset({"conf"})


def test_parse_cli_nargs_optional_const():
    for flag, dest in [("--save", "save"), ("--summary", "summary"),
                       ("--heatmap", "heatmap"), ("--charts", "charts")]:
        assert getattr(parse_cli([]), dest) is None
        assert getattr(parse_cli([flag]), dest) is True
        assert getattr(parse_cli([flag, "x.out"]), dest) == "x.out"


def test_parse_cli_append_and_min_call_gap():
    ns = parse_cli(["--aux-stream", "a:eye_only:l:S1",
                    "--aux-stream", "b:eye_only:m:S2"])
    assert ns.aux_streams_raw == ["a:eye_only:l:S1", "b:eye_only:m:S2"]
    ns = parse_cli(["--rf-gazelle-interval", "10"])
    assert ns.rf_gazelle_interval == 10
    assert ns.min_call_gap is None


def test_parse_cli_rejects_bad_values():
    with pytest.raises(SystemExit):
        parse_cli(["--anonymize", "rainbow"])
    with pytest.raises(SystemExit):
        parse_cli(["--totally-unknown-flag"])
