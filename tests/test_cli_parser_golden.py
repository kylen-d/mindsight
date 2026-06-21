"""SP1.3 golden gate: pin the live CLI parser's full spec + --help output.

Captured from the hand-written parser BEFORE the schema-generated one replaces
it (scripts/capture_cli_parser_spec.py).  Once the swap lands (step 3) this same
test exercises the generated parser through ``ms.cli._args`` and proves it
reproduces every flag, default, type, nargs/const, choice, group, and help line.

The census is also pinned here so drift in the schema/alias/exclusion tables --
which drive generation -- is caught: 150 actions = 107 core (71 schema-cli + 12
CLI_ALIASES + 24 excluded) + 43 plugin.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_cli_parser_spec import (  # noqa: E402
    capture_parser,
    format_help_pinned,
    parser_spec,
)

CORE_GROUPS = {"options", "Performance", "Depth Estimation",
               "Ray Forming (Gazelle blend)"}


def _norm_type(t):
    """str and None are argparse-equivalent (identity vs str() on a str); the
    generated parser emits None where the legacy parser wrote type=str."""
    return None if t == "str" else t


def _norm_action(a):
    a = dict(a)
    a["type"] = _norm_type(a["type"])
    return a


def _load_golden():
    return json.loads((DATA_DIR / "cli_parser_golden.json").read_text())


def _live_spec():
    return parser_spec(capture_parser())


def test_parser_prog_and_action_count():
    spec = _live_spec()
    golden = _load_golden()
    assert spec["prog"] == golden["prog"]
    assert len(spec["actions"]) == 150
    assert len(golden["actions"]) == 150


def test_parser_spec_matches_golden():
    spec = _live_spec()
    golden = _load_golden()
    live = [_norm_action(a) for a in spec["actions"]]
    want = [_norm_action(a) for a in golden["actions"]]
    # compare action-by-action for a readable failure
    assert [a["dest"] for a in live] == [a["dest"] for a in want]
    for got, exp in zip(live, want):
        assert got == exp, f"action {exp['dest']!r} diverged"


def test_help_output_matches_golden():
    parser = capture_parser()
    got = format_help_pinned(parser)
    want = (DATA_DIR / "cli_help_golden.txt").read_text()
    assert got == want


def test_census_partition_welds_to_tables():
    """The 107 core flags partition exactly into schema-cli / alias / excluded,
    and the plugin groups supply the remaining 43 -- the invariant generation
    relies on."""
    from mindsight.config import PipelineConfig
    from mindsight.config_compat import CLI_ALIASES, EXCLUDED_CLI_FLAGS

    spec = _live_spec()
    core = [a for a in spec["actions"] if a["group"] in CORE_GROUPS]
    plugin = [a for a in spec["actions"] if a["group"] not in CORE_GROUPS]
    assert len(core) == 107
    assert len(plugin) == 43

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

    assert len(schema_cli) == 71
    assert len(CLI_ALIASES) == 12
    assert len(EXCLUDED_CLI_FLAGS) == 24

    core_flags = {a["option_strings"][0] for a in core}
    union = schema_cli | set(CLI_ALIASES) | set(EXCLUDED_CLI_FLAGS)
    # disjoint and exactly covering the core flag set
    assert len(union) == 107
    assert core_flags == union
