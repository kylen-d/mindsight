"""Capture the live MindSight argparse parser as a committed golden spec.

SP1.3 freezes the hand-written CLI parser BEFORE it is replaced by the
schema-generated one, so the generated parser can be proven byte-identical to
what shipped.  This script introspects the parser that ``ms.cli._args`` builds
(by intercepting the first ``parse_known_args`` -- the SUPPRESS pass in _args)
and writes:

- ``tests/data/cli_parser_golden.json``: prog + an ordered list of every action
  (excluding -h/--help): option strings, dest, action class, default, type name,
  nargs, const, choices, required, metavar, help, and owning argument-group title.
- ``tests/data/cli_help_golden.txt``: ``parser.format_help()`` at COLUMNS=100.

Run:  ~/claudeyolo/bin/python scripts/capture_cli_parser_spec.py
"""

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"


def capture_parser(argv=None):
    """Return the ArgumentParser that ms.cli._args constructs."""
    captured: dict = {}
    orig = argparse.ArgumentParser.parse_known_args

    def hook(self, *a, **k):
        captured.setdefault("parser", self)
        return orig(self, *a, **k)

    argparse.ArgumentParser.parse_known_args = hook
    try:
        from ms.cli import _args
        _args(argv or [])
    finally:
        argparse.ArgumentParser.parse_known_args = orig
    return captured["parser"]


def _group_titles(parser):
    """Map id(action) -> argument-group title."""
    titles: dict[int, str] = {}
    for group in parser._action_groups:
        for action in group._group_actions:
            titles[id(action)] = group.title
    return titles


def _json_safe(value):
    """Coerce argparse defaults to JSON-safe values (sets/tuples -> lists)."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def parser_spec(parser):
    """Serialisable spec of a parser: prog + ordered non-help actions."""
    titles = _group_titles(parser)
    actions = []
    for action in parser._actions:
        if action.dest == "help":
            continue
        default = _json_safe(action.default)
        # fail loudly on anything json can't represent
        json.dumps(default)
        actions.append({
            "option_strings": list(action.option_strings),
            "dest": action.dest,
            "action": type(action).__name__,
            "default": default,
            "type": action.type.__name__ if action.type is not None else None,
            "nargs": action.nargs,
            "const": _json_safe(action.const),
            "choices": list(action.choices) if action.choices is not None else None,
            "required": action.required,
            "metavar": action.metavar,
            "help": action.help,
            "group": titles.get(id(action)),
        })
    return {"prog": parser.prog, "actions": actions}


def format_help_pinned(parser, columns=100):
    """format_help() with a pinned terminal width for a stable golden."""
    prev = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = str(columns)
    try:
        return parser.format_help()
    finally:
        if prev is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = prev


def main():
    parser = capture_parser()
    spec = parser_spec(parser)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / "cli_parser_golden.json"
    json_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")

    help_path = DATA_DIR / "cli_help_golden.txt"
    help_path.write_text(format_help_pinned(parser))

    n = len(spec["actions"])
    print(f"prog: {spec['prog']!r}")
    print(f"actions (excl. -h): {n}")
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {help_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
