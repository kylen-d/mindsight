"""
GUI/arg_introspector.py — Introspect plugin add_arguments() calls to discover CLI flags.

Provides a fake argparse parser that captures add_argument() metadata without
actually parsing anything.  The GUI uses this to dynamically build Qt widgets
for any installed plugin's CLI flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ArgSpec:
    """Metadata for a single CLI argument discovered from a plugin."""
    dest: str               # argparse dest (e.g. "ns_speed_thresh")
    flag: str               # original flag (e.g. "--ns-speed-thresh")
    type: type | None       # float, int, str, or None for store_true
    default: Any = None
    help: str = ""
    choices: list | None = None
    action: str = "store"   # "store", "store_true", "store_false"
    metavar: str | None = None
    nargs: str | None = None
    group_title: str = ""   # from add_argument_group title


class _FakeGroup:
    """Mimics argparse argument group — captures add_argument calls."""

    def __init__(self, title: str = "", specs: list | None = None):
        self._title = title
        self._specs = specs if specs is not None else []

    def add_argument(self, *args, **kwargs):
        # Find the flag name (longest --flag)
        flags = [a for a in args if a.startswith("--")]
        if not flags:
            return  # positional args — skip
        flag = max(flags, key=len)

        dest = kwargs.get("dest", flag.lstrip("-").replace("-", "_"))
        action = kwargs.get("action", "store")
        arg_type = kwargs.get("type")
        if action in ("store_true", "store_false"):
            arg_type = None

        self._specs.append(ArgSpec(
            dest=dest,
            flag=flag,
            type=arg_type,
            default=kwargs.get("default"),
            help=kwargs.get("help", ""),
            choices=kwargs.get("choices"),
            action=action,
            metavar=kwargs.get("metavar"),
            nargs=kwargs.get("nargs"),
            group_title=self._title,
        ))

    def add_mutually_exclusive_group(self, **kwargs):
        return _FakeGroup(title=self._title, specs=self._specs)


class FakeArgumentParser:
    """Mimics argparse.ArgumentParser to capture add_argument() metadata."""

    def __init__(self):
        self._specs: list[ArgSpec] = []
        self._root_group = _FakeGroup(title="", specs=self._specs)

    def add_argument(self, *args, **kwargs):
        self._root_group.add_argument(*args, **kwargs)

    def add_argument_group(self, title="", **kwargs):
        return _FakeGroup(title=title, specs=self._specs)

    def add_mutually_exclusive_group(self, **kwargs):
        return _FakeGroup(title="", specs=self._specs)

    @property
    def specs(self) -> list[ArgSpec]:
        return list(self._specs)


def introspect_plugin(plugin_cls) -> list[ArgSpec]:
    """Call plugin_cls.add_arguments(fake_parser) and return discovered ArgSpec list.

    Returns an empty list if the plugin raises an exception or has no arguments.
    """
    parser = FakeArgumentParser()
    try:
        plugin_cls.add_arguments(parser)
    except Exception:
        return []
    return parser.specs
