"""
inference_settings -- the Analyze Footage Inference Settings dialog (UP2 B).

``SETTINGS_SPEC`` (spec.py) is the layout contract transcribed from the
user-triaged spec doc; ``InferenceSettingsDialog`` (dialog.py) renders it over
the RunSettings store.  Import the dialog lazily where Qt is available.
"""
from __future__ import annotations

from .spec import SETTINGS_SPEC, all_dests, field_meta, iter_fields

__all__ = [
    "SETTINGS_SPEC",
    "all_dests",
    "field_meta",
    "iter_fields",
    "InferenceSettingsDialog",
]


def __getattr__(name):
    if name == "InferenceSettingsDialog":
        from .dialog import InferenceSettingsDialog
        return InferenceSettingsDialog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
