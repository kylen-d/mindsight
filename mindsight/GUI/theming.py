"""
theming.py -- light/dark theme control via Qt's NATIVE color scheme.

Eyes-on 2026-07-11 (r2): the first pass shipped a custom logo-family palette;
the user preferred Fusion's own dark rendering, so this now just steers Qt's
built-in scheme (Qt 6.8+): ``auto`` follows the OS live (the default and the
pre-existing behavior), ``light``/``dark`` force a scheme.  No palettes are
constructed here -- Fusion derives both looks itself, which is exactly what
the user liked.

The choice persists in the GUI state and is switchable from View > Theme.
"""

from __future__ import annotations

THEME_MODES = ("auto", "light", "dark")


def apply_theme(app, mode: str = "auto"):
    """Steer the application's color scheme: auto | light | dark.

    No-op on Qt builds without ``QStyleHints.setColorScheme`` (< 6.8), where
    the app simply keeps following the OS.
    """
    from PyQt6.QtCore import Qt

    hints = app.styleHints()
    try:
        if mode == "dark":
            hints.setColorScheme(Qt.ColorScheme.Dark)
        elif mode == "light":
            hints.setColorScheme(Qt.ColorScheme.Light)
        else:
            hints.unsetColorScheme()
    except AttributeError:  # pragma: no cover - Qt < 6.8
        pass
