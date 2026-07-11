"""
theming.py -- Fusion light/dark theming that follows the OS (eyes-on
2026-07-11, user request).

Three modes: ``auto`` (default -- follow the OS color scheme, live), ``light``
(stock Fusion), ``dark`` (a Fusion palette in MindSight's logo family: deep
indigo surfaces, plum highlight, green links).  The choice persists in the GUI
state and is switchable from View > Theme.

The existing hand-set widget styles (green go buttons, amber warnings, the
#1a1a2e preview panes, the dark chart panels) were chosen against dark
backgrounds already, so the dark palette makes them look at home rather than
fighting them.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette

THEME_MODES = ("auto", "light", "dark")

# Logo family: indigo surfaces, plum highlight, green links.
_C = {
    "window": "#1d1a2b",
    "text": "#e8e6f0",
    "base": "#151221",
    "alt_base": "#201c31",
    "button": "#262238",
    "highlight": "#a8447c",
    "link": "#6db384",
    "disabled": "#6a6680",
    "placeholder": "#77738c",
    "tooltip": "#262238",
}


def dark_palette() -> QPalette:
    """MindSight's dark Fusion palette (logo-aligned)."""
    p = QPalette()
    c = {k: QColor(v) for k, v in _C.items()}
    p.setColor(QPalette.ColorRole.Window, c["window"])
    p.setColor(QPalette.ColorRole.WindowText, c["text"])
    p.setColor(QPalette.ColorRole.Base, c["base"])
    p.setColor(QPalette.ColorRole.AlternateBase, c["alt_base"])
    p.setColor(QPalette.ColorRole.Text, c["text"])
    p.setColor(QPalette.ColorRole.PlaceholderText, c["placeholder"])
    p.setColor(QPalette.ColorRole.Button, c["button"])
    p.setColor(QPalette.ColorRole.ButtonText, c["text"])
    p.setColor(QPalette.ColorRole.BrightText, QColor("#ff6b6b"))
    p.setColor(QPalette.ColorRole.Highlight, c["highlight"])
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Link, c["link"])
    p.setColor(QPalette.ColorRole.ToolTipBase, c["tooltip"])
    p.setColor(QPalette.ColorRole.ToolTipText, c["text"])
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, c["disabled"])
    return p


def _os_prefers_dark(app) -> bool:
    try:
        from PyQt6.QtCore import Qt
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:  # pragma: no cover - Qt < 6.5
        return False


def apply_theme(app, mode: str = "auto"):
    """Apply *mode* ("auto" | "light" | "dark") to the running *app*.

    In auto mode the palette follows the CURRENT OS scheme; call this again
    from a colorSchemeChanged hookup to track live changes.
    """
    if mode not in THEME_MODES:
        mode = "auto"
    dark = mode == "dark" or (mode == "auto" and _os_prefers_dark(app))
    app.setPalette(dark_palette() if dark else QPalette())


def wire_auto_theme(app, current_mode) -> None:
    """Re-apply on OS scheme changes while the mode resolves to auto.

    *current_mode* is a zero-arg callable returning the live setting, so a
    later View-menu switch to an explicit mode wins without rewiring.
    """
    try:
        app.styleHints().colorSchemeChanged.connect(
            lambda _s: current_mode() == "auto" and apply_theme(app, "auto"))
    except Exception:  # pragma: no cover - Qt < 6.5
        pass
