"""Bundled-resource resolution for wheel installs.

A source checkout resolves shipped resources (the weights manifest, config
presets, the docs tree) as plain files under ``PROJECT_ROOT``.  A wheel
install has no checkout, so ``scripts/sync_bundled_resources.py`` copies
those files into ``mindsight/_bundled/`` at build time and they ship as
package data.  The lookup order everywhere is: the explicit ``PROJECT_ROOT``
file first (user-editable, byte-identical for checkouts), the bundled copy
second.  In a checkout ``_bundled/`` does not exist and every lookup behaves
exactly as before.
"""

from pathlib import Path

_BUNDLED = Path(__file__).resolve().parent / "_bundled"


def bundled_path(rel: str) -> Path | None:
    """A bundled resource path, or None when not shipped (checkout runs)."""
    p = _BUNDLED / rel
    return p if p.exists() else None


def resource_path(rel: str) -> Path | None:
    """``PROJECT_ROOT/rel`` if present, else the bundled copy, else None."""
    from mindsight import constants

    p = constants.PROJECT_ROOT / rel
    if p.exists():
        return p
    return bundled_path(rel)
