"""
pipeline_loader.py -- Backward-compatible shim (SP1.3).

The legacy YAML-into-namespace loader (``load_pipeline`` and its ``_YAML_MAP`` /
``_is_default`` machinery) was folded into ``ms.config_compat`` in SP1.3, where
it lives next to the schema loader ``load_yaml``.  This module re-exports the
public names so existing imports keep working; new code should import from
``ms.config_compat``.  Slated for removal in SP1.6 with the other legacy shims.
"""

from ms.config_compat import (  # noqa: F401
    _flatten,
    _is_default,
    load_pipeline,
)
