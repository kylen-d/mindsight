"""SP1.6: legacy ``import ms.*`` fails with a rename hint (meta-path finder)."""

import pytest


def test_import_ms_raises_hint():
    import mindsight  # noqa: F401  -- installs the finder

    with pytest.raises(ModuleNotFoundError, match="renamed from 'ms' to 'mindsight'"):
        import ms  # noqa: F401


def test_import_ms_submodule_raises_hint():
    import mindsight  # noqa: F401

    with pytest.raises(ModuleNotFoundError, match="mindsight"):
        import ms.pipeline_config  # noqa: F401
