"""mindsight -- MindSight core package."""

import sys as _sys
from importlib.abc import MetaPathFinder as _MetaPathFinder

__version__ = "1.3.2"


class _MsMigrationHintFinder(_MetaPathFinder):
    """Explain the ms -> mindsight rename when a legacy import fails.

    Appended to the END of ``sys.meta_path``: it is consulted only after
    every real finder has failed, i.e. exactly when ``import ms...`` would
    already raise -- it never shadows a genuinely installed ``ms`` package.
    Delete after one release cycle past v1.0.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ms" or fullname.startswith("ms."):
            raise ModuleNotFoundError(
                f"No module named {fullname!r}: the MindSight core package "
                "was renamed from 'ms' to 'mindsight' in v1.0 and the "
                "compatibility shims were removed. Update the import, "
                "e.g. 'from ms.pipeline_config import ...' -> "
                "'from mindsight.pipeline_config import ...'.",
                name=fullname,
            )
        return None


if not any(isinstance(_f, _MsMigrationHintFinder) for _f in _sys.meta_path):
    _sys.meta_path.append(_MsMigrationHintFinder())
