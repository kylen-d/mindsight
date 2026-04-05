"""
weights.py — Centralized weight-file resolution for all MindSight backends.

Every backend resolves model paths through :func:`resolve_weight` so that
bare filenames (e.g. ``"yolov8n.pt"``) land in ``Weights/{backend}/``
while absolute or relative paths with directory components are respected
as-is.
"""
from pathlib import Path

from ms.constants import PROJECT_ROOT

WEIGHTS_ROOT = PROJECT_ROOT / "Weights"


def resolve_weight(backend: str, filename: str) -> Path:
    """Resolve a weight file path, preferring ``Weights/{backend}/``.

    Parameters
    ----------
    backend : str
        Subdirectory name under ``Weights/`` (e.g. ``"YOLO"``, ``"L2CS"``).
    filename : str
        Model filename or path.  Bare filenames are resolved against
        ``Weights/{backend}/``.  Paths with directory components (relative
        or absolute) are returned unchanged.

    Returns
    -------
    Path
        Resolved path to the weight file.
    """
    p = Path(filename)
    if p.is_absolute() or p.parent != Path("."):
        return p
    target_dir = WEIGHTS_ROOT / backend
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / p.name
