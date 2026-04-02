"""
GazeTracking/gaze_factory.py — Factory for gaze estimation backends.

Centralizes the plugin discovery and fallback chain so both
the CLI (MindSight.py) and GUI share one code path.

Resolution order
-----------------
1. Explicit *backend* name (e.g. ``"gazelle"``, ``"mgaze"``).
2. Each registered gaze plugin's ``from_args()`` (non-fallback first).
3. Fallback plugins (``is_fallback = True``, e.g. MGaze).
"""
from Plugins import gaze_registry


def create_gaze_engine(
    plugin_args=None,
    backend: str | None = None,
    **backend_kwargs,
):
    """
    Create and return a gaze estimation engine.

    Parameters
    ----------
    plugin_args    : Parsed ``argparse.Namespace`` forwarded to ``from_args()``.
    backend        : Explicit backend name (e.g. ``"gazelle"``, ``"mgaze"``).
                     ``None`` = auto-detect via plugin chain.
    **backend_kwargs : Extra keyword arguments forwarded to the backend
                       constructor when *backend* is given explicitly.

    Returns
    -------
    A gaze engine instance (GazePlugin subclass) with an ``estimate()`` or
    ``estimate_frame()`` method and optionally ``run_pipeline()``.
    """
    # 1. Explicit backend requested (e.g. by GUI)
    if backend is not None:
        pcls = gaze_registry.get(backend)
        return pcls(**backend_kwargs)

    # 2. Try each registered non-fallback plugin
    if plugin_args is not None:
        for pname in gaze_registry.names():
            pcls = gaze_registry.get(pname)
            if getattr(pcls, 'is_fallback', False):
                continue
            try:
                eng = pcls.from_args(plugin_args)
            except Exception as exc:
                raise RuntimeError(
                    f"Gaze plugin '{pname}' failed to initialize: {exc}"
                ) from exc
            if eng is not None:
                return eng

    # 3. Try fallback plugins (e.g. MGaze)
    if plugin_args is not None:
        for pname in gaze_registry.names():
            pcls = gaze_registry.get(pname)
            if not getattr(pcls, 'is_fallback', False):
                continue
            try:
                eng = pcls.from_args(plugin_args)
            except Exception as exc:
                raise RuntimeError(
                    f"Fallback gaze plugin '{pname}' failed to initialize: {exc}"
                ) from exc
            if eng is not None:
                return eng

    raise RuntimeError(
        "No gaze backend could be activated. "
        f"Available plugins: {gaze_registry.names()}"
    )
