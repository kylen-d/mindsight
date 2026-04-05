"""
GazeTracking/Backends/UniGaze/ -- UniGaze per-face gaze estimation backend.

This backend is OPTIONAL.  It requires the ``unigaze`` and ``timm`` packages
which are NOT bundled with MindSight (non-commercial license).  Install them
separately::

    pip install unigaze timm==0.3.2

If the packages are not installed, this plugin silently does not register.
"""
try:
    from .UniGaze_Tracking import UniGazePlugin  # noqa: F401
    PLUGIN_CLASS = UniGazePlugin
except ImportError:
    # unigaze or timm not installed -- silently skip registration
    pass
