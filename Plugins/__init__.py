"""
Plugins/__init__.py — Unified plugin framework for MindSight.

Directory layout
----------------
Each plugin lives in its own named subfolder under the relevant type directory:

Plugins/
├── Gaze/
│   └── MyGazePlugin/          one folder per plugin
│       ├── __init__.py
│       └── my_gaze_plugin.py  any *.py file exposing PLUGIN_CLASS
├── ObjectDetection/
│   └── MyDetector/
│       ├── __init__.py
│       └── my_detector.py
└── Phenomena/
    └── MyPhenomena/
        ├── __init__.py
        └── my_phenomena.py

Writing a plugin (all types)
-----------------------------
1. Create a subfolder under the appropriate type directory
   (e.g. ``Plugins/Phenomena/MyPlugin/``).
2. Add an ``__init__.py`` (may be empty) and a ``*.py`` module.
3. In the module, subclass the matching base: ``GazePlugin``,
   ``ObjectDetectionPlugin``, or ``PhenomenaPlugin``.
4. Set a unique, non-empty ``name`` class attribute.
5. Expose ``PLUGIN_CLASS = YourClass`` at module level — the registry
   discovers this sentinel automatically.
6. Implement the CLI protocol:
     add_arguments(cls, parser)  — add plugin-specific argparse flags.
     from_args(cls, args)        — return an initialized instance when
                                   activated, else ``None``.

See each base class below for the full per-type lifecycle.

Module-level registries (auto-populated on import)
---------------------------------------------------
gaze_registry              PluginRegistry for GazePlugin subclasses.
object_detection_registry  PluginRegistry for ObjectDetectionPlugin subclasses.
phenomena_registry         PluginRegistry for PhenomenaPlugin subclasses.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from abc import ABC
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Generic registry
# ══════════════════════════════════════════════════════════════════════════════

class PluginRegistry:
    """Discovers, registers, and vends plugin classes for one domain."""

    def __init__(self) -> None:
        self._plugins: dict[str, type] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, cls: type) -> type:
        """Register *cls*.  May also be used as a class decorator."""
        plugin_name = getattr(cls, "name", "")
        if not plugin_name:
            raise ValueError(
                f"{cls.__name__} must define a non-empty class attribute 'name'."
            )
        if plugin_name in self._plugins:
            warnings.warn(
                f"Plugin '{plugin_name}' already registered "
                f"({self._plugins[plugin_name].__name__}); overwriting with "
                f"{cls.__name__}.",
                RuntimeWarning, stacklevel=2,
            )
        self._plugins[plugin_name] = cls
        return cls

    # ── Discovery ────────────────────────────────────────────────────────────

    def discover(self, directory: Path, namespace: str | None = None) -> None:
        """
        Scan *directory* for named plugin subfolders and register any module
        that exposes a ``PLUGIN_CLASS`` attribute.

        Each plugin must live in its own subfolder (e.g.
        ``Plugins/Phenomena/MyPlugin/my_plugin.py``).  Folders whose names
        start with ``_`` are skipped; ``__init__.py`` files are ignored.

        Parameters
        ----------
        directory : Path
            Type-level plugin directory to scan
            (e.g. ``Plugins/Gaze/``, ``Plugins/Phenomena/``).
            No-ops silently if the path does not exist.
        namespace : str, optional
            Dotted ``sys.modules`` prefix for loaded modules
            (e.g. ``"Plugins.Gaze"``).  Derived from *directory* when omitted.
        """
        if not directory.is_dir():
            return
        if namespace is None:
            namespace = f"{directory.parent.name}.{directory.name}"

        for subdir in sorted(directory.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue
            sub_ns = f"{namespace}.{subdir.name}"
            for path in sorted(subdir.glob("*.py")):
                if path.name.startswith("_"):
                    continue
                module_name = path.stem
                full_name   = f"{sub_ns}.{module_name}"
                spec        = importlib.util.spec_from_file_location(full_name, path)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                # Pre-register in sys.modules so absolute imports inside the
                # plugin that reference this namespace resolve correctly.
                sys.modules[full_name] = mod
                try:
                    spec.loader.exec_module(mod)      # type: ignore[union-attr]
                    if hasattr(mod, "PLUGIN_CLASS"):
                        self.register(mod.PLUGIN_CLASS)
                except Exception as exc:
                    sys.modules.pop(full_name, None)
                    warnings.warn(
                        f"Could not load plugin '{module_name}' "
                        f"from {subdir}: {exc}",
                        RuntimeWarning, stacklevel=2,
                    )

    # ── Access ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> type:
        if name not in self._plugins:
            raise KeyError(
                f"No plugin '{name}'.  Available: {self.names()}"
            )
        return self._plugins[name]

    def names(self) -> list[str]:
        return sorted(self._plugins)

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    def __repr__(self) -> str:
        return f"PluginRegistry({self.names()})"


# ══════════════════════════════════════════════════════════════════════════════
# Domain base classes
# ══════════════════════════════════════════════════════════════════════════════

class GazePlugin(ABC):
    """
    Base class for gaze estimation backend plugins.

    Plugin lifecycle
    ----------------
    1. The registry discovers the plugin and calls ``add_arguments`` on startup.
    2. ``from_args`` is called with the parsed CLI namespace; return an
       initialized instance to activate the plugin, or ``None`` to skip it.
    3. The first plugin whose ``from_args`` returns a non-None instance is used
       as the gaze backend for the entire run; subsequent plugins are skipped.
       Plugins with ``is_fallback = True`` are tried last.
    4. Each frame, the coordinator calls ``run_pipeline()`` if implemented.
       Otherwise, ``estimate`` (per-face) or ``estimate_frame`` (scene) is
       called by the coordinator's default pipeline handler.

    Modes
    -----
    Set ``mode = "per_face"`` and implement ``estimate(face_bgr)``, which
    returns ``(pitch_rad, yaw_rad, confidence)``.

    Set ``mode = "scene"`` and implement ``estimate_frame(frame_bgr, bboxes)``,
    which returns ``[(gaze_xy_px, confidence), …]`` — one entry per bbox.

    Custom pipelines
    ----------------
    Override ``run_pipeline()`` to provide a self-contained pipeline that
    handles estimation, smoothing, and ray construction.  The coordinator
    in ``GazeTracking/gaze_pipeline.py`` will call it instead of the default
    per-face / scene handler.  See ``Plugins/Gaze/MGaze/`` for an example.
    """

    name: str = ""

    #: ``"per_face"`` or ``"scene"`` — controls which method the pipeline calls.
    mode: str = "per_face"

    #: If ``True``, this plugin is tried last after all non-fallback plugins.
    is_fallback: bool = False

    def estimate(self, face_bgr):
        """Per-face estimation.  Returns ``(pitch_rad, yaw_rad, confidence)``."""
        raise NotImplementedError(f"Plugin '{self.name}' has no estimate().")

    def estimate_frame(self, frame_bgr, face_bboxes_px: list) -> list:
        """Scene-level estimation.  Returns ``[(gaze_xy_px, confidence), …]``."""
        raise NotImplementedError(f"Plugin '{self.name}' has no estimate_frame().")

    def run_pipeline(self, **kwargs):
        """
        *Optional.*  Plugin-specific estimation pipeline.

        Override to provide a self-contained pipeline that handles face
        cropping, estimation, temporal smoothing, and ray construction.
        Each plugin pulls only the kwargs it needs.

        Common kwargs
        -------------
        frame           : BGR numpy array at display resolution.
        faces           : List of detected face dicts (from RetinaFace).
        objects         : Non-person detection list.
        gaze_cfg        : GazeConfig with ray parameters.
        smoother        : Optional GazeSmootherReID instance.
        snap_hysteresis : Optional SnapHysteresisTracker instance.
        aux_frames      : dict[(pid_label, stream_type), ndarray | None] —
                          per-participant auxiliary video frames.  Empty dict
                          when no auxiliary streams are configured.

        Returns
        -------
        tuple of (persons_gaze, face_confs, face_bboxes, face_track_ids,
                  face_objs, ray_snapped, ray_extended)
        """
        raise NotImplementedError

    @classmethod
    def add_arguments(cls, parser) -> None:
        """*Optional.*  Add plugin-specific flags to an argparse parser."""

    @classmethod
    def from_args(cls, args):
        """*Optional.*  Return an instance if activated by CLI args, else ``None``."""
        return None


class ObjectDetectionPlugin(ABC):
    """
    Base class for custom object detection plugins.

    Plugin lifecycle
    ----------------
    1. The registry discovers the plugin and calls ``add_arguments`` on startup.
    2. ``from_args`` is called with the parsed CLI namespace; return an
       initialized instance to activate the plugin, or ``None`` to skip it.
    3. Each frame, ``detect()`` is called after the default YOLO pass.
       The plugin receives the current detection list and may augment, filter,
       or replace it entirely by returning a new list.

    YOLO remains the default/fallback detector.  Plugins augment it.
    """

    name: str = ""

    def detect(self, *, frame, detection_frame, all_dets: list,
               det_cfg, **kwargs) -> list | None:
        """
        Post-process or replace the detection list for one frame.

        Parameters
        ----------
        frame           : BGR numpy array at full display resolution.
        detection_frame : Frame at detection scale (may be downscaled).
        all_dets        : Current detection list from YOLO (or prior plugin).
        det_cfg         : DetectionConfig with conf, class_ids, etc.

        Returns
        -------
        list[dict] — Updated detection list (returned to the pipeline).
        Return ``None`` to keep the current list unchanged.
        """
        return None

    @classmethod
    def add_arguments(cls, parser) -> None:
        """*Optional.*  Add plugin-specific flags to an argparse parser."""

    @classmethod
    def from_args(cls, args):
        """*Optional.*  Return an instance if activated by CLI args, else ``None``."""
        return None


class PhenomenaPlugin(ABC):
    """
    Base class for additional gaze-phenomena tracking plugins.

    Plugin lifecycle
    ----------------
    1. The registry discovers the plugin and calls ``add_arguments`` on startup.
    2. ``from_args`` is called with the parsed CLI namespace; return an
       initialized instance to activate the plugin, or ``None`` to skip it.
    3. Each video frame, ``update`` is called with the full gaze state.
    4. After ``update``, ``draw_frame`` is called so the plugin may annotate
       the video frame in-place.
    5. ``dashboard_section`` is called during ``compose_dashboard`` to append
       a panel section (see ``dashboard_panel`` to choose left or right).
    6. After the run, ``csv_rows`` is called and its rows are appended to the
       summary CSV (if ``--summary`` was requested).

    Minimal plugin
    --------------
    Override ``update`` (and optionally any of the display methods), set
    ``name``, implement ``add_arguments`` / ``from_args``, and expose
    ``PLUGIN_CLASS = YourClass`` at module level.

    Drawing helpers (for ``dashboard_section``)
    -------------------------------------------
    Import from ``DataCollection.dashboard_output``:
        _draw_panel_section(panel, y, title, title_col, rows, line_h) -> int
        _dash_line_h() -> int
        _DASH_HEAD, _DASH_DIM  (colour constants)
    """

    name: str = ""

    #: Which dashboard panel to draw into: ``"left"`` or ``"right"``.
    dashboard_panel: str = "right"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def update(self, **kwargs) -> dict:
        """
        Per-frame state update.  Called once per frame, before display.

        Accepts all frame data as keyword arguments.  Each plugin pulls only
        the keys it needs, making it safe to add new data without breaking
        existing plugins.

        Common kwargs
        -------------
        frame_no       : int — Current frame index.
        persons_gaze   : list of (origin, ray_end, angles) — one per face.
        face_bboxes    : list of (x1,y1,x2,y2) in display pixels.
        hit_events     : list[dict]  per-hit records (face_idx = stable track ID).
        joint_objs     : set of joint-attention object indices.
        dets           : list[dict]  non-person YOLO detections.
        n_faces        : int — Number of visible faces this frame.
        face_track_ids : list[int]  stable Re-ID track IDs (same order as
                         persons_gaze). Falls back to list-position indices when
                         the smoother is disabled.
        hits           : set of (face_list_idx, obj_list_idx) pairs — pre-computed
                         gaze-object intersections.
        aux_frames     : dict[(pid_label, stream_type), ndarray | None] —
                         per-participant auxiliary video frames (e.g. eye-tracking
                         cameras, first-person views).  Empty dict when no
                         auxiliary streams are configured.

        Returns
        -------
        dict  — plugin-specific live state (may be empty).
        """
        return {}

    def draw_frame(self, frame) -> None:
        """
        *Optional.*  Annotate the video frame in-place.

        Called after the built-in ``draw_overlay`` and after ``update`` so the
        plugin's latest state is available.  ``frame`` is a BGR numpy array.
        """

    def dashboard_section(self, panel, y: int, line_h: int, *,
                          pid_map=None) -> int:
        """
        *Optional.*  Draw one section into a dashboard side-panel.

        ``panel`` is a numpy view of the canvas for the relevant side panel
        (determined by ``self.dashboard_panel``).  Modify it in-place and
        return the new y coordinate after the section.

        *pid_map* maps internal track IDs to custom participant labels.

        .. deprecated:: 0.2.1
            Use :meth:`dashboard_data` instead.  The matplotlib dashboard
            calls ``dashboard_data()`` and renders it uniformly.
        """
        return y

    def dashboard_data(self, *, pid_map=None) -> dict:
        """Return structured data for the matplotlib dashboard renderer.

        Returns
        -------
        dict with keys:
            title : str — section heading (e.g. "MUTUAL GAZE")
            colour : tuple[int,int,int] — BGR accent colour
            rows : list[dict] — each dict has 'label' (str) and optionally
                   'value' (str|float) and 'pct' (float 0–1 for bar rendering)
            empty_text : str — placeholder when rows is empty (default "--")
        """
        return {'title': self.name.upper().replace('_', ' '),
                'colour': (180, 180, 180),
                'rows': [],
                'empty_text': '--'}

    def csv_rows(self, total_frames: int, *, pid_map=None) -> list:
        """
        *Optional.*  Return rows to append to the post-run summary CSV.

        Each row is a list of values (strings or numbers).  Include a blank
        row and a header row before data rows for readability.

        *pid_map* maps internal track IDs to custom participant labels.
        """
        return []

    def console_summary(self, total_frames: int, *, pid_map=None) -> str | None:
        """
        *Optional.*  Return a multi-line string for post-run stdout summary.

        Return ``None`` (the default) to skip.

        *pid_map* maps internal track IDs to custom participant labels.
        """
        return None

    # ── Time-series / charting protocol ──────────────────────────────────────

    def time_series_data(self) -> dict:
        """Return accumulated time-series data for post-run chart generation.

        Returns
        -------
        dict mapping series_name -> dict with keys:
            x     : list[int]   — frame numbers
            y     : list[float] — metric values
            label : str         — human-readable label for the series
            chart_type : str    — 'line', 'area', or 'step' (default 'line')
            color : tuple       — BGR accent colour (optional, falls back to
                                  the tracker's accent)
        Return an empty dict to skip chart generation for this tracker.
        """
        return {}

    def latest_metric(self) -> float | None:
        """Return the current-frame scalar metric value for live charting.

        Called each frame by the live dashboard bridge (GUI mode only).
        Return ``None`` to indicate no metric is available this frame.
        """
        return None

    def latest_metrics(self) -> dict | None:
        """Return per-series metric values for the live dashboard.

        Returns dict mapping series_key -> dict with keys:
            value   : float — current value for this series
            label   : str   — human-readable series label (e.g. "P0 ↔ P1")
            y_label : str   — y-axis unit description (e.g. "pairs", "frames", "%")

        Return ``None`` to fall back to :meth:`latest_metric` as a single
        aggregate scalar.
        """
        return None

    # ── Custom Qt dashboard widget (optional, GUI mode only) ─────────────────

    def dashboard_widget(self):
        """Return a custom QWidget for the live Qt dashboard, or ``None``.

        When ``None`` is returned (the default), the live dashboard uses a
        standard rolling line-chart widget driven by :meth:`latest_metric`.

        The returned widget must be safe to call from the Qt main thread.
        Guard PyQt6 imports with ``try/except ImportError`` so CLI mode
        does not break.
        """
        return None

    def dashboard_widget_update(self, data: dict) -> None:
        """Push new frame data to the custom dashboard widget.

        Only called if :meth:`dashboard_widget` returned a non-``None``
        widget.  *data* contains the same keys as the dashboard snapshot
        (frame_no, fps, n_faces, tracker_metrics, etc.).
        """

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """*Optional.*  Add plugin-specific flags to an argparse parser."""

    @classmethod
    def from_args(cls, args):
        """*Optional.*  Return an instance if activated by CLI args, else ``None``."""
        return None


class DataCollectionPlugin(ABC):
    """
    Base class for custom data output plugins.

    Plugin lifecycle
    ----------------
    1. The registry discovers the plugin and calls ``add_arguments`` on startup.
    2. ``from_args`` is called with the parsed CLI namespace; return an
       initialized instance to activate the plugin, or ``None`` to skip it.
    3. Each video frame, ``on_frame`` is called with the full pipeline context
       as keyword arguments.
    4. After the run completes, ``on_run_complete`` is called with summary data.

    Minimal plugin
    --------------
    Override ``on_frame`` and/or ``on_run_complete``, set ``name``,
    implement ``add_arguments`` / ``from_args``, and expose
    ``PLUGIN_CLASS = YourClass`` at module level.
    """

    name: str = ""

    def on_frame(self, **kwargs) -> None:
        """
        Per-frame data collection hook.  Called once per frame after all
        pipeline stages and display updates.

        Common kwargs: frame_no, persons_gaze, face_bboxes, hit_events,
        face_track_ids, hits, objects, confirmed_objs, etc.
        """

    def on_run_complete(self, **kwargs) -> None:
        """
        Post-run hook.  Called after the video loop ends with summary data.

        Common kwargs: total_frames, joint_frames, confirmed_frames,
        total_hits, look_counts, source, all_trackers, etc.
        """

    def generate_charts(self, output_dir: str, **kwargs) -> list[str]:
        """Generate custom post-run charts and return saved file paths.

        Called during ``finalize_run`` when ``--charts`` is enabled.
        *output_dir* is the directory where chart files should be saved.
        *kwargs* contains the same summary data as ``on_run_complete``.

        Return a list of file paths that were created (for logging).
        """
        return []

    @classmethod
    def add_arguments(cls, parser) -> None:
        """*Optional.*  Add plugin-specific flags to an argparse parser."""

    @classmethod
    def from_args(cls, args):
        """*Optional.*  Return an instance if activated by CLI args, else ``None``."""
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Module-level registries — auto-populated on first import
# ══════════════════════════════════════════════════════════════════════════════

_PLUGINS_ROOT = Path(__file__).parent

#: Registry for :class:`GazePlugin` backends (``Plugins/GazeTracking/``).
gaze_registry = PluginRegistry()
gaze_registry.discover(_PLUGINS_ROOT / "GazeTracking", namespace="Plugins.GazeTracking")

#: Built-in backends under ``GazeTracking/Backends/``.
gaze_registry.discover(
    _PLUGINS_ROOT.parent / "ms" / "GazeTracking" / "Backends",
    namespace="ms.GazeTracking.Backends",
)

#: Registry for :class:`ObjectDetectionPlugin` backends (``Plugins/ObjectDetection/``).
object_detection_registry = PluginRegistry()
object_detection_registry.discover(
    _PLUGINS_ROOT / "ObjectDetection", namespace="Plugins.ObjectDetection"
)

#: Registry for :class:`PhenomenaPlugin` backends (``Plugins/Phenomena/``).
phenomena_registry = PluginRegistry()
phenomena_registry.discover(_PLUGINS_ROOT / "Phenomena", namespace="Plugins.Phenomena")

#: Registry for :class:`DataCollectionPlugin` backends (``Plugins/DataCollection/``).
data_collection_registry = PluginRegistry()
data_collection_registry.discover(
    _PLUGINS_ROOT / "DataCollection", namespace="Plugins.DataCollection"
)
