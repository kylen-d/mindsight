# Plugin System

MindSight's plugin framework lets developers extend gaze estimation, object detection, phenomena tracking, and data collection without modifying core code. All plugins are auto-discovered at import time from subdirectories under `Plugins/`.

---

## Overview

MindSight defines four plugin base classes, one per domain:

| Base class | Purpose | Registry |
|---|---|---|
| `GazePlugin` | Gaze estimation backends | `gaze_registry` |
| `ObjectDetectionPlugin` | Post-YOLO detection augmentation | `object_detection_registry` |
| `PhenomenaPlugin` | Gaze-phenomena trackers | `phenomena_registry` |
| `DataCollectionPlugin` | Custom data output | `data_collection_registry` |

All four base classes and their registries are defined in `Plugins/__init__.py`. The registries are populated automatically when the package is first imported.

---

## Directory Layout

```
Plugins/
├── GazeTracking/              # Gaze backend plugins
│   └── MyGaze/
│       ├── __init__.py
│       └── my_gaze.py         # exposes PLUGIN_CLASS
├── ObjectDetection/           # Detection augmentation plugins
│   └── MyDetector/
│       ├── __init__.py
│       └── my_detector.py
├── Phenomena/                 # Phenomena tracker plugins
│   └── NovelSalience/
│       ├── __init__.py
│       └── novel_salience.py
├── DataCollection/            # Custom data output plugins
│   └── MyExporter/
│       ├── __init__.py
│       └── my_exporter.py
└── TEMPLATE/                  # Skeleton plugin for developers
    ├── __init__.py
    └── my_plugin.py
```

Each plugin lives in its own named subfolder under the relevant type directory. The subfolder must contain at least one `.py` file (besides `__init__.py`) that exposes the `PLUGIN_CLASS` sentinel.

---

## PluginRegistry

`PluginRegistry` is the discovery and registration engine. There is one instance per plugin type.

### Construction

```python
registry = PluginRegistry()
```

Creates an empty registry backed by an internal `dict[str, type]`.

### Methods

| Method | Description |
|---|---|
| `discover(directory, namespace=None)` | Scan subdirectories for modules exposing `PLUGIN_CLASS` and register each one. |
| `register(cls)` | Register a class directly. Can also be used as a `@registry.register` decorator. |
| `get(name)` | Return the class registered under `name`. Raises `KeyError` if not found. |
| `names()` | Return a sorted list of all registered plugin names. |
| `name in registry` | Membership test (`__contains__`). |

### Discovery Process

When `discover(directory, namespace)` is called, the registry:

1. Iterates sorted subdirectories of `directory`.
2. Skips any directory whose name starts with `_`.
3. Within each subdirectory, globs for `*.py` files (sorted).
4. Skips any `.py` file whose name starts with `_` (including `__init__.py`).
5. Imports the module via `importlib.util.spec_from_file_location`.
6. Pre-registers the module in `sys.modules` so internal absolute imports resolve.
7. Checks for a `PLUGIN_CLASS` attribute on the loaded module.
8. If found, calls `register(mod.PLUGIN_CLASS)`.
9. If import fails, emits a `RuntimeWarning` and continues to the next file.

If `namespace` is omitted, it is derived as `"{parent_dir_name}.{directory_name}"`.

### Module-Level Registries

These are created and populated at import time in `Plugins/__init__.py`:

```python
gaze_registry              = PluginRegistry()
object_detection_registry  = PluginRegistry()
phenomena_registry         = PluginRegistry()
data_collection_registry   = PluginRegistry()
```

The gaze registry scans two locations:

- `Plugins/GazeTracking/` -- third-party gaze plugins.
- `ms/GazeTracking/Backends/` -- built-in gaze backends shipped with MindSight.

The other three registries each scan their corresponding `Plugins/<Type>/` directory.

---

## GazePlugin

Base class for gaze estimation backends.

### Class Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Required. Unique identifier for the backend. |
| `mode` | `str` | `"per_face"` (default) or `"scene"`. Controls which estimation method the pipeline calls. |
| `is_fallback` | `bool` | If `True`, this backend is tried last after all non-fallback backends. Default `False`. |

### Methods

**`estimate(face_bgr)`** -- Per-face estimation. Returns `(pitch_rad, yaw_rad, confidence)`. Implement this when `mode = "per_face"`.

**`estimate_frame(frame_bgr, face_bboxes_px)`** -- Scene-level estimation. Returns `[(gaze_xy_px, confidence), ...]`, one entry per bounding box. Implement this when `mode = "scene"`.

**`run_pipeline(**kwargs)`** -- Optional. Override to provide a self-contained pipeline that handles face cropping, estimation, temporal smoothing, and ray construction. When implemented, the coordinator in `ms/GazeTracking/gaze_pipeline.py` calls this instead of the default per-face or scene handler.

`run_pipeline` keyword arguments:

| kwarg | Description |
|---|---|
| `frame` | BGR numpy array at display resolution. |
| `faces` | List of detected face dicts (from RetinaFace). |
| `objects` | Non-person detection list. |
| `gaze_cfg` | `GazeConfig` with ray parameters. |
| `smoother` | Optional `GazeSmootherReID` instance. |
| `snap_hysteresis` | Optional `SnapHysteresisTracker` instance. |
| `aux_frames` | `dict[(pid_label, stream_type), ndarray | None]` -- per-participant auxiliary video frames. Empty dict when no auxiliary streams are configured. |

`run_pipeline` returns a tuple of `(persons_gaze, face_confs, face_bboxes, face_track_ids, face_objs, ray_snapped, ray_extended)`.

### Lifecycle

1. Registry discovers the plugin and calls `add_arguments` on startup.
2. `from_args` is called with the parsed CLI namespace. Return an initialized instance to activate, or `None` to skip.
3. The first plugin whose `from_args` returns non-`None` is used as the gaze backend for the entire run. Plugins with `is_fallback = True` are tried last.
4. Each frame, the coordinator calls `run_pipeline()` if implemented. Otherwise, `estimate` or `estimate_frame` is called depending on `mode`.

---

## ObjectDetectionPlugin

Post-YOLO detection augmentation. YOLO remains the default/fallback detector; plugins augment it.

### Methods

**`detect(*, frame, detection_frame, all_dets, det_cfg, **kwargs)`**

Called after the YOLO pass each frame. Parameters:

| Parameter | Description |
|---|---|
| `frame` | BGR numpy array at full display resolution. |
| `detection_frame` | Frame at detection scale (may be downscaled). |
| `all_dets` | Current detection list from YOLO (or a prior plugin). |
| `det_cfg` | `DetectionConfig` with confidence thresholds, class IDs, etc. |

Returns a `list[dict]` to replace the detection list, or `None` to keep it unchanged.

---

## PhenomenaPlugin

Custom phenomena trackers. This is the most common plugin type.

### Class Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Required. Unique identifier. |
| `dashboard_panel` | `str` | `"left"` or `"right"` (default `"right"`). Which dashboard side-panel to draw into. |

### Lifecycle Methods

**`update(**kwargs)`** -- Per-frame state update. Called once per frame before display. Returns a `dict` of plugin-specific live state (may be empty).

Common `update` kwargs:

| kwarg | Description |
|---|---|
| `frame_no` | Current frame index. |
| `persons_gaze` | List of `(origin, ray_end, angles)` per face. |
| `face_bboxes` | List of `(x1, y1, x2, y2)` in display pixels. |
| `hit_events` | `list[dict]` per-hit records (`face_idx` = stable track ID). |
| `joint_objs` | Set of joint-attention object indices. |
| `dets` | `list[dict]` non-person YOLO detections. |
| `n_faces` | Number of visible faces this frame. |
| `face_track_ids` | `list[int]` stable Re-ID track IDs (same order as `persons_gaze`). |
| `hits` | Set of `(face_list_idx, obj_list_idx)` pairs. |
| `aux_frames` | `dict[(pid_label, stream_type), ndarray | None]` per-participant auxiliary video frames. |
| `tip_convergences` | Tip convergence data (when available). |
| `detect_extend` | Extended detection metadata (when available). |

**`draw_frame(frame)`** -- Optional. Annotate the BGR video frame in-place. Called after `update`.

**`dashboard_data(*, pid_map=None)`** -- Return structured data for the matplotlib dashboard renderer.

Return format:

```python
{
    "title": "SECTION HEADING",
    "colour": (180, 180, 180),   # BGR accent colour
    "rows": [
        {"label": "P0 -> P1", "value": "12.3s", "pct": 0.45},
        {"label": "P1 -> P0", "value": "8.1s"},
    ],
    "empty_text": "--",          # shown when rows is empty
}
```

**`csv_rows(total_frames, *, pid_map=None)`** -- Return a `list` of rows to append to the summary CSV. Each row is a list of values.

**`console_summary(total_frames, *, pid_map=None)`** -- Return a multi-line string for post-run stdout output, or `None` to skip.

### Time-Series and Charting

**`time_series_data()`** -- Return accumulated time-series data for post-run chart generation.

```python
{
    "series_name": {
        "x": [0, 1, 2, ...],       # frame numbers
        "y": [0.1, 0.5, ...],      # metric values
        "label": "Human-readable",
        "chart_type": "line",       # "line", "area", or "step"
        "color": (255, 128, 0),     # optional BGR accent
    }
}
```

**`latest_metric()`** -- Return a `float` for the current frame's scalar metric (live charting in GUI mode), or `None`.

**`latest_metrics()`** -- Return per-series metric values for the live dashboard, or `None` to fall back to `latest_metric`.

```python
{
    "series_key": {
        "value": 3.14,
        "label": "P0 <-> P1",
        "y_label": "pairs",
    }
}
```

### Custom Qt Dashboard Widget

**`dashboard_widget()`** -- Return a custom `QWidget` for the live Qt dashboard, or `None` to use the standard rolling line-chart. Guard PyQt6 imports with `try/except ImportError` so CLI mode does not break.

**`dashboard_widget_update(data)`** -- Push new frame data to the custom widget. Only called when `dashboard_widget()` returned non-`None`.

---

## DataCollectionPlugin

Custom data output hooks.

### Methods

**`on_frame(**kwargs)`** -- Per-frame data collection hook. Called once per frame after all pipeline stages and display updates. Common kwargs: `frame_no`, `persons_gaze`, `face_bboxes`, `hit_events`, `face_track_ids`, `hits`, `objects`, `confirmed_objs`.

**`on_run_complete(**kwargs)`** -- Post-run hook. Called after the video loop ends. Common kwargs: `total_frames`, `joint_frames`, `confirmed_frames`, `total_hits`, `look_counts`, `source`, `all_trackers`.

**`generate_charts(output_dir, **kwargs)`** -- Generate custom post-run charts when `--charts` is enabled. Save files into `output_dir`. Return a list of created file paths. `kwargs` contains the same summary data as `on_run_complete`.

---

## CLI Protocol

All four plugin types share the same CLI integration pattern:

### `add_arguments(cls, parser)` (classmethod)

Register plugin-specific argparse flags. Called once at startup for every discovered plugin.

```python
@classmethod
def add_arguments(cls, parser):
    parser.add_argument("--my-plugin", action="store_true",
                        help="Enable the MyPlugin tracker.")
    parser.add_argument("--my-threshold", type=float, default=0.5)
```

### `from_args(cls, args)` (classmethod)

Inspect the parsed `argparse.Namespace` and return an initialized instance if the plugin should be activated, or `None` to skip.

```python
@classmethod
def from_args(cls, args):
    if getattr(args, "my_plugin", False):
        return cls(threshold=args.my_threshold)
    return None
```

### Integration in `MindSight.py`

On startup, `MindSight.py` iterates all four registries and calls `add_arguments` on every registered class. After argument parsing, it iterates again and calls `from_args`. For gaze plugins, the first non-`None` result wins (fallbacks tried last). For other types, all activated instances are collected and used.

---

## PLUGIN_CLASS Sentinel

Every plugin module must expose a module-level variable named `PLUGIN_CLASS` pointing to the plugin class:

```python
# my_plugin.py

from Plugins import PhenomenaPlugin

class NovelSalience(PhenomenaPlugin):
    name = "novel_salience"
    ...

PLUGIN_CLASS = NovelSalience
```

This is the sole discovery mechanism. Without `PLUGIN_CLASS`, the module is imported but silently ignored. The variable name is case-sensitive and must be exactly `PLUGIN_CLASS`.

Files whose names start with `_` are never scanned, so private helper modules (e.g., `_utils.py`) are safe to include alongside the plugin module.
