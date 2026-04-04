# Writing a Plugin

MindSight supports four plugin types, each extending a different stage of the processing pipeline:

| Type | Base Class | Purpose |
|------|-----------|---------|
| **Gaze** | `GazePlugin` | Custom gaze estimation backends |
| **Object Detection** | `ObjectDetectionPlugin` | Augment or replace YOLO detections |
| **Phenomena** | `PhenomenaPlugin` | Track gaze-based social phenomena |
| **Data Collection** | `DataCollectionPlugin` | Custom data output and charting |

All four types share the same discovery mechanism: the plugin registry scans subdirectories under `Plugins/`, finds any `*.py` module that exposes a `PLUGIN_CLASS` module-level variable, and registers the class. All four types also share the same CLI protocol (`add_arguments` + `from_args`) for activation via command-line flags.

---

## Common Steps (All Plugin Types)

### Step 1: Create Your Plugin Directory

Create a named subfolder under the appropriate type directory. Each folder needs an `__init__.py` (can be empty) and one or more `*.py` modules:

```
Plugins/GazeTracking/MyBackend/
    __init__.py
    my_backend.py

Plugins/ObjectDetection/MyDetector/
    __init__.py
    my_detector.py

Plugins/Phenomena/MyTracker/
    __init__.py
    my_tracker.py

Plugins/DataCollection/MyOutput/
    __init__.py
    my_output.py
```

The filename does not matter. Only the class and `PLUGIN_CLASS` variable are used by the registry. Folders whose names start with `_` are skipped during discovery.

### Step 2: Set Up Imports

Plugins live in subdirectories and are loaded dynamically, so Python's default import resolution will not find top-level packages like `Plugins` or `DataCollection` from that depth. Add this boilerplate at the top of your module:

```python
from __future__ import annotations

import sys
from pathlib import Path

# Walk up to the repository root so sibling imports resolve correctly.
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

The number of `.parent` calls depends on your plugin's depth relative to the repository root. For a standard plugin at `Plugins/<Type>/<Name>/module.py`, three levels is correct.

Then import the base class for your plugin type:

```python
from Plugins import GazePlugin           # for gaze plugins
from Plugins import ObjectDetectionPlugin # for object detection plugins
from Plugins import PhenomenaPlugin       # for phenomena plugins
from Plugins import DataCollectionPlugin  # for data collection plugins
```

### Step 3: Implement the CLI Protocol

Every plugin type uses the same two classmethods for CLI integration:

```python
@classmethod
def add_arguments(cls, parser) -> None:
    """Called once at startup. Add plugin-specific flags to argparse."""
    g = parser.add_argument_group("My Plugin")
    g.add_argument("--my-plugin", action="store_true",
                   help="Enable the my-plugin plugin.")
    g.add_argument("--my-param", type=float, default=0.5,
                   help="Example parameter (default: 0.5).")

@classmethod
def from_args(cls, args):
    """Called after argument parsing. Return an instance if activated, else None."""
    if not getattr(args, "my_plugin", False):
        return None
    return cls(param=getattr(args, "my_param", 0.5))
```

- **`add_arguments`** -- Create an argument group with `parser.add_argument_group()` to keep your flags visually grouped in `--help` output.
- **`from_args`** -- Return `None` if the activation flag was not set; the registry will skip your plugin. Use `getattr(args, ..., default)` rather than direct attribute access to be resilient against missing attributes.

### Step 4: Expose PLUGIN_CLASS

At the bottom of your module file:

```python
PLUGIN_CLASS = MyClass
```

This is the sentinel that `PluginRegistry.discover()` looks for when it scans plugin directories. If it is missing, your plugin will be silently skipped.

---

## Writing a Gaze Plugin

Gaze plugins provide custom gaze estimation backends. The first plugin whose `from_args` returns a non-`None` instance is used as the gaze backend for the entire run. Plugins with `is_fallback = True` are tried last.

### Class Skeleton

```python
class MyGaze(GazePlugin):
    """One-line description of your gaze backend."""

    name = "my_gaze"
    mode = "per_face"   # or "scene"
    is_fallback = False

    def __init__(self, model_path: str = "default"):
        self._model_path = model_path
        # Load your model here
```

| Attribute | Purpose |
|-----------|---------|
| `name` | Unique string identifier for the registry. |
| `mode` | `"per_face"` or `"scene"` -- controls which estimation method the pipeline calls. |
| `is_fallback` | When `True`, this plugin is tried only after all non-fallback plugins. |

### Per-Face Mode

Set `mode = "per_face"` and implement `estimate`:

```python
def estimate(self, face_bgr):
    """Estimate gaze from a cropped face image.

    Parameters
    ----------
    face_bgr : ndarray
        Cropped face region as a BGR numpy array.

    Returns
    -------
    tuple of (pitch_rad, yaw_rad, confidence)
        Pitch and yaw in radians, confidence in [0, 1].
    """
    pitch, yaw = self._model.predict(face_bgr)
    return (pitch, yaw, 0.9)
```

The pipeline crops each detected face and calls `estimate` once per face per frame.

### Scene-Level Mode

Set `mode = "scene"` and implement `estimate_frame`:

```python
def estimate_frame(self, frame_bgr, face_bboxes_px: list) -> list:
    """Estimate gaze for all faces in a full frame.

    Parameters
    ----------
    frame_bgr : ndarray
        Full frame as a BGR numpy array.
    face_bboxes_px : list
        List of face bounding boxes in pixel coordinates.

    Returns
    -------
    list of (gaze_xy_px, confidence)
        One entry per bounding box. gaze_xy_px is the predicted
        gaze target point in pixel coordinates.
    """
    results = []
    for bbox in face_bboxes_px:
        gaze_point = self._model.predict_scene(frame_bgr, bbox)
        results.append((gaze_point, 0.85))
    return results
```

### Custom Pipeline (Advanced)

Override `run_pipeline()` for full control over face cropping, estimation, temporal smoothing, and ray construction. When implemented, the coordinator in `GazeTracking/gaze_pipeline.py` calls this instead of the default per-face or scene handler.

```python
def run_pipeline(self, **kwargs):
    """Self-contained gaze estimation pipeline.

    Common kwargs
    -------------
    frame           : BGR numpy array at display resolution.
    faces           : List of detected face dicts (from RetinaFace).
    objects         : Non-person detection list.
    gaze_cfg        : GazeConfig with ray parameters.
    smoother        : Optional GazeSmootherReID instance.
    snap_hysteresis : Optional SnapHysteresisTracker instance.
    aux_frames      : dict[(pid_label, stream_type), ndarray | None] --
                      per-participant auxiliary video frames.

    Returns
    -------
    tuple of (persons_gaze, face_confs, face_bboxes, face_track_ids,
              face_objs, ray_snapped, ray_extended)
    """
    frame = kwargs['frame']
    faces = kwargs.get('faces', [])
    # ... full pipeline logic ...
    return (persons_gaze, face_confs, face_bboxes,
            face_track_ids, face_objs, ray_snapped, ray_extended)
```

### Selection Behavior

At startup the registry iterates all discovered gaze plugins and calls `from_args` on each. The first plugin that returns a non-`None` instance wins and is used for the entire run. Plugins with `is_fallback = True` are deferred to the end of the iteration order, so they only activate when no other gaze plugin was selected.

For a complete worked example, see [Gaze Plugin Tutorial](gaze-plugin-tutorial.md).

---

## Writing an Object Detection Plugin

Object detection plugins augment or replace the default YOLO detection pass. Multiple plugins can be active simultaneously and are chained: each receives the output of the previous plugin.

### Class Skeleton

```python
class MyDetector(ObjectDetectionPlugin):
    """One-line description of your detector."""

    name = "my_detector"

    def __init__(self, conf_threshold: float = 0.3):
        self._conf = conf_threshold
```

### The detect() Method

```python
def detect(self, *, frame, detection_frame, all_dets, det_cfg, **kwargs):
    """Post-process or replace the detection list for one frame.

    Called after YOLO each frame, BEFORE the person/object split
    and BEFORE gaze estimation. Multiple plugins chain: each
    receives the output of the previous plugin.

    Parameters
    ----------
    frame           : ndarray -- full resolution BGR frame.
    detection_frame : ndarray -- possibly downscaled frame fed to YOLO.
    all_dets        : list[Detection] -- current detections (persons + objects).
    det_cfg         : DetectionConfig -- current detection configuration.

    Returns
    -------
    list[Detection] to replace the detection list, or None to keep unchanged.
    """
    # Example: filter out low-confidence detections
    filtered = [d for d in all_dets if d.conf >= self._conf]
    return filtered
```

You can modify the list in-place and return `None`, or return a new list to replace it entirely.

### Parameter Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `ndarray` | Full resolution BGR frame |
| `detection_frame` | `ndarray` | Possibly downscaled frame fed to YOLO |
| `all_dets` | `list[Detection]` | Current detections (persons + objects) |
| `det_cfg` | `DetectionConfig` | Current detection config (conf thresholds, class IDs, etc.) |

For a complete worked example, see [Object Detection Plugin Tutorial](object-detection-plugin-tutorial.md).

---

## Writing a Phenomena Plugin

Phenomena plugins track gaze-based social phenomena (mutual gaze, joint attention, gaze following, etc.) and integrate with the dashboard, CSV output, and live charting.

### Class Skeleton

```python
class MyTracker(PhenomenaPlugin):
    """One-line description of your phenomena tracker."""

    name = "my_tracker"
    dashboard_panel = "right"   # "left" or "right"

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._events = []
```

| Attribute | Purpose |
|-----------|---------|
| `name` | Unique string identifier. Appears in CSV headers, dashboard titles, and the internal registry. |
| `dashboard_panel` | Which side panel to draw into: `"left"` or `"right"`. |

### The update() Method

Called once per video frame with all available pipeline data as keyword arguments. Pull only what you need.

```python
def update(self, **kwargs) -> dict:
    frame_no     = kwargs['frame_no']
    persons_gaze = kwargs.get('persons_gaze', [])
    hits         = kwargs.get('hits', set())
    # ... your tracking logic ...
    return {}
```

#### Full kwargs Reference

| Key | Type | Description |
|-----|------|-------------|
| `frame_no` | `int` | Current frame index |
| `persons_gaze` | `list[(origin, ray_end, angles)]` | Per-face gaze data |
| `face_bboxes` | `list[(x1, y1, x2, y2)]` | Face bounding boxes |
| `hit_events` | `list[dict]` | Per-hit records with `face_idx`, `object`, `object_conf`, `bbox` |
| `joint_objs` | `set[int]` | Object indices currently under joint attention |
| `dets` | `list[Detection]` | Non-person YOLO detections |
| `n_faces` | `int` | Number of visible faces this frame |
| `face_track_ids` | `list[int]` | Stable re-ID track IDs (one per face) |
| `hits` | `set[(face_idx, obj_idx)]` | Gaze-object intersection pairs |
| `aux_frames` | `dict` | Auxiliary video frames keyed by `(pid, stream_type)` |

The return value is a dict of plugin-specific live state. Other parts of the system (e.g. the GUI) may inspect it. Return `{}` if you have nothing to report.

### Dashboard Display

Implement `dashboard_data()` to provide structured data for the dashboard renderer:

```python
def dashboard_data(self, *, pid_map=None) -> dict:
    rows = []
    if self._events:
        rows.append({'label': 'Total events', 'value': str(len(self._events))})
        rows.append({'label': 'Last event', 'value': f'frame {self._events[-1]}'})
    return {
        'title':      'MY TRACKER',
        'colour':     (200, 200, 200),   # BGR tuple
        'rows':       rows,
        'empty_text': '--',
    }
```

| Key | Type | Description |
|-----|------|-------------|
| `title` | `str` | Section heading in the panel |
| `colour` | `tuple` | BGR colour for the title |
| `rows` | `list[dict]` | Each dict has `'label'` and optionally `'value'`, `'pct'` |
| `empty_text` | `str` | Placeholder when rows is empty |

### CSV Output

Implement `csv_rows()` to append rows to the post-run summary CSV:

```python
def csv_rows(self, total_frames: int, *, pid_map=None) -> list:
    if not self._events:
        return []
    return [
        [],                                     # blank separator line
        ["my_tracker_events"],                   # section header
        ["category", "frame_no", "value"],       # column header
        *[["event", e['frame'], e['val']] for e in self._events],
    ]
```

The pattern is: blank row, section-name row, column-header row, then data rows. Return an empty list if there is nothing to write.

### Optional Methods

These methods have default (no-op) implementations in `PhenomenaPlugin`. Override as needed:

| Method | Purpose |
|--------|---------|
| `draw_frame(frame)` | Annotate the video frame in-place (called after `update`). |
| `console_summary(total_frames, *, pid_map)` | Return a string for post-run stdout output. |
| `time_series_data()` | Return time-series data for post-run chart generation. |
| `latest_metric()` | Return the current-frame scalar metric value for live charting. |
| `latest_metrics()` | Return a dict of per-series metric values for the live dashboard. |
| `dashboard_widget()` | Return a custom QWidget for the GUI dashboard, or `None`. |

For a complete worked example, see [Phenomena Plugin Tutorial](phenomena-plugin-tutorial.md).

---

## Writing a Data Collection Plugin

Data collection plugins provide custom output hooks that run every frame and after the run completes. Use them for custom file formats, database writes, streaming output, or chart generation.

### Class Skeleton

```python
class MyOutput(DataCollectionPlugin):
    """One-line description of your data output."""

    name = "my_output"

    def __init__(self, output_path: str = "output.json"):
        self._path = output_path
        self._records = []
```

### Per-Frame Hook

```python
def on_frame(self, **kwargs) -> None:
    """Called once per frame after all pipeline stages and display updates.

    Common kwargs: frame_no, persons_gaze, face_bboxes, hit_events,
    face_track_ids, hits, objects, confirmed_objs, pid_map
    """
    frame_no = kwargs['frame_no']
    hits = kwargs.get('hits', set())
    self._records.append({'frame': frame_no, 'n_hits': len(hits)})
```

### Post-Run Hook

```python
def on_run_complete(self, **kwargs) -> None:
    """Called after the video loop ends with summary data.

    Common kwargs: total_frames, total_hits, look_counts,
    source, all_trackers
    """
    import json
    with open(self._path, 'w') as f:
        json.dump(self._records, f)
```

### Chart Generation (Optional)

```python
def generate_charts(self, output_dir: str, **kwargs) -> list[str]:
    """Generate custom post-run charts. Called when --charts is enabled.

    Parameters
    ----------
    output_dir : str
        Directory where chart files should be saved.
    **kwargs
        Same summary data as on_run_complete.

    Returns
    -------
    list[str] -- file paths of created chart images.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([r['frame'] for r in self._records],
            [r['n_hits'] for r in self._records])
    path = f"{output_dir}/my_output_chart.png"
    fig.savefig(path)
    plt.close(fig)
    return [path]
```

For a complete worked example, see [Data Collection Plugin Tutorial](data-collection-plugin-tutorial.md).

---

## Testing Your Plugin

Run MindSight with your activation flag:

```bash
python MindSight.py --source video.mp4 --my-plugin --my-param 0.7
```

Check that:

1. Your plugin prints any startup confirmation (if you added one in `from_args`).
2. No import errors appear at startup -- if your plugin fails to load, the registry emits a `RuntimeWarning` with the traceback.
3. Your output appears in the expected place (dashboard panel, CSV section, chart file, or custom output).
4. No errors appear when new kwargs are added by other pipeline components -- always use `kwargs.get()` with defaults for optional keys.
