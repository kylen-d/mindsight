# Plugin Base Classes

MindSight defines **four** plugin base classes in `Plugins/__init__.py`. Every
plugin subclasses exactly one of them, sets a unique `name`, and exposes a
module-level `PLUGIN_CLASS` sentinel so the registry can discover it.

```python
from Plugins import (
    GazePlugin,              # gaze estimation backends
    ObjectDetectionPlugin,   # detection post-processors
    PhenomenaPlugin,         # gaze-phenomena trackers
    DataCollectionPlugin,    # custom output / chart writers
)
```

!!! note "Signatures below are the real v1.0.0 contract"
    Every method signature, keyword name, and return shape on this page is
    copied from `Plugins/__init__.py`. The lifecycle hooks receive **keyword
    arguments** (`update(self, **kwargs)`, `on_frame(self, **kwargs)`, ...), so a
    plugin only pulls the keys it needs. Pulling a key by the wrong name fails
    silently, so match the names exactly.

---

## Discovery and registration

Each plugin lives in its own named subfolder under the matching type directory:

```
Plugins/
├── GazeTracking/    MyGaze/my_gaze.py         (exposes PLUGIN_CLASS)
├── ObjectDetection/ MyDetector/my_detector.py
├── Phenomena/       MyPhenomenon/my_phenom.py
└── DataCollection/  MyWriter/my_writer.py
```

On import, `Plugins/__init__.py` builds four module-level registries and calls
`discover()` on each type directory:

| Registry | Base class | Scanned directory |
|---|---|---|
| `gaze_registry` | `GazePlugin` | `Plugins/GazeTracking/` |
| `object_detection_registry` | `ObjectDetectionPlugin` | `Plugins/ObjectDetection/` |
| `phenomena_registry` | `PhenomenaPlugin` | `Plugins/Phenomena/` |
| `data_collection_registry` | `DataCollectionPlugin` | `Plugins/DataCollection/` |

`PluginRegistry.discover()` walks each subfolder, imports every `*.py` file that
does not start with `_`, and registers the module's `PLUGIN_CLASS` attribute.
Folders and files whose names start with `_` are skipped. A load failure is
recorded on `registry.load_errors` and emitted as a `RuntimeWarning` — it does
not abort discovery. `register()` raises `ValueError` if `name` is empty and
warns (then overwrites) on a duplicate `name`.

```python
from Plugins import gaze_registry, phenomena_registry
gaze_registry.names()          # ['gazelle', 'iris_refined'] — sorted plugin names
phenomena_registry.get("joint_attention")   # -> the class
"joint_attention" in phenomena_registry      # membership test
```

The CLI protocol is identical for all four bases:

```python
@classmethod
def add_arguments(cls, parser) -> None: ...   # add argparse flags (optional)

@classmethod
def from_args(cls, args):                     # return an instance, or None
    return None                               # None = plugin not activated
```

`from_args` is a **classmethod** (not static) and **may return `None`** — that
is how a plugin declines to activate for a given run. The factory
(`mindsight/factory.py`) calls `from_args` for every registered plugin and keeps
only the non-`None` instances.

---

## GazePlugin

Base class for gaze estimation backends. Selected by
`GazeTracking/gaze_factory.create_gaze_engine`; the first plugin whose
`from_args` returns non-`None` becomes the backend for the whole run (plugins
with `is_fallback = True` are tried last).

### Class attributes

| Attribute | Default | Meaning |
|---|---|---|
| `name` | `""` | Unique backend id. Must be non-empty. |
| `mode` | `"per_face"` | `"per_face"` calls `estimate`; `"scene"` calls `estimate_frame`. |
| `is_fallback` | `False` | If `True`, tried only after all non-fallback plugins. |

### Methods

```python
def estimate(self, face_bgr) -> tuple:
    """Per-face estimation. Returns (pitch_rad, yaw_rad, confidence)."""

def estimate_frame(self, frame_bgr, face_bboxes_px: list) -> list:
    """Scene estimation. Returns [(gaze_xy_px, confidence), ...], one per bbox."""

def run_pipeline(self, **kwargs):
    """Optional. Self-contained pipeline. Returns the 7-tuple
    (persons_gaze, face_confs, face_bboxes, face_track_ids,
     face_objs, ray_snapped, ray_extended)."""
```

Note the shapes: `estimate` returns a **3-tuple including confidence** (not
`(pitch, yaw)`), and `estimate_frame` returns **pixel gaze points with
confidence**, not angles. Common `run_pipeline` kwargs: `frame`, `faces`,
`objects`, `gaze_cfg`, `smoother` (a `GazeSmootherReID`), `snap_temporal` (a
`SnapTemporalState`), and `aux_frames`.

### Minimal runnable skeleton

```python
# Plugins/GazeTracking/ConstantGaze/constant_gaze.py
from Plugins import GazePlugin


class ConstantGaze(GazePlugin):
    name = "constant"
    mode = "per_face"

    def estimate(self, face_bgr):
        # (pitch_rad, yaw_rad, confidence) — always looks slightly down
        return (0.1, 0.0, 1.0)

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--constant-gaze", action="store_true",
                            help="Use the constant-gaze backend.")

    @classmethod
    def from_args(cls, args):
        return cls() if getattr(args, "constant_gaze", False) else None


PLUGIN_CLASS = ConstantGaze
```

The built-in **MobileGaze** backend (`--mgaze-*` flags) lives outside the plugin
tree at `mindsight/GazeTracking/Backends/MGaze/`; **Gaze-LLE** ships as a real
plugin at `Plugins/GazeTracking/Gazelle/` and overrides `run_pipeline`.

---

## ObjectDetectionPlugin

Runs **after** the default YOLO pass each frame. A plugin may augment, filter, or
replace the detection list by returning a new list, or return `None` to leave it
unchanged.

### Method

```python
def detect(self, *, frame, detection_frame, all_dets: list,
           det_cfg, **kwargs) -> list | None:
    """Post-process one frame's detections. Return an updated list, or
    None to keep all_dets unchanged."""
```

`detect` is **keyword-only** and takes `**kwargs`. The four named parameters are
always supplied; the pipeline passes additional context (e.g.
`prev_persons_gaze`, `prev_face_track_ids`) through `**kwargs`, so pull those
only if you need them.

| Parameter | Meaning |
|---|---|
| `frame` | BGR array at full display resolution. |
| `detection_frame` | Frame at detection scale (may be downscaled). |
| `all_dets` | Current detection list (YOLO output or prior plugin). |
| `det_cfg` | `DetectionConfig` — `conf`, `class_ids`, merge/blacklist settings. |

Each detection is a **dict** with keys `x1, y1, x2, y2, conf, cls, name`
(and `depth_median` when depth is enabled) — see
[the Detection dataclass reference](detection-dataclass.md). There is **no
`bbox` key**; read the corners directly.

### Minimal runnable skeleton

```python
# Plugins/ObjectDetection/ConfFloor/conf_floor.py
from Plugins import ObjectDetectionPlugin


class ConfFloor(ObjectDetectionPlugin):
    name = "conf_floor"

    def __init__(self, floor: float):
        self.floor = floor

    def detect(self, *, frame, detection_frame, all_dets, det_cfg, **kwargs):
        # Drop weak detections the default gate let through.
        return [d for d in all_dets if d["conf"] >= self.floor]

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--conf-floor", type=float, default=None,
                            help="Extra confidence floor for detections.")

    @classmethod
    def from_args(cls, args):
        floor = getattr(args, "conf_floor", None)
        return cls(floor) if floor is not None else None


PLUGIN_CLASS = ConfFloor
```

---

## PhenomenaPlugin

Base class for gaze-phenomena trackers. This is the richest base — most hooks are
optional and no-op by default.

### Class attributes

| Attribute | Default | Meaning |
|---|---|---|
| `name` | `""` | Unique registry id; keys flags and dashboards. Keep stable. |
| `summary_label` | `self.name` (property) | Prettier `phenomenon` column label; override with a class-level string. |
| `dashboard_panel` | `"right"` | Which side panel to draw into: `"left"` or `"right"`. |
| `live_chart_type` | `"line"` | Live-dashboard chart style: `"line"`, `"area"`, or `"step"`. |
| `preferred_video_types` | `[]` | Aux-stream `VideoType`s this plugin consumes (auto-routing). |
| `preferred_stream_labels` | `[]` | Aux-stream labels this plugin consumes. |

### Lifecycle hooks

```python
def update(self, **kwargs) -> dict:
    """Per-frame state update, before display. Return live state (may be {})."""

def finalize(self, frame_no: int, **kwargs) -> None:
    """Optional run-end hook. Close in-flight episodes before summaries are
    written. frame_no is one past the last processed frame."""

def draw_frame(self, frame) -> None:
    """Optional. Annotate the BGR frame IN-PLACE. Returns None."""
```

`update` receives frame data by keyword. The real key names (pre-1.0 vocab was
renamed) are:

| kwarg | Meaning |
|---|---|
| `frame_no` | Current frame index (int). |
| `persons_gaze` | list of `(origin, ray_end, angles)` — one per face. |
| `face_bboxes` | list of `(x1, y1, x2, y2)` in display pixels. |
| `face_track_ids` | list of stable Re-ID track IDs, same order as `persons_gaze`. |
| `hits` | set of `(face_list_idx, obj_list_idx)` pairs — gaze-object intersections. |
| `hit_events` | list of per-hit dicts (`face_idx` = stable track ID). |
| `joint_objs` | set of joint-attention object indices. |
| `dets` | list of non-person detection dicts. |
| `n_faces` | number of visible faces this frame. |
| `aux_frames` | `dict[(pid_label, stream_type), ndarray | None]` — aux video frames. |

`draw_frame` mutates the frame in place and **returns `None`** — do not return
the frame.

### Output hooks

```python
def summary_metrics(self, total_frames, fps, *, pid_map=None) -> list: ...
def summary_tables(self, total_frames, fps, *, pid_map=None) -> dict: ...
def episode_rows(self, total_frames, fps, *, pid_map=None) -> list: ...
def console_summary(self, total_frames, *, pid_map=None) -> str | None: ...
def csv_rows(self, total_frames, *, pid_map=None) -> list: ...   # legacy
```

- `summary_metrics` — preferred scalar hook. Returns a list of dicts with keys
  `phenomenon` (optional; defaults to `summary_label`), `participant`, `partner`,
  `object`, `metric` (snake_case with the unit encoded — `*_frames` /
  `*_seconds` / `*_pct`), and `value`. The writer emits one long-format row per
  dict into `{stem}_summary.csv`.
- `summary_tables` — tidy stream tables as `{table_name: (header, rows)}`. Each
  becomes `{stem}_{table_name}.csv` with `video_name`/`conditions` prepended.
- `episode_rows` — tidy episode records merged into `{stem}_phenomena_events.csv`
  (keys `phenomenon, participant, partner, object, frame_start, frame_end`). The
  base implementation reads `self._episodes` (an
  `mindsight.Phenomena.helpers.EpisodeLog`) and resolves integer track IDs
  through `resolve_display_pid`.
- `csv_rows` — **legacy** (deprecated since 1.0). A plugin overriding *only* this
  hook still writes `{stem}_plugin_{name}.csv` verbatim, so old third-party
  plugins keep working.

### Dashboard hooks

```python
def dashboard_data(self, *, pid_map=None) -> dict:
    """Structured data for the matplotlib dashboard. Keys:
    title, colour (BGR tuple), rows (list of {label, value?, pct?}),
    empty_text."""

def dashboard_section(self, panel, y: int, line_h: int, *, pid_map=None) -> int:
    """DEPRECATED since 0.2.1 — use dashboard_data instead."""

def time_series_data(self) -> dict: ...      # post-run charts
def latest_metric(self) -> float | None: ... # live single scalar
def latest_metrics(self) -> dict | None: ... # live per-series
def dashboard_widget(self): ...              # custom Qt widget or None
def dashboard_widget_update(self, data: dict) -> None: ...
```

!!! warning "`dashboard_section` is deprecated"
    `dashboard_section` has been deprecated since **0.2.1**. The matplotlib
    dashboard calls `dashboard_data()` and renders it uniformly. Implement
    `dashboard_data` for new plugins.

### Aux-stream helper

```python
def get_aux_frame(self, aux_frames: dict, pid: str, **overrides):
    """Best aux frame for pid using preferred_video_types then
    preferred_stream_labels, falling back to any stream for pid.
    video_type / stream_label overrides win over preferences."""
```

### Minimal runnable skeleton

```python
# Plugins/Phenomena/BlinkCount/blink_count.py
from Plugins import PhenomenaPlugin


class BlinkCount(PhenomenaPlugin):
    name = "blink_count"
    summary_label = "blink_count"

    def __init__(self):
        self._faces_seen = 0

    def update(self, **kwargs):
        self._faces_seen = max(self._faces_seen, kwargs.get("n_faces", 0))
        return {"faces": self._faces_seen}

    def summary_metrics(self, total_frames, fps, *, pid_map=None):
        return [{
            "participant": "", "partner": "", "object": "",
            "metric": "max_faces", "value": self._faces_seen,
        }]

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--blink-count", action="store_true",
                            help="Track the max simultaneous face count.")

    @classmethod
    def from_args(cls, args):
        return cls() if getattr(args, "blink_count", False) else None


PLUGIN_CLASS = BlinkCount
```

---

## DataCollectionPlugin

Base class for custom output / chart writers.

!!! warning "Only `generate_charts()` is wired in v1.0.0"
    The `on_frame` and `on_run_complete` hooks are documented for completeness
    but have **zero call sites** in v1.0.0, and `data_collection_registry` is not
    wired into the CLI parser (`add_arguments` flags never register). The one
    hook the pipeline actually invokes is `generate_charts()` (during
    `finalize_run` when `--charts` is enabled). See the
    [data-collection tutorial](../developer/data-collection-plugin-tutorial.md)
    for the working path and the wiring status.

### Methods

```python
def on_frame(self, **kwargs) -> None:
    """Per-frame hook. Spec, not wired in 1.0.0.
    Common kwargs: frame_no, persons_gaze, face_bboxes, hit_events,
    face_track_ids, hits, objects, confirmed_objs."""

def on_run_complete(self, **kwargs) -> None:
    """Post-run hook. Spec, not wired in 1.0.0.
    Common kwargs: total_frames, joint_frames, confirmed_frames,
    total_hits, look_counts, source, all_trackers."""

def generate_charts(self, output_dir: str, **kwargs) -> list[str]:
    """Working hook. Save charts under output_dir; return the file paths
    created. kwargs carry the run summary (total_frames, fps, all_trackers,
    pid_map)."""
```

### Minimal runnable skeleton

```python
# Plugins/DataCollection/FrameCountChart/frame_count_chart.py
import os
from Plugins import DataCollectionPlugin


class FrameCountChart(DataCollectionPlugin):
    name = "frame_count_chart"

    def generate_charts(self, output_dir, **kwargs):
        path = os.path.join(output_dir, "frame_count.txt")
        with open(path, "w") as fh:
            fh.write(f"frames={kwargs.get('total_frames', 0)}\n")
        return [path]

    @classmethod
    def from_args(cls, args):
        # Note: this registry is not wired into argparse in 1.0.0.
        return cls() if getattr(args, "frame_count_chart", False) else None


PLUGIN_CLASS = FrameCountChart
```
