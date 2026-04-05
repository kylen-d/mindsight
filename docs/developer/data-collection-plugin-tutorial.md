# Data Collection Plugin Tutorial: JSON Event Logger

A full worked example building a realistic **JsonLogger** DataCollection plugin that writes per-frame gaze events to a structured JSON file. Use this alongside the [Writing a Plugin](writing-a-plugin.md) guide to see how each concept plays out in production code.

For tutorials on other plugin types, see: [Phenomena Plugin Tutorial](phenomena-plugin-tutorial.md) | [Gaze Plugin Tutorial](gaze-plugin-tutorial.md) | [Object Detection Plugin Tutorial](object-detection-plugin-tutorial.md)

---

## Overview

DataCollectionPlugins handle custom data output. They have per-frame (`on_frame`) and post-run (`on_run_complete`) hooks, plus optional chart generation via `generate_charts()`. This tutorial builds **JsonLogger** — a plugin that writes structured JSON event logs, complementing the built-in CSV output.

Source (hypothetical): `Plugins/DataCollection/JsonLogger/json_logger.py`

---

## What It Does

Collects per-frame gaze hit events into a structured JSON format with metadata (source video, total frames, participant labels) and writes the file on run completion. Useful for integration with external analysis tools that prefer JSON over CSV.

Each event records:

- Frame number
- Participant label (using the standard `resolve_display_pid` mapping)
- Object class name and detection confidence
- Bounding box coordinates
- Whether joint attention was active at the time

---

## File Structure

```
Plugins/DataCollection/JsonLogger/
├── __init__.py          # empty
└── json_logger.py       # PLUGIN_CLASS = JsonLoggerPlugin
```

The `__init__.py` is empty. All logic lives in `json_logger.py`.

---

## Class Definition

```python
import json
from pathlib import Path
from Plugins import DataCollectionPlugin
from ms.pipeline_config import resolve_display_pid

class JsonLoggerPlugin(DataCollectionPlugin):
    name = "json_logger"

    def __init__(self, output_path: str = "gaze_events.json"):
        self._path = output_path
        self._events = []
        self._metadata = {}
```

### Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `output_path` | `"gaze_events.json"` | File path for the JSON output. Parent directories are created automatically. |

The `_events` list accumulates per-frame data in memory. The `_metadata` dict is populated at run completion with summary information.

---

## The `on_frame()` Method

Called once per frame with the current pipeline state. This is where you collect data.

```python
def on_frame(self, **kwargs):
    frame_no = kwargs.get('frame_no', 0)
    hit_events = kwargs.get('hit_events', [])
    face_track_ids = kwargs.get('face_track_ids', [])
    confirmed_objs = kwargs.get('confirmed_objs', set())
    pid_map = kwargs.get('pid_map')

    for ev in hit_events:
        self._events.append({
            'frame': frame_no,
            'participant': resolve_display_pid(ev['face_idx'], pid_map),
            'object': ev['object'],
            'confidence': round(ev['object_conf'], 3),
            'bbox': list(ev['bbox']),
            'joint_attention': bool(confirmed_objs),
        })
```

### Step-by-step walkthrough

1. **Extract frame context.** All data arrives via `**kwargs`. Use `.get()` with sensible defaults so the plugin does not crash if a field is absent (forward compatibility).

2. **Iterate hit events.** Each `hit_event` represents a gaze-object intersection — a participant looking at a detected object in this frame. Not every frame produces hit events.

3. **Resolve participant labels.** `resolve_display_pid` maps internal face track IDs to stable display labels like `"P0"`, `"P1"`, etc. Always use this function for participant labels to stay consistent with the built-in CSV and dashboard outputs.

4. **Convert bbox to list.** Bounding boxes may arrive as tuples or numpy arrays. Wrapping in `list()` ensures JSON serializability.

5. **Record joint attention state.** `confirmed_objs` is a set of object names that two or more participants are looking at simultaneously. Converting to `bool` gives a simple "was JA happening?" flag per event.

### What `on_frame()` receives

| Keyword | Type | Description |
|---------|------|-------------|
| `frame_no` | `int` | Current frame number (0-indexed). |
| `hit_events` | `list[dict]` | Gaze-object intersections for this frame. |
| `face_track_ids` | `list[int]` | Active face track IDs. |
| `confirmed_objs` | `set[str]` | Objects under joint attention. |
| `pid_map` | `dict` or `None` | Face track ID to display label mapping. |

The `**kwargs` catch-all ensures forward compatibility as new fields are added.

---

## The `on_run_complete()` Method

Called once after the last frame has been processed. This is where you write output files.

```python
def on_run_complete(self, **kwargs):
    total_frames = kwargs.get('total_frames', 0)
    source = kwargs.get('source', '')

    output = {
        'metadata': {
            'source': str(source),
            'total_frames': total_frames,
            'total_events': len(self._events),
        },
        'events': self._events,
    }

    Path(self._path).parent.mkdir(parents=True, exist_ok=True)
    with open(self._path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"JSON log → {self._path}")
```

### Key details

- **Create parent directories.** `mkdir(parents=True, exist_ok=True)` ensures the output path works even if intermediate directories do not exist yet. This is important when the user specifies a path like `output/logs/events.json`.

- **Wrap events in metadata.** The top-level structure includes a `metadata` block with summary statistics. This makes the JSON file self-describing — a downstream tool can read the metadata without scanning all events.

- **Print a confirmation line.** A short message to stdout lets the user know where the file was written. Keep it brief — the pipeline already prints its own summary.

---

## Optional: `generate_charts()`

DataCollectionPlugins can optionally generate charts (matplotlib figures, images, etc.) that appear in the MindSight dashboard or are saved to the output directory.

```python
def generate_charts(self, output_dir, **kwargs):
    # Could generate custom matplotlib charts here.
    # Return a list of file paths created.
    return []
```

If your plugin does not produce charts, return an empty list. The method is optional but included here for completeness. A more advanced version might generate a timeline plot of gaze events per participant.

---

## CLI Activation

```python
@classmethod
def add_arguments(cls, parser):
    g = parser.add_argument_group("JSON Logger plugin")
    g.add_argument("--json-log", default=None, metavar="PATH",
                   help="Enable JSON event logging to PATH.")

@classmethod
def from_args(cls, args):
    path = getattr(args, "json_log", None)
    if not path:
        return None
    return cls(output_path=path)
```

### The path-as-activation pattern

Notice that `--json-log` takes a path argument rather than being a boolean flag. The presence of a path activates the plugin; omitting the flag disables it. This is a common pattern for DataCollection plugins because they always need an output location. Compare this with the boolean `--gaze-boost` flag used by ObjectDetection plugins that modify data in-place without producing files.

---

## Running It

```bash
python MindSight.py --source video.mp4 --json-log events.json --joint-attention
```

This runs the standard pipeline with JSON logging enabled. The `--joint-attention` flag activates joint attention tracking, which populates the `confirmed_objs` field that JsonLogger records.

### Example output

```json
{
  "metadata": {
    "source": "video.mp4",
    "total_frames": 3600,
    "total_events": 847
  },
  "events": [
    {
      "frame": 42,
      "participant": "P0",
      "object": "knife",
      "confidence": 0.87,
      "bbox": [120, 80, 200, 160],
      "joint_attention": false
    },
    {
      "frame": 42,
      "participant": "P1",
      "object": "cup",
      "confidence": 0.93,
      "bbox": [340, 200, 410, 290],
      "joint_attention": false
    },
    {
      "frame": 108,
      "participant": "P0",
      "object": "cup",
      "confidence": 0.91,
      "bbox": [338, 198, 412, 292],
      "joint_attention": true
    }
  ]
}
```

Note how the last event has `"joint_attention": true` — both P0 and P1 are looking at the cup.

---

## Key Design Patterns

### 1. Accumulate in `on_frame()`, write in `on_run_complete()`

```
on_frame()         → append to self._events (in memory)
on_run_complete()  → write self._events to disk
```

This is the standard DataCollectionPlugin pattern. Writing per-frame would be wasteful (repeated file opens, partial output on crash). Accumulating in memory and flushing once at the end is simpler and faster.

### 2. Use `resolve_display_pid` for participant labels

```python
resolve_display_pid(ev['face_idx'], pid_map)
```

This maps raw face track IDs to stable display labels (`"P0"`, `"P1"`, etc.) that match the built-in CSV output and dashboard. Never use raw track IDs in user-facing output — they can change across runs.

### 3. Create parent directories before writing

```python
Path(self._path).parent.mkdir(parents=True, exist_ok=True)
```

Always do this. Users will pass paths like `results/experiment1/events.json` and expect intermediate directories to be created.

### 4. Path argument activates the plugin

```python
@classmethod
def from_args(cls, args):
    path = getattr(args, "json_log", None)
    if not path:
        return None       # plugin not activated
    return cls(output_path=path)
```

When `from_args` returns `None`, the plugin is not registered. This is the standard opt-in mechanism for all plugin types.

### 5. `generate_charts()` is optional

Return an empty list if your plugin does not produce charts. The pipeline will not call it if it is not defined, but defining it explicitly communicates intent.

---

## Complete Code

```python
import json
from pathlib import Path
from Plugins import DataCollectionPlugin
from ms.pipeline_config import resolve_display_pid


class JsonLoggerPlugin(DataCollectionPlugin):
    """Write per-frame gaze events to a structured JSON file."""

    name = "json_logger"

    def __init__(self, output_path: str = "gaze_events.json"):
        self._path = output_path
        self._events = []
        self._metadata = {}

    # ── Per-frame collection ────────────────────────────────────────

    def on_frame(self, **kwargs):
        frame_no = kwargs.get('frame_no', 0)
        hit_events = kwargs.get('hit_events', [])
        confirmed_objs = kwargs.get('confirmed_objs', set())
        pid_map = kwargs.get('pid_map')

        for ev in hit_events:
            self._events.append({
                'frame': frame_no,
                'participant': resolve_display_pid(ev['face_idx'], pid_map),
                'object': ev['object'],
                'confidence': round(ev['object_conf'], 3),
                'bbox': list(ev['bbox']),
                'joint_attention': bool(confirmed_objs),
            })

    # ── Post-run output ─────────────────────────────────────────────

    def on_run_complete(self, **kwargs):
        total_frames = kwargs.get('total_frames', 0)
        source = kwargs.get('source', '')

        output = {
            'metadata': {
                'source': str(source),
                'total_frames': total_frames,
                'total_events': len(self._events),
            },
            'events': self._events,
        }

        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"JSON log → {self._path}")

    # ── Optional charts ─────────────────────────────────────────────

    def generate_charts(self, output_dir, **kwargs):
        return []

    # ── CLI integration ─────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser):
        g = parser.add_argument_group("JSON Logger plugin")
        g.add_argument("--json-log", default=None, metavar="PATH",
                       help="Enable JSON event logging to PATH.")

    @classmethod
    def from_args(cls, args):
        path = getattr(args, "json_log", None)
        if not path:
            return None
        return cls(output_path=path)


PLUGIN_CLASS = JsonLoggerPlugin
```
