# Data Collection Plugin Tutorial: JSON Summary Writer

A worked example building a **JsonSummary** DataCollection plugin that writes the
run's tracker metrics to a structured JSON file. Use it alongside the
[Writing a Plugin](writing-a-plugin.md) guide.

For tutorials on other plugin types, see: [Phenomena Plugin Tutorial](phenomena-plugin-tutorial.md) | [Gaze Plugin Tutorial](gaze-plugin-tutorial.md) | [Object Detection Plugin Tutorial](object-detection-plugin-tutorial.md)

---

!!! warning "Wiring status in v1.0.0 — read this first"
    The `DataCollectionPlugin` base declares three hooks, but only **one** is
    wired in v1.0.0:

    - **`generate_charts(output_dir, **kwargs)` is invoked** — during
      `finalize_run` when `--charts` is enabled
      (`mindsight/outputs/chart_output.py:188-199`). This tutorial builds against it.
    - **`on_frame()` and `on_run_complete()` have zero call sites.** They are
      documented for completeness as the intended per-frame / post-run contract,
      but the pipeline never calls them in 1.0.0.
    - **`data_collection_registry` is not wired into argparse.** `build_parser`
      only loops the gaze, object-detection, and phenomena registries
      (`mindsight/cli_flags.py:243-252`), so a DataCollection plugin's
      `add_arguments()` flags never register. A plugin can still activate by
      keying `from_args` off an **existing** flag (e.g. `--charts`) or
      unconditionally — see [CLI activation](#cli-activation).

    Net effect: a DataCollection plugin can contribute post-run chart/output
    files via `generate_charts`, but it cannot see per-frame state and cannot own
    its own CLI flag until the registry is wired in a future release.

---

## Overview

`DataCollectionPlugin` handles custom post-run output. In v1.0.0 the working
surface is `generate_charts()`, which receives the run's summary data and returns
the paths it wrote. This tutorial builds **JsonSummary** — a plugin that dumps
every tracker's scalar metrics to a JSON file next to the charts.

Source: `Plugins/DataCollection/JsonSummary/json_summary.py`

---

## File structure

```
Plugins/DataCollection/JsonSummary/
├── __init__.py          # empty
└── json_summary.py      # PLUGIN_CLASS = JsonSummaryPlugin
```

---

## Class definition

```python
import json
import os
from Plugins import DataCollectionPlugin


class JsonSummaryPlugin(DataCollectionPlugin):
    name = "json_summary"
```

DataCollection plugins are stateless in 1.0.0 — there is no per-frame hook to
accumulate into, so there is nothing to initialise. Everything happens in
`generate_charts`.

---

## The `generate_charts()` method (the working hook)

`generate_charts` is called once, after the run, when `--charts` is enabled. It
receives the output directory as its first positional argument and the run
summary as keyword arguments.

```python
def generate_charts(self, output_dir, **kwargs):
    total_frames = kwargs.get('total_frames', 0)
    fps = kwargs.get('fps', 0.0)
    all_trackers = kwargs.get('all_trackers', [])
    pid_map = kwargs.get('pid_map')

    metrics = []
    for tracker in all_trackers:
        if hasattr(tracker, 'summary_metrics'):
            metrics.extend(
                tracker.summary_metrics(total_frames, fps, pid_map=pid_map))

    path = os.path.join(output_dir, "run_summary.json")
    with open(path, 'w') as fh:
        json.dump({
            'total_frames': total_frames,
            'fps': fps,
            'metrics': metrics,
        }, fh, indent=2)
    return [path]
```

### What `generate_charts()` receives

The exact call is in `chart_output.generate_run_charts`:

```python
plugin.generate_charts(str(chart_dir),
                       total_frames=total_frames,
                       fps=fps,
                       all_trackers=all_trackers,
                       pid_map=pid_map)
```

| Keyword | Type | Description |
|---------|------|-------------|
| `output_dir` (positional) | `str` | The resolved chart directory. |
| `total_frames` | `int` | Total frames processed. |
| `fps` | `float` | Source frame rate (for seconds conversions). |
| `all_trackers` | `list` | Every active phenomena tracker instance. |
| `pid_map` | `dict` or `None` | Face track ID to display-label mapping. |

Reach the phenomena metrics through each tracker's `summary_metrics()` hook (see
[Plugin Base Classes](../reference/plugin-base-classes.md)) — that is the same
tidy data the built-in `{stem}_summary.csv` is built from. Exceptions raised here
are caught and downgraded to a `RuntimeWarning`, so a broken chart plugin never
crashes the run.

### Return value

Return a **list of file paths** you created (for logging). Return `[]` if you
wrote nothing.

---

## CLI activation

Because `add_arguments()` is not wired for this registry (see the warning above),
the plugin cannot register its own flag in 1.0.0. Activate it by keying
`from_args` off an existing flag — `--charts` is the natural choice, since that is
the flag that gates `generate_charts` being called at all:

```python
@classmethod
def from_args(cls, args):
    # No dedicated flag yet (registry not wired into argparse). Activate
    # whenever charts are being generated.
    return cls() if getattr(args, "charts", None) else None
```

When `from_args` returns `None`, the plugin is not built. `build_data_plugins`
(`mindsight/factory.py:76-88`) instantiates whichever plugins activate and seeds
`ctx['data_plugins']`, which `finalize_run` consumes.

You may still define `add_arguments` for forward compatibility — it is harmless,
it simply has no effect until the registry is wired in a later release.

---

## Running it

```bash
python MindSight.py --source video.mp4 --charts charts/ --joint-attention
```

`--charts` both enables chart generation and (via the `from_args` above) activates
JsonSummary. The plugin writes `charts/run_summary.json` alongside the built-in
time-series charts.

### Example output

```json
{
  "total_frames": 3600,
  "fps": 30.0,
  "metrics": [
    {"phenomenon": "joint_attention", "participant": "all", "partner": "",
     "object": "", "metric": "frames_active", "value": 143},
    {"phenomenon": "gaze_following", "participant": "P0", "partner": "P1",
     "object": "", "metric": "event_count", "value": 6}
  ]
}
```

---

## The spec-only hooks (`on_frame` / `on_run_complete`)

The base class also declares a per-frame and a post-run hook. They are the
**intended** contract for stateful, per-frame collection, but **the pipeline does
not call them in v1.0.0** — do not build a plugin that depends on them yet.

```python
def on_frame(self, **kwargs) -> None:
    """SPEC ONLY — not called in 1.0.0.
    Intended per-frame hook. Common kwargs: frame_no, persons_gaze,
    face_bboxes, hit_events, face_track_ids, hits, objects, confirmed_objs."""

def on_run_complete(self, **kwargs) -> None:
    """SPEC ONLY — not called in 1.0.0.
    Intended post-run hook. Common kwargs: total_frames, joint_frames,
    confirmed_frames, total_hits, look_counts, source, all_trackers."""
```

If you need per-frame gaze events **today**, write a `PhenomenaPlugin` instead —
its `update(**kwargs)` hook is called every frame and its `summary_tables()` hook
writes a real per-event stream CSV. See the
[Phenomena Plugin Tutorial](phenomena-plugin-tutorial.md).

---

## Complete code

```python
import json
import os
from Plugins import DataCollectionPlugin


class JsonSummaryPlugin(DataCollectionPlugin):
    """Dump every tracker's scalar metrics to a JSON file (post-run)."""

    name = "json_summary"

    def generate_charts(self, output_dir, **kwargs):
        total_frames = kwargs.get('total_frames', 0)
        fps = kwargs.get('fps', 0.0)
        all_trackers = kwargs.get('all_trackers', [])
        pid_map = kwargs.get('pid_map')

        metrics = []
        for tracker in all_trackers:
            if hasattr(tracker, 'summary_metrics'):
                metrics.extend(
                    tracker.summary_metrics(total_frames, fps, pid_map=pid_map))

        path = os.path.join(output_dir, "run_summary.json")
        with open(path, 'w') as fh:
            json.dump({
                'total_frames': total_frames,
                'fps': fps,
                'metrics': metrics,
            }, fh, indent=2)
        return [path]

    @classmethod
    def from_args(cls, args):
        # No dedicated flag yet (registry not wired into argparse).
        return cls() if getattr(args, "charts", None) else None


PLUGIN_CLASS = JsonSummaryPlugin
```
