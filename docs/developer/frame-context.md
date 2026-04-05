# FrameContext Reference

## What is FrameContext?

`FrameContext` is a mutable dict-like container created once per frame and passed through all four pipeline stages. Each stage reads the keys it needs and writes its results back into the context. This design decouples the stages from each other -- adding new data fields never requires changing function signatures upstream or downstream.

`FrameContext` is defined in `ms/pipeline_config.py`.

## API

```python
ctx = FrameContext(frame=img, frame_no=42)

ctx['objects'] = detected_list       # __setitem__
objs = ctx.get('objects', [])        # get with default
if 'hits' in ctx: ...                # __contains__
ctx.update({'key': val})             # bulk update
kwargs = ctx.as_kwargs()             # shallow copy for **kwargs unpacking
```

Internally, `FrameContext` uses `__slots__ = ('data',)` for memory efficiency. All key-value pairs are stored in `ctx.data`, a plain Python dict. The dict-like dunder methods (`__getitem__`, `__setitem__`, `__contains__`) delegate to this internal dict.

## Lifecycle

Each frame follows this lifecycle:

1. **Creation** -- In the `run()` loop, a new `FrameContext` is created:
   ```python
   ctx = FrameContext(frame=frame, frame_no=N, **run_ctx_base)
   ```

2. **Seeding** -- `run_ctx_base` injects run-level state that persists across frames: the smoother, locker, phenomena trackers, output paths, PID map, and other long-lived objects.

3. **Processing** -- `process_frame(ctx)` passes the context through the four pipeline stages in order. Each stage reads what it needs and writes its outputs.

4. **Consumption** -- After all stages complete, the display renderer and dashboard read the final state of the context to draw overlays, update charts, and flush buffered data.

5. **Disposal** -- The context is discarded at the end of the frame iteration. Run-level objects (smoother, locker, trackers) survive because they are referenced by `run_ctx_base`, not owned by the context.

## Key Registry

The table below lists all known keys, the type of value stored, which component writes the key, and a brief description.

| Key | Type | Written By | Description |
|-----|------|-----------|-------------|
| `frame` | `np.ndarray` | constructor | Current BGR frame |
| `frame_no` | `int` | constructor | Frame counter (0-based) |
| `all_dets` | `list[Detection]` | detection_pipeline | All raw detections before filtering |
| `persons` | `list[Detection]` | detection_pipeline | Person-class detections only |
| `objects` | `list[Detection]` | detection_pipeline | Non-person detections (furniture, screens, etc.) |
| `face_bboxes` | `list` | gaze_pipeline | Detected face bounding boxes |
| `face_confs` | `list[float]` | gaze_pipeline | Face detection confidence scores |
| `face_track_ids` | `list[int]` | gaze_pipeline | Re-ID track IDs assigned to each face |
| `persons_gaze` | `list[tuple]` | gaze_pipeline | `(origin_xy, tip_xy)` gaze ray per tracked face |
| `hits` | `list[set]` | gaze_pipeline | Set of object indices hit by each face's gaze ray |
| `hit_events` | `list[dict]` | gaze_pipeline | Per-hit dicts with `face_idx`, `object`, `bbox`, `conf` |
| `joint_objs` | `set[int]` | process_frame | Object indices under raw joint attention (2+ faces) |
| `confirmed_objs` | `set[int]` | phenomena_pipeline | Temporally confirmed joint attention objects |
| `tip_convergences` | `list` | process_frame | Gaze tip convergence clusters |
| `smoother` | `GazeSmootherReID` | run_ctx_base | Gaze smoothing with re-identification |
| `locker` | `GazeLockTracker` | run_ctx_base | Fixation lock-on / dwell tracker |
| `all_trackers` | `list[PhenomenaPlugin]` | run_ctx_base | All active phenomena tracker instances |
| `look_counts` | `dict` | run_ctx_base | Per-(participant, object) cumulative frame counts |
| `heatmap_gaze` | `dict` | run_ctx_base | Per-participant gaze point accumulation for heatmaps |
| `pid_map` | `dict` | run_ctx_base | Track ID to participant label mapping |
| `aux_frames` | `dict` | run loop | Auxiliary stream frames keyed by `(pid, type)` |
| `fps` | `float` | run loop | Current rolling FPS estimate |
| `anonymize` | `str` or `None` | run_ctx_base | Face anonymization mode (`"blur"`, `"box"`, or `None`) |

## Extending FrameContext

Plugins can write any new keys into the context. To avoid collisions, follow this naming convention:

- Prefix your keys with your plugin name or a short unique identifier.
- Example: a salience-mapping plugin might write `salience_map`, `salience_peaks`, `salience_threshold`.

Always read keys defensively using `.get()` with a default value:

```python
salience = ctx.get('salience_map', None)
if salience is not None:
    # use it
```

This ensures your code works gracefully when the producing plugin is not loaded or has not yet run for the current frame.
