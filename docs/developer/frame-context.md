# FrameContext Reference

## What is FrameContext?

`FrameContext` is a mutable dict-like container created once per frame and passed through all four pipeline stages. Each stage reads the keys it needs and writes its results back into the context. This design decouples the stages from each other -- adding new data fields never requires changing function signatures upstream or downstream.

`FrameContext` is defined in `mindsight/pipeline_config.py`.

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

The **authoritative, exhaustive** key catalogue — every constructor, detection,
gaze, process-frame, phenomena, run-context-base, run-loop, and finalize key with
its exact type and producer/consumer — lives in
[FrameContext Keys](../reference/frame-context-keys.md). Consult that page rather
than duplicating it here.

The keys below are the ones you will touch most often when writing a plugin. Note
the shapes that changed in the restructure:

| Key | Type | Written By | Description |
|-----|------|-----------|-------------|
| `frame` | `np.ndarray` | constructor | Current BGR frame (mutated in-place by overlay) |
| `frame_no` | `int` | constructor | Frame counter (0-based) |
| `objects` | `list[Detection]` | detection_pipeline | Non-person detections |
| `persons` | `list[Detection]` | detection_pipeline | Person-class detections only |
| `persons_gaze` | `list[tuple]` | gaze_pipeline | Per-person gaze as `(origin, ray_end, angles)` tuples; `angles` may be `None` |
| `face_bboxes` | `list[tuple]` | gaze_pipeline | Face boxes `(x1, y1, x2, y2)` |
| `face_track_ids` | `list[int]` | gaze_pipeline | Stable Re-ID track ID per face |
| `hits` | `set[tuple]` | gaze_pipeline | Set of `(face_idx, target_idx)` pairs that intersect this frame |
| `hit_events` | `list[dict]` | gaze_pipeline | Structured per-hit records for CSV logging (`face_idx` = stable track ID) |
| `joint_objs` | `set[int]` | process_frame | Object indices under raw joint attention (2+ faces) |
| `confirmed_objs` | `set[int]` | phenomena_pipeline | Temporally confirmed joint-attention object indices |
| `smoother` | `GazeSmootherReID` | run context base | Gaze smoothing with re-identification |
| `locker` | `GazeLockTracker` | run context base | Fixation lock-on / dwell tracker |
| `snap_temporal` | `SnapTemporalState` | run context base | Temporal snap engage/release state (replaces the old `snap_hysteresis`) |
| `all_trackers` | `list` | run context base | All active phenomena tracker instances |
| `pid_map` | `dict` or `None` | run context base | Track ID to participant-label mapping |
| `aux_frames` | `dict` | run loop | Auxiliary stream frames keyed by the 3-tuple `(pid, stream_label, video_type)` |
| `anonymize` | `str` or `None` | run context base | Face anonymization mode (`"blur"`, `"black"`, or `None`) |

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
