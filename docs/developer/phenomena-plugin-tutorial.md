# Phenomena Plugin Tutorial: Novel Salience

A full worked example walking through the real **NovelSalience** phenomena plugin that ships with MindSight. Use this alongside the [Writing a Plugin](writing-a-plugin.md) guide to see how each concept plays out in production code.

For tutorials on other plugin types, see: [Gaze Plugin Tutorial](gaze-plugin-tutorial.md) | [Object Detection Plugin Tutorial](object-detection-plugin-tutorial.md) | [Data Collection Plugin Tutorial](data-collection-plugin-tutorial.md)

---

## Overview

NovelSalience detects rapid gaze shifts (saccades) as a proxy for novel stimulus salience. When a participant's gaze endpoint moves faster than a configurable speed threshold, an event is fired. The plugin works with both per-face (pitch/yaw) and scene-level (Gazelle) gaze backends.

Source: `Plugins/Phenomena/NovelSalience/novel_salience.py`

---

## What It Does

When the gaze endpoint moves faster than `speed_thresh` pixels/frame:

1. A **novel salience event** is recorded with direction (`LEFT`, `RIGHT`, `UP`, `DOWN`), speed, and frame number.
2. A **fading cyan ring** is drawn around the face bounding box.
3. A **directional arrow** appears at the gaze endpoint indicating the saccade direction.
4. A **text label** ("NS! LEFT", etc.) is shown near the endpoint.

The direction is derived from screen-space displacement of the gaze ray endpoint, so it is always correct regardless of which gaze backend is active.

---

## File Structure

```
Plugins/Phenomena/NovelSalience/
    __init__.py          # empty
    novel_salience.py    # PLUGIN_CLASS = NovelSalienceTracker
```

The `__init__.py` is empty. All logic lives in `novel_salience.py`.

---

## Class Definition

```python
class NovelSalienceTracker(PhenomenaPlugin):

    name            = "novel_salience"
    dashboard_panel = "right"

    _NS_COL = (0, 215, 255)   # BGR vivid amber-cyan

    def __init__(
        self,
        speed_thresh: float = 40.0,
        cooldown: int       = 20,
        history: int        = 2,
        flash: int          = 12,
    ) -> None:
        self._thresh          = speed_thresh
        self._cooldown_frames = cooldown
        self._hist_len        = max(1, history)
        self._flash_len       = max(1, flash)

        # Per-track-ID state
        self._pos_hist:   dict[int, collections.deque] = {}
        self._angle_hist: dict[int, collections.deque] = {}
        self._cooldown:   dict[int, int]               = {}
        self._flash:      dict[int, dict]              = {}

        # Full event log
        self.events: list[dict] = []

        # Cached for draw_frame
        self._last_persons_gaze:   list = []
        self._last_face_bboxes:    list = []
        self._last_face_track_ids: list = []
```

### Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `speed_thresh` | 40.0 | Minimum gaze-endpoint speed in px/frame to flag an event. Lower values = more sensitive. |
| `cooldown` | 20 | Frames between consecutive events for the same face track. Prevents one long saccade from generating a burst of events. |
| `history` | 2 | Sliding-window depth for velocity smoothing. 1 = instantaneous, 3 = heavier smoothing. |
| `flash` | 12 | How many frames the visual ring/arrow indicator persists after an event fires. |

### State

- **`_pos_hist`** -- A `deque` per track ID storing recent gaze-ray endpoint positions. Used to compute smoothed velocity.
- **`_cooldown`** -- Frames remaining before the next event can fire for each track.
- **`_flash`** -- Active visual indicators. Each entry holds `frames` (remaining), `direction`, and `speed_px`.
- **`events`** -- The full event log. Every event fired during the run is appended here and persists for CSV output.

---

## The `update()` Method

This is the core per-frame logic. Here is a step-by-step walkthrough.

### 1. Get frame data from kwargs

```python
frame_no       = kwargs['frame_no']
persons_gaze   = kwargs.get('persons_gaze', [])
face_bboxes    = kwargs.get('face_bboxes', [])
face_track_ids = kwargs.get('face_track_ids')
```

Note that `face_track_ids` may be `None` if no re-ID tracker is running, so the plugin falls back to index-based IDs:

```python
tids = face_track_ids if face_track_ids is not None \
       else list(range(len(persons_gaze)))
```

### 2. For each tracked face: update position history, compute smoothed velocity

```python
for fi, (origin, ray_end, angles) in enumerate(persons_gaze):
    tid = tids[fi] if fi < len(tids) else fi

    # Tick down cooldown and flash timers
    if self._cooldown.get(tid, 0) > 0:
        self._cooldown[tid] -= 1

    # Store current position in the sliding window
    pos = np.array([float(ray_end[0]), float(ray_end[1])])
    self._pos_hist[tid].append(pos)

    # Smoothed velocity: average displacement over the window
    deltas = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
    mean_delta = np.mean(deltas, axis=0)
    speed_px   = float(np.linalg.norm(mean_delta))
```

The smoothing averages displacement vectors across the `history` window, which filters out single-frame jitter while still responding quickly to real saccades.

### 3. Determine direction from displacement vector

```python
ax, ay = float(mean_delta[0]), float(mean_delta[1])
if abs(ax) >= abs(ay):
    direction = "LEFT" if ax < 0 else "RIGHT"
else:
    direction = "UP" if ay < 0 else "DOWN"
```

Direction is always derived from screen-space displacement (not pitch/yaw), so it works identically with any gaze backend.

### 4. If speed > threshold AND cooldown expired, fire event

```python
if speed_px >= self._thresh and self._cooldown.get(tid, 0) == 0:
    event = {
        'frame_no':  frame_no,
        'face_id':   tid,
        'speed_px':  speed_px,
        'speed_deg': speed_deg,
        'direction': direction,
        'delta_x':   ax,
        'delta_y':   ay,
    }
    self.events.append(event)
    self._cooldown[tid] = self._cooldown_frames
    self._flash[tid]    = {
        'frames':    self._flash_len,
        'direction': direction,
        'speed_px':  speed_px,
    }
```

The event is stored in `self.events` (for CSV) and the flash indicator is started (for `draw_frame`).

### Return value

```python
return {'events': current_events}
```

Returns the list of events fired this frame. Other pipeline components or the GUI can inspect this.

---

## Visual Overlay: `draw_frame()`

Called every frame after `update()`. Draws indicators for each face with an active flash.

```python
def draw_frame(self, frame) -> None:
    if not self._flash:
        return                      # early exit when nothing to draw

    for fi, (origin, ray_end, _) in enumerate(self._last_persons_gaze):
        tid = self._last_face_track_ids[fi]
        if tid not in self._flash:
            continue

        fdata = self._flash[tid]
        frac  = fdata['frames'] / self._flash_len   # 1.0 -> 0.0 fade
        col   = tuple(int(c * frac) for c in self._NS_COL)
```

Three visual elements are drawn:

1. **Ring around face bbox** -- `cv2.circle` with thickness proportional to `frac`, so it fades smoothly.
2. **Directional arrow** -- `cv2.arrowedLine` from the gaze endpoint in the saccade direction. Length and thickness both scale with `frac`.
3. **Text label** -- `"NS! LEFT"` (or whichever direction) placed near the gaze endpoint.

The fade is frame-based (not time-based), so it looks consistent regardless of video FPS.

---

## Dashboard Integration

### `dashboard_data()` (new-style)

Returns a structured dict consumed by the GUI dashboard renderer:

```python
def dashboard_data(self, *, pid_map=None) -> dict:
    rows = []
    if self.events:
        for ev in self.events[-3:]:
            plbl = resolve_display_pid(ev['face_id'], pid_map)
            rows.append({
                'label': f"{plbl} ->  {ev['direction']}",
                'value': f"{speed_str}  @f{ev['frame_no']}",
            })
        # Per-face tally as final row
        rows.append({'label': f"total: {tally}"})
    return {
        'title':      'NOVEL SALIENCE',
        'colour':     self._NS_COL,
        'rows':       rows,
        'empty_text': '--',
    }
```

Shows the 3 most recent saccade events (face label, direction, speed, frame number) plus a per-face tally line.

### `dashboard_section()` (legacy, direct-draw)

Same data, but drawn directly onto the panel `ndarray` using `_draw_panel_section`. Both methods are implemented for backward compatibility.

---

## CSV Output

`csv_rows()` produces two sections:

### Per-event detail

```
novel_salience_events
category, frame_no, face_id, direction, speed_px, speed_deg, delta_x, delta_y
novel_salience, 142, P1, LEFT, 53.21, 4.12, -48.30, 22.10
novel_salience, 207, P2, DOWN, 61.88, 5.03, 12.44, 60.62
...
```

### Per-face summary

```
novel_salience_summary, face_id, event_count, total_frames, rate_pct
novel_salience_summary, P1, 8, 3600, 0.2222
novel_salience_summary, P2, 3, 3600, 0.0833
```

Uses `resolve_display_pid(track_id, pid_map)` to convert internal track IDs into human-readable participant labels.

---

## CLI Activation

Five flags in a dedicated argument group:

```python
@classmethod
def add_arguments(cls, parser) -> None:
    g = parser.add_argument_group("Novel Salience plugin")
    g.add_argument("--novel-salience",   action="store_true",
                   help="Enable novel-salience detection.")
    g.add_argument("--ns-speed-thresh",  type=float, default=40.0, metavar="PX")
    g.add_argument("--ns-cooldown",      type=int,   default=20,   metavar="N")
    g.add_argument("--ns-history",       type=int,   default=2,    metavar="N")
    g.add_argument("--ns-flash",         type=int,   default=12,   metavar="N")
```

`from_args` checks the activation flag, constructs the instance, and prints a config summary:

```python
@classmethod
def from_args(cls, args):
    if not getattr(args, "novel_salience", False):
        return None
    inst = cls(
        speed_thresh = getattr(args, "ns_speed_thresh", 40.0),
        cooldown     = getattr(args, "ns_cooldown",     20),
        history      = getattr(args, "ns_history",      2),
        flash        = getattr(args, "ns_flash",        12),
    )
    print(f"NovelSalience: thresh={inst._thresh}px/f  "
          f"cooldown={inst._cooldown_frames}f  "
          f"history={inst._hist_len}f  flash={inst._flash_len}f")
    return inst
```

---

## Running It

```bash
python MindSight.py --source video.mp4 --novel-salience --ns-speed-thresh 30 --ns-cooldown 15
```

Expected output on startup:

```
NovelSalience: thresh=30.0px/f  cooldown=15f  history=2f  flash=12f
```

During playback, saccade events will appear as fading cyan rings and arrows in the video overlay, and as entries in the right dashboard panel.

---

## Key Design Patterns

What makes NovelSalience a well-built plugin:

- **Only pulls needed kwargs.** Uses `kwargs.get()` with defaults, so the plugin never breaks when new keys are added to the pipeline.
- **Uses `resolve_display_pid` for participant labels.** Internal track IDs are integers; display labels (P1, S70, etc.) are resolved via `pid_map` at output time.
- **Lazy imports for dashboard helpers.** The `_dash()` helper function imports from `DataCollection.dashboard_output` only when called, so the plugin module loads even if the DataCollection package is not yet initialised.
- **Smooth visual indicators with frame-based fade.** The `frac = frames_remaining / flash_len` pattern gives a linear fade that is consistent across video frame rates.
- **Both `dashboard_section` (legacy) and `dashboard_data` (new) implemented.** This ensures the plugin works with both the direct-draw dashboard renderer and the newer structured-data renderer used by the GUI.
