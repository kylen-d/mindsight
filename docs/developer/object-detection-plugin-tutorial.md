# Object Detection Plugin Tutorial: Confidence Boosting

A full worked example building a realistic **GazeBoost** ObjectDetection plugin that post-processes YOLO detections to boost confidence of objects near people. Use this alongside the [Writing a Plugin](writing-a-plugin.md) guide to see how each concept plays out in production code.

For tutorials on other plugin types, see: [Phenomena Plugin Tutorial](phenomena-plugin-tutorial.md) | [Gaze Plugin Tutorial](gaze-plugin-tutorial.md) | [Data Collection Plugin Tutorial](data-collection-plugin-tutorial.md)

---

## Overview

ObjectDetectionPlugins augment, filter, or replace YOLO detections after the default detection pass. They are called once per frame via `detect()`. This tutorial builds **GazeBoost** — a plugin that increases confidence of objects near gaze ray endpoints, reducing false negatives for objects being looked at.

Source (hypothetical): `Plugins/ObjectDetection/GazeBoost/gaze_boost.py`

---

## What It Does

For each detected object: if any person detection's bounding box is within a configurable margin, boost the object's confidence by a factor. This helps retain detections at the detection confidence threshold boundary when behavioral evidence (proximity to a person) supports their presence.

The intuition is simple: objects near people are more likely to be real detections because they are in the interaction space. A knife with 0.25 confidence sitting on a table far from anyone might be a false positive, but the same knife at 0.25 confidence right next to a person's hand is worth keeping.

---

## File Structure

```
Plugins/ObjectDetection/GazeBoost/
├── __init__.py          # empty
└── gaze_boost.py        # PLUGIN_CLASS = GazeBoostPlugin
```

The `__init__.py` is empty. All logic lives in `gaze_boost.py`.

---

## Class Definition

```python
from Plugins import ObjectDetectionPlugin
from ObjectDetection.detection import Detection

class GazeBoostPlugin(ObjectDetectionPlugin):
    name = "gaze_boost"

    def __init__(self, boost_factor: float = 1.3, margin_px: float = 50.0):
        self._boost = boost_factor
        self._margin = margin_px
```

### Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `boost_factor` | `1.3` | Multiplicative confidence boost for objects near people. Capped at 1.0. |
| `margin_px` | `50.0` | Pixel margin added around person bounding boxes when checking proximity. |

The `name` attribute is the string used in logging and the plugin registry. It must be unique across all ObjectDetection plugins.

---

## The `detect()` Method

This is the only required method beyond `__init__`. It receives the current frame, all detections from the YOLO pass, and the active detection configuration.

```python
def detect(self, *, frame, detection_frame, all_dets, det_cfg, **kwargs):
    persons = [d for d in all_dets if d['class_name'].lower() == 'person']
    if not persons:
        return None  # nothing to boost against

    modified = False
    for det in all_dets:
        if det['class_name'].lower() == 'person':
            continue
        for p in persons:
            if self._near(det, p):
                det['conf'] = min(1.0, det['conf'] * self._boost)
                modified = True
                break
    return all_dets if modified else None
```

### Step-by-step walkthrough

1. **Extract person detections.** We use person bounding boxes as a proxy for "interesting" regions. If there are no people in the frame, there is nothing to boost against — return `None` immediately.

2. **Iterate non-person detections.** For each object detection, check whether it is near any person bounding box using the `_near()` helper (defined below).

3. **Boost in-place.** Detection objects are mutable — modify `det['conf']` directly. Cap at `1.0` to keep confidence in the valid range.

4. **Return `None` if nothing changed.** This is an important convention: returning `None` tells the pipeline "I made no changes", which avoids unnecessary list copying. Only return the modified list when at least one detection was actually boosted.

### The proximity helper

```python
def _near(self, det, person) -> bool:
    """Return True if det's bbox is within margin_px of person's bbox."""
    dx1, dy1, dx2, dy2 = det['bbox']
    px1, py1, px2, py2 = person['bbox']

    # Expand person bbox by margin
    px1 -= self._margin
    py1 -= self._margin
    px2 += self._margin
    py2 += self._margin

    # Check overlap
    return not (dx2 < px1 or dx1 > px2 or dy2 < py1 or dy1 > py2)
```

The margin expansion is applied to the person bbox, not the object. This means "objects within 50 pixels of a person" rather than "objects whose expanded box touches a person".

### What `detect()` receives

| Keyword | Type | Description |
|---------|------|-------------|
| `frame` | `np.ndarray` | The original video frame (BGR). |
| `detection_frame` | `np.ndarray` | The frame as preprocessed for detection (may be resized). |
| `all_dets` | `list[Detection]` | All detections from the YOLO pass, including persons. |
| `det_cfg` | `DetectionConfig` | Current detection configuration (thresholds, model path, etc.). |

The `**kwargs` catch-all ensures forward compatibility as new fields are added.

---

## CLI Activation

Every plugin provides two classmethods for CLI integration: `add_arguments` registers argparse flags, and `from_args` decides whether to instantiate the plugin based on those flags.

```python
@classmethod
def add_arguments(cls, parser):
    g = parser.add_argument_group("Gaze Boost plugin")
    g.add_argument("--gaze-boost", action="store_true",
                   help="Enable confidence boosting for objects near people.")
    g.add_argument("--gaze-boost-factor", type=float, default=1.3,
                   help="Multiplicative boost factor (default: 1.3).")
    g.add_argument("--gaze-boost-margin", type=float, default=50.0,
                   help="Pixel margin around person bboxes (default: 50).")

@classmethod
def from_args(cls, args):
    if not getattr(args, "gaze_boost", False):
        return None
    return cls(boost_factor=getattr(args, "gaze_boost_factor", 1.3),
               margin_px=getattr(args, "gaze_boost_margin", 50.0))
```

### Why `getattr` with defaults?

When plugins are loaded dynamically, the argument namespace may not contain your plugin's flags (e.g., when running from the GUI or when another entry point skips plugin argument registration). Using `getattr` with a fallback default makes the plugin robust to these situations.

---

## Running It

```bash
python MindSight.py --source video.mp4 --gaze-boost --gaze-boost-factor 1.5
```

This runs the standard MindSight pipeline with GazeBoost enabled. Any object detection with confidence below the threshold but within 50 pixels of a person will get a 1.5x confidence boost, potentially pushing it above the threshold and into the final detection set.

---

## Key Design Patterns

### 1. `detect()` runs after YOLO but before person/object split

Your plugin sees **all** detections in a single flat list, including persons. This is intentional — it lets you use person detections as context for boosting or filtering object detections, as GazeBoost does.

### 2. Return `None` when nothing changed

```python
return all_dets if modified else None
```

Returning `None` (not the unchanged list) signals "no modifications" to the pipeline. This avoids unnecessary list copying and makes it easy for the pipeline to short-circuit.

### 3. Detection objects are mutable

Detections behave like dictionaries. Modify fields in-place:

```python
det['conf'] = min(1.0, det['conf'] * self._boost)
```

You can also add new keys (e.g., `det['boosted'] = True`) for downstream plugins or phenomena to read.

### 4. Access detection configuration

The `det_cfg` parameter gives you the current `DetectionConfig`, including the active confidence threshold. This is useful for plugins that need threshold-aware logic:

```python
if det['conf'] < det_cfg.conf_threshold and boosted_conf >= det_cfg.conf_threshold:
    # This detection was rescued from below threshold
    det['conf'] = boosted_conf
```

### 5. Multiple ObjectDetection plugins are chained

If multiple ObjectDetection plugins are active, they are called in sequence. Each receives the output of the previous plugin. Design your plugin to be composable — don't assume you are the only one modifying detections.

---

## Limitations and Design Considerations

- **`detect()` receives previous-frame gaze data via `**kwargs`.** The pipeline now passes `prev_persons_gaze` and `prev_face_track_ids` from the previous frame. See `Plugins/ObjectDetection/GazeBoost/` for a production plugin that uses gaze endpoints to boost detection confidence, including sub-threshold rescue.

- **ObjectDetectionPlugins are best for:** NMS adjustments, class remapping, bounding box refinement, confidence recalibration, and external detector fusion (e.g., merging detections from a second model).

- **Avoid heavy computation in `detect()`.** It is called every frame. If your plugin needs expensive setup (loading a model, reading a config file), do it in `__init__`.

- **Be careful with confidence boosting near 1.0.** Downstream code may treat high-confidence detections differently. Consider capping your boost so that boosted detections never exceed, say, 0.95.

---

## Complete Code

```python
from Plugins import ObjectDetectionPlugin
from ObjectDetection.detection import Detection


class GazeBoostPlugin(ObjectDetectionPlugin):
    """Boost confidence of object detections near person bounding boxes."""

    name = "gaze_boost"

    def __init__(self, boost_factor: float = 1.3, margin_px: float = 50.0):
        self._boost = boost_factor
        self._margin = margin_px

    # ── Detection hook ──────────────────────────────────────────────

    def detect(self, *, frame, detection_frame, all_dets, det_cfg, **kwargs):
        persons = [d for d in all_dets if d['class_name'].lower() == 'person']
        if not persons:
            return None

        modified = False
        for det in all_dets:
            if det['class_name'].lower() == 'person':
                continue
            for p in persons:
                if self._near(det, p):
                    det['conf'] = min(1.0, det['conf'] * self._boost)
                    modified = True
                    break
        return all_dets if modified else None

    def _near(self, det, person) -> bool:
        dx1, dy1, dx2, dy2 = det['bbox']
        px1, py1, px2, py2 = person['bbox']
        px1 -= self._margin
        py1 -= self._margin
        px2 += self._margin
        py2 += self._margin
        return not (dx2 < px1 or dx1 > px2 or dy2 < py1 or dy1 > py2)

    # ── CLI integration ─────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser):
        g = parser.add_argument_group("Gaze Boost plugin")
        g.add_argument("--gaze-boost", action="store_true",
                       help="Enable confidence boosting for objects near people.")
        g.add_argument("--gaze-boost-factor", type=float, default=1.3,
                       help="Multiplicative boost factor (default: 1.3).")
        g.add_argument("--gaze-boost-margin", type=float, default=50.0,
                       help="Pixel margin around person bboxes (default: 50).")

    @classmethod
    def from_args(cls, args):
        if not getattr(args, "gaze_boost", False):
            return None
        return cls(boost_factor=getattr(args, "gaze_boost_factor", 1.3),
                   margin_px=getattr(args, "gaze_boost_margin", 50.0))


PLUGIN_CLASS = GazeBoostPlugin
```
