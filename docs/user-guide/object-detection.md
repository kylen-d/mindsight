# Object Detection

## Overview

MindSight uses YOLO for real-time object detection in video frames. Two modes are available:

- **Standard YOLO** -- text-class prompting using COCO class names.
- **YOLOE** -- visual prompting with reference images and annotated bounding boxes, enabling zero-shot detection of arbitrary objects.

Detected objects are separated into two categories:

| Category | Purpose |
|----------|---------|
| **Persons** | Used as input for face detection and gaze estimation |
| **Objects** | Treated as gaze targets for intersection testing |

This separation drives the downstream gaze pipeline: persons produce gaze rays, objects receive them.

---

## YOLO Model Selection

Select a model with the `--model` flag:

```
--model yolov8n.pt
```

Available models, from fastest to most accurate:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | Nano (default) | Fastest | Lowest |
| `yolov8s.pt` | Small | Fast | Low |
| `yolov8m.pt` | Medium | Moderate | Moderate |
| `yolov8l.pt` | Large | Slow | High |
| `yolov8x.pt` | Extra-large | Slowest | Highest |

Weights are auto-downloaded on first use and cached locally.

---

## Filtering Classes

Restrict detection to specific COCO class names with `--classes`:

```
--classes person knife cup
```

Exclude specific classes with `--blacklist`:

```
--blacklist chair couch
```

Both flags accept one or more space-separated COCO class names. When `--classes` is set, only those classes are detected. When `--blacklist` is set, those classes are excluded from results. If both are provided, `--classes` is applied first, then `--blacklist` removes from that set.

---

## Confidence Threshold

```
--conf 0.35
```

The confidence threshold (default `0.35`) controls the minimum score a detection must reach to be kept. Higher values reduce false positives but may miss weaker detections. Lower values produce more detections at the cost of more noise.

---

## Detection Scale

```
--detect-scale 1.0
```

Values less than `1.0` downscale the frame before running detection, then rescale the resulting coordinates back to the original resolution. This trades detection accuracy for speed -- useful on high-resolution video or slower hardware.

| Value | Effect |
|-------|--------|
| `1.0` | Full resolution (default) |
| `0.5` | Half resolution, roughly 4x faster |
| `0.25` | Quarter resolution, roughly 16x faster |

---

## Object Persistence Cache

```
--obj-persistence N
```

When an object disappears from detection (due to momentary occlusion or a YOLO miss), the persistence cache keeps its last-known bounding box alive for `N` additional frames. This prevents downstream gaze hits from flickering.

- Default: `0` (disabled).
- Ghost detections are rendered slightly transparent to distinguish them from live detections.
- The cached bounding box is static (not interpolated), so large values may produce stale positions.

---

## Visual Prompt Mode (YOLOE)

Instead of detecting objects by COCO class name, visual prompt mode lets you provide reference images with annotated bounding boxes. This enables zero-shot detection of custom objects that are not in the COCO class set.

```
--vp-file prompt.vp.json --vp-model yoloe-26l-seg.pt
```

The `.vp.json` file describes the reference images and their annotated regions. YOLOE uses these visual examples to locate matching objects in video frames.

For a full walkthrough on creating and using visual prompts, see the [Visual Prompts guide](visual-prompts.md).

---

## Parameter Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `yolov8n.pt` | YOLO model weights |
| `--conf` | float | `0.35` | Detection confidence threshold |
| `--classes` | str[] | `[]` | Filter to specific COCO class names |
| `--blacklist` | str[] | `[]` | Exclude specific COCO class names |
| `--skip-frames` | int | `1` | Run detection every N frames |
| `--detect-scale` | float | `1.0` | Scale factor for detection pass |
| `--vp-file` | str | None | Path to `.vp.json` visual prompt file |
| `--vp-model` | str | `yoloe-26l-seg.pt` | YOLOE model for VP mode |
| `--obj-persistence` | int | `0` | Frames to keep ghost detections alive |

!!! tip "Under the hood"
    For implementation details, see [developer/object-detection-module.md](../developer/object-detection-module.md).
