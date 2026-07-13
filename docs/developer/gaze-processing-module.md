# Gaze Processing Module

Developer reference for gaze estimation and ray forming in MindSight.

## 1. Overview

Two packages share the work:

| Package | Role |
|---|---|
| `mindsight/GazeTracking/` | Backend selection, per-frame coordination, per-face pitch/yaw estimation, and the re-identifying smoother. |
| `mindsight/PostProcessing/RayForming/` | The **primary ray-forming path** — turns raw pitch/yaw (plus optional Gaze-LLE heatmaps) into finalized gaze rays through blend, depth, snap, lock-on, and hit detection. |

Key files in `GazeTracking/`:

| File | Purpose |
|---|---|
| `gaze_factory.py` | `create_gaze_engine(...)` selects and instantiates the active gaze backend. |
| `gaze_pipeline.py` | `run_gaze_step(ctx, ...)` — per-frame coordinator that wires detection, estimation, and ray forming together. |
| `gaze_processing.py` | `GazeSmootherReID` (temporal EMA + colour-histogram Re-ID) and the extensible `GazeToolkit`. |
| `pitchyaw_pipeline.py` | Per-face pitch/yaw estimation helpers. |
| `Backends/MGaze/` | The built-in **MobileGaze** per-face backend (this is the *only* backend under `Backends/`). |

!!! note "Gaze-LLE is a plugin, not a backend sibling"
    Gaze-LLE does **not** live in `GazeTracking/Backends/`. It ships as a
    discovered plugin at `Plugins/GazeTracking/Gazelle/`. Its heatmaps feed the
    ray-forming blend through `PostProcessing/RayForming/gazelle_provider.py` and
    `gazelle_blender.py` (see §5).

## 2. Per-frame flow

`run_gaze_step(ctx, *, face_det, gaze_eng, gaze_cfg, **kwargs)` is the entry
point. It runs one of three paths, then a shared post-processing tail:

```
face_det.detect(detection_frame)     # RetinaFace; rescaled by inverse_scale
        │
        ├─ Path A (per_face backend + ray_cfg): PRIMARY
        │    _estimate_pitchyaw → RawGaze list
        │    gazelle_provider.step(...)            # optional heatmap inference
        │    run_ray_forming(raw_gazes, objects, ... )   →  RayFormingResult
        │
        ├─ Path B (backend overrides run_pipeline, e.g. Gaze-LLE plugin)
        │    gaze_eng.run_pipeline(frame=..., faces=..., snap_temporal=..., ...)
        │
        └─ Path C (scene backend, no ray_cfg): _default_scene_pipeline(...)
        │
   apply_tip_snapping(...)      # per-face ray-to-ray tip snapping
   apply_lock_on(...)           # fixation lock-on
   compute_ray_intersections(...)   # ray-bbox / ray-cone hits + confidence gate
```

**ctx reads:** `frame`, `detection_frame`, `inverse_scale`, `objects`,
`cached_faces`, `smoother`, `locker`, `snap_temporal`, `smooth_snap_tracker`,
`gazelle_provider`, `ray_cfg`, `depth_map`, `depth_cfg`, `video_fps`.

**ctx writes:** `persons_gaze`, `face_confs`, `face_bboxes`, `face_track_ids`,
`all_targets`, `hits`, `hit_events`, `lock_info`, `ray_snapped`, `ray_extended`,
`faces`.

Path A is used whenever the backend is per-face (`mode == "per_face"`) and a
`RayFormingConfig` is present — this is the default MobileGaze route.

## 3. Gaze factory

```
create_gaze_engine(...) -> GazePlugin
```

Selects the active backend. Plugin backends registered in `gaze_registry`
(the first whose `from_args` returns non-`None`) win; otherwise the built-in
MobileGaze backend is used. The returned engine conforms to the `GazePlugin`
interface — see the [plugin base classes reference](../reference/plugin-base-classes.md).

## 4. GazeSmootherReID

**File:** `gaze_processing.py`

Per-face temporal EMA smoother with colour-histogram re-identification. Its one
public entry point is `update` — there is no `smooth_and_track` method.

```python
def update(self, faces):
    """
    faces:   [(center, pitch, yaw, crop)]      # crop is optional per entry
    Returns: [(smooth_pitch, smooth_yaw, track_id)]
    """
```

Constructor: `GazeSmootherReID(alpha=SMOOTH_ALPHA, max_dist=200,
hist_weight=0.35, hist_bins=16, grace_frames=0)`.

- Match score = `positional_distance * (1 + hist_weight * bhattacharyya_dist)`;
  `hist_weight=0` falls back to positional-only matching.
- `grace_frames > 0` holds an unmatched track in a dead-track buffer for that
  many frames so the original ID is revived across brief occlusions.
- `_estimate_global_shift` subtracts median inter-frame displacement before
  matching so camera motion does not blow past `max_dist` (handheld/moving-rig
  robustness).

`GazeToolkit` (same file) is an extensible base plugin authors can subclass to
override or add processing steps.

## 5. RayForming package (primary path)

**Package:** `mindsight/PostProcessing/RayForming/`. Public API is re-exported
from its `__init__.py`:

```python
from mindsight.PostProcessing.RayForming import (
    run_ray_forming, RawGaze, RayFormingConfig,
    GazeLLEBlender, HeatmapCache, ObjectSnap,
    SmoothSnapTracker, SnapTemporalState, snap_score, apply_tip_snapping,
    GazeLockTracker, apply_lock_on, compute_ray_intersections, GazelleProvider,
)
```

Module-level members:

| Module | Key members |
|---|---|
| `ray_pipeline.py` | `run_ray_forming(...)`, `RawGaze`, `RayFormingResult` |
| `ray_config.py` | `RayFormingConfig` (the largest config object; `.from_namespace(args)`) |
| `gazelle_provider.py` | `GazelleProvider` — owns the scheduler + heatmap cache; `.from_namespace(args, device=...)` |
| `gazelle_blender.py` | `GazeLLEBlender` — turns (accept, trust) into a smoothed endpoint |
| `inference_scheduler.py` | `InferenceScheduler` — fixation-aware Gaze-LLE call gating |
| `heatmap_cache.py` | `HeatmapCache` |
| `fixation.py` | `GazeLockTracker`, `apply_lock_on` |
| `object_snap.py` | `ObjectSnap`, `SmoothSnapTracker`, `SnapTemporalState`, `snap_score`, `apply_tip_snapping` |
| `hit_detection.py` | `compute_ray_intersections` |
| `depth_ray.py` | `depth_adjusted_length` |

### run_ray_forming

```python
def run_ray_forming(
    raw_gazes: list[RawGaze], objects: list, face_objs: list,
    frame_h: int, frame_w: int, cfg: RayFormingConfig, *,
    gazelle_provider=None, gazelle_blender=None, object_snap=None,
    depth_map=None, dt: float = 1.0/30.0,
) -> RayFormingResult
```

Per-face order: (1) base ray from pitch/yaw + confidence-scaled length, (2)
forward-gaze dead-zone check, (3) Gaze-LLE blend (scheduler trust + One-Euro
smoothing), (4) depth-adjusted length, (5) object snap. `RawGaze` fields:
`origin, pitch, yaw, confidence, face_width, track_id, face_bbox`.
`RayFormingResult` fields: `persons_gaze, face_confs, face_bboxes,
face_track_ids, face_objs, ray_snapped, ray_extended`, where each `persons_gaze`
entry is `(origin, ray_end, (pitch, yaw))`.

## 6. Snap and lock-on state

### SnapTemporalState (`object_snap.py`)

Lightweight per-face temporal state for snap scoring — this **replaces** the
pre-1.0 `SnapHysteresisTracker`, which no longer exists.

```python
SnapTemporalState(release_frames=5, engage_frames=0, grid_px=40)
```

- `release_frames` — no-match frames before releasing a held snap.
- `engage_frames` — consecutive matches required before first engaging (0 =
  instant).
- `grid_px` — grid cell size for quantising object centres to stable keys.

`update(face_idx, snap_center, found, gaze_conf=None) -> (filtered_center_or_None,
did_snap)` handles the release countdown (accelerated when `gaze_conf` is low)
and the optional engage delay.

### apply_tip_snapping (`object_snap.py`)

```python
apply_tip_snapping(persons_gaze, ray_snapped, ray_extended, gaze_eng, gaze_cfg,
                   *, face_track_ids=None, smooth_snap_tracker=None)
    -> (persons_gaze, ray_snapped, ray_extended)
```

Snaps an unsnapped ray tip to another person's ray tip when the rays converge
within `gaze_cfg.snap_dist`.

### GazeLockTracker (`fixation.py`)

Moved here from `gaze_processing.py`. Fixation lock-on: snaps a gaze ray to an
object after `dwell_frames` of sustained attention.

```python
GazeLockTracker(dwell_frames=15, release_frames=10, lock_dist=100, max_face_dist=120)
def update(self, persons_gaze, objects) -> list  # [(snapped_end, obj_idx_or_None, frac)]
```

Wrapper used by the coordinator:

```python
apply_lock_on(persons_gaze, locker, objects) -> (persons_gaze, lock_info)
# lock_info: [(obj_idx_or_None, frac)]; ray_end is snapped where locked
```

## 7. Ray geometry

**File:** `mindsight/utils/geometry.py`. The bbox helpers take a **detection
dict** (with `x1, y1, x2, y2` keys), not four scalars.

| Function | Signature |
|---|---|
| `pitch_yaw_to_2d` | `(pitch, yaw) -> ndarray` (unit 2-D direction) |
| `ray_hits_box` | `(start, end, x1, y1, x2, y2) -> bool` (Liang-Barsky) |
| `ray_hits_cone` | `(origin, direction, x1, y1, x2, y2, cone_half_angle_deg, ray_length=None) -> bool` |
| `extend_ray` | `(origin, end, length=_RAY_EXT_LENGTH) -> ndarray` |
| `bbox_center` | `(obj) -> ndarray` (obj is a detection dict) |
| `bbox_diagonal` | `(obj) -> float` |
| `sample_depth_patch` | `(depth_map, x, y, radius=2) -> float` (median of patch) |

## 8. Hit detection

```python
compute_ray_intersections(persons_gaze, face_confs, face_track_ids, face_objs,
                          objects, cfg, *, depth_map=None, gaze_sample_radius=2)
    -> (all_targets, hits, hit_events)
```

Tests ray-bbox (or ray-cone) intersection for every (person, object) pair and
gates by face-detection confidence. `hits` is a set of `(face_list_idx,
obj_list_idx)` pairs; `hit_events` is a list of per-hit dicts keyed by stable
track ID.

## 9. Backends

| Backend | Location | Model | Granularity |
|---|---|---|---|
| **MobileGaze** | `mindsight/GazeTracking/Backends/MGaze/` | ONNX or PyTorch (auto-detected) | Per-face (`mode="per_face"`) |
| **Gaze-LLE** | `Plugins/GazeTracking/Gazelle/` (plugin) | DINOv2 heatmap | Scene-level; blended into ray forming |

MobileGaze is the default per-face backend and drives Path A. Gaze-LLE is a
discovered plugin whose heatmaps are consumed as a blend signal in the primary
ray-forming path (or, standalone, via Path B / Path C).
