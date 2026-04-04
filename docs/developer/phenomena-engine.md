# Phenomena Engine

Developer reference for the phenomena tracking system in MindSight.

## 1. Overview

The phenomena engine is spread across four files:

| File | Role |
|---|---|
| `phenomena_pipeline.py` | Per-frame coordinator and lifecycle manager |
| `phenomena_config.py` | Configuration dataclass with all toggles and parameters |
| `phenomena_tracking.py` | CLI argument registration and re-exports for backward compatibility |
| `helpers.py` | Shared utility functions (joint attention, gaze convergence) |

## 2. Phenomena Pipeline

**File:** `phenomena_pipeline.py`

### init_phenomena_trackers

```
init_phenomena_trackers(cfg: PhenomenaConfig) -> list[PhenomenaPlugin]
```

Instantiates trackers from configuration flags. Ordering is significant:

1. **JA tracker is always first** -- downstream trackers depend on `confirmed_objs` that JA produces.
2. **Left-panel trackers:** `mutual_gaze`, `social_ref`, `gaze_follow`
3. **Right-panel trackers:** `attn_span`, `gaze_aversion`, `scanpath`, `gaze_leader`

### update_phenomena_step

```
update_phenomena_step(ctx)
```

Called once per frame:

1. Builds a `tracker_kwargs` dict from the current `FrameContext`:
   - `frame_no`, `persons_gaze`, `face_bboxes`, `hit_events`, `joint_objs`, `dets`, `n_faces`, `face_track_ids`, `hits`, `tip_convergences`, etc.
2. Iterates `all_trackers` and calls `tracker.update(**tracker_kwargs)` on each.
3. The JA tracker's return dict sets `confirmed_objs`, `extra_hud`, and `joint_pct` in `ctx`.
4. Subsequent trackers see the updated `joint_objs` if JA modifies it.

### post_run_summary

```
post_run_summary(all_trackers, total_frames, pid_map)
```

Calls `console_summary()` on each tracker after video processing completes.

## 3. PhenomenaConfig

**File:** `phenomena_config.py`

A dataclass holding all phenomena toggles and their parameters. `from_namespace()` constructs an instance from parsed CLI args and honours the `--all-phenomena` flag.

| Field | Type | Description |
|---|---|---|
| `joint_attention` | bool | Enable joint attention tracking |
| `ja_window` | int | Sliding window size (frames) |
| `ja_window_thresh` | float | Threshold within the JA window |
| `ja_quorum` | int | Minimum participants for JA |
| `mutual_gaze` | bool | Enable mutual gaze detection |
| `social_ref` | bool | Enable social referencing |
| `social_ref_window` | int | Social referencing window (frames) |
| `gaze_follow` | bool | Enable gaze following |
| `gaze_follow_lag` | int | Allowed lag for gaze following (frames) |
| `gaze_aversion` | bool | Enable gaze aversion detection |
| `aversion_window` | int | Aversion window (frames) |
| `aversion_conf` | float | Confidence threshold for aversion |
| `scanpath` | bool | Enable scanpath analysis |
| `scanpath_dwell` | int | Minimum dwell for scanpath fixation (frames) |
| `gaze_leader` | bool | Enable gaze leadership detection |
| `gaze_leader_tips` | bool | Use ray tips for leadership |
| `gaze_leader_tip_lag` | int | Tip lag for leadership (frames) |
| `attn_span` | bool | Enable attention span tracking |

## 4. CLI Argument Registration

**File:** `phenomena_tracking.py`

```
add_arguments(parser)
```

Registers all phenomena-related flags on the `argparse` parser. This module also re-exports all tracker classes and helper functions for backward compatibility so that external code can import from a single location.

## 5. Helper Functions

**File:** `helpers.py`

### joint_attention

```
joint_attention(persons_gaze, hits, quorum) -> set[int]
```

Returns the set of object indices that are under joint attention (i.e., at least `quorum` participants are gazing at the same object).

### gaze_convergence

```
gaze_convergence(persons_gaze, tip_radius) -> list[tuple[set, ndarray]]
```

Clusters gaze ray tips that fall within `tip_radius` of each other. Returns a list of `(face_set, centroid)` tuples, where `face_set` is the set of participant IDs whose tips converge and `centroid` is the cluster center.

## 6. Tracker Ordering

JA must always be the first tracker in the list because other trackers consume its `confirmed_objs` output.

Dashboard panel assignment determines display layout:

- **Left panel** trackers render in list order: mutual gaze, social referencing, gaze following.
- **Right panel** trackers render in list order: attention span, gaze aversion, scanpath, gaze leadership.

Order within each panel affects the vertical stacking of HUD elements.

## 7. Data Flow

```mermaid
sequenceDiagram
    participant M as MindSight.py
    participant I as init_phenomena_trackers
    participant T as all_trackers list
    participant U as update_phenomena_step
    participant P as tracker.update
    participant C as ctx

    M->>I: cfg
    I-->>T: [JA, mutual_gaze, ...]

    loop every frame
        M->>U: ctx
        U->>U: build tracker_kwargs from ctx
        loop each tracker in all_trackers
            U->>P: tracker.update(**kwargs)
            P-->>C: confirmed_objs, extra_hud, joint_pct (JA only)
        end
    end

    M->>T: post_run_summary(all_trackers, total_frames, pid_map)
```
