# Data Collection Module

## Overview

The `DataCollection/` module is responsible for all output generation in MindSight: CSV summaries, dashboard video overlays, heatmaps, time-series charts, and project-level CSV aggregation. It contains seven files:

| File | Purpose |
|---|---|
| `data_pipeline.py` | Pipeline step coordinator (`collect_frame_data`, `finalize_run`) |
| `csv_output.py` | Summary CSV writer |
| `global_csv.py` | Project-level CSV aggregation and per-condition splitting |
| `dashboard_output.py` | Frame overlay + dashboard compositor |
| `heatmap_output.py` | Per-participant heatmap generation |
| `chart_output.py` | Time-series chart generation |
| `dashboard_matplotlib.py` | Matplotlib-based dashboard rendering for headless/CLI runs |

---

## Data Pipeline

**File:** `data_pipeline.py`

This file coordinates all data collection during and after a run.

### `collect_frame_data(ctx, log_csv, frame_no, hit_events, face_track_ids, persons_gaze)`

Called once per frame. Responsibilities:

- Accumulates the `look_counts` dictionary, mapping `(face_idx, obj_cls)` pairs to frame counts.
- If a `log_csv` writer is provided, writes per-hit rows to the open log CSV.
- In project mode, prepends `video_name` and `conditions` columns to each CSV row (read from `ctx`).
- If `heatmap_path` is set on the context, accumulates gaze endpoint coordinates for later heatmap generation.

### `finalize_run(ctx)`

Called once at the end of a run. Responsibilities:

1. Prints run statistics to the console (total frames processed, hit event count).
2. Writes the summary CSV via `csv_output.write_summary_csv()`, passing `video_name` and `conditions` from context for project mode.
3. Generates heatmaps via `heatmap_output.save_heatmaps()`.
4. Generates charts via `chart_output.generate_run_charts()`.

---

## Summary CSV

**File:** `csv_output.py`

### `resolve_summary_path(summary_arg, source)`

Returns a concrete file path or `None`.

- If `summary_arg` is `True`, an automatic path is derived from `source`.
- If `summary_arg` is a string, it is used as-is.

### `write_summary_csv(path, total_frames, look_counts, all_trackers, pid_map, video_name, conditions)`

Writes the summary CSV with multiple sections:

1. **`object_look_time`** section -- built-in, not contributed by any tracker. Contains per-object gaze duration data derived from `look_counts`.
2. **Tracker sections** -- iterates `all_trackers` and calls `tracker.csv_rows(total_frames, pid_map=pid_map)` on each. Every tracker contributes its own section, separated from the previous one by a blank row.

When `video_name` is not `None` (project mode), every data row is prepended with `video_name` and `conditions` values, and every header row is prepended with the corresponding column names. Comment rows (starting with `#`) are left unchanged.

---

## Global CSV Aggregation

**File:** `global_csv.py`

This module handles project-level CSV aggregation, called after all per-video processing is complete.

### `generate_global_csv(csv_dir, csv_type)`

Combines all per-video CSVs of the given type (`"summary"` or `"events"`) into a single file (`Global_Summary.csv` or `Global_Events.csv`). Comment lines are stripped, and duplicate header rows are deduplicated by comparing against the header from the first file.

Returns the path to the written file, or `None` if no source files were found.

### `generate_condition_csvs(global_csv_path, condition_dir, csv_type)`

Splits a global CSV by the `conditions` column. Each unique tag gets its own CSV. A video with multiple pipe-delimited tags (e.g., `"Emotional|Group A"`) appears in both `Emotional_Summary.csv` and `Group A_Summary.csv`. Tag names are sanitized for filesystem safety.

---

## Dashboard Output

**File:** `dashboard_output.py`

### `draw_overlay(ctx, gaze_cfg)`

Annotates the current frame with visual indicators:

- Gaze rays
- Object bounding boxes
- Joint attention markers
- Lock badges
- Convergence markers
- Dwell arcs

When **lite-overlay mode** is active, expensive visuals are skipped for performance.

### `compose_dashboard(ctx)`

Composes the final display image from the annotated frame and side panels:

- Queries each tracker's `dashboard_data()` method for panel content.
- Trackers declare which side they appear on via the `dashboard_panel` attribute (`"left"` or `"right"`).
- Left and right panels are assembled independently and composited alongside the frame.

### `open_video_writer(save_arg, source, cap)`

Opens a `cv2.VideoWriter` for saving the dashboard output to a video file.

### `apply_face_anonymization(frame, face_bboxes, mode, padding, ...)`

Applies face anonymization to the frame. Supported modes:

- **blur** -- Gaussian blur over face regions.
- **black** -- Solid black rectangles over face regions.

### `AnonSmoother`

Temporal smoothing class for anonymization bounding boxes. Prevents flickering when face detection is intermittent across frames.

### `_draw_panel_section(panel, y, title, colour, rows, line_h)`

Internal helper used by trackers that implement the legacy `dashboard_section()` interface.

---

## Heatmap Output

**File:** `heatmap_output.py`

### `extract_mid_frame(source)`

Extracts a single reference frame from the midpoint of the source video, used as the background for heatmap overlays.

### `save_heatmaps(path, source, bg, heatmap_gaze, pid_map)`

Generates per-participant heatmap images:

1. Takes the accumulated gaze endpoint coordinates from `heatmap_gaze`.
2. Applies Gaussian blur (sigma defined in constants) to produce a density map.
3. Overlays the density map onto the reference frame.
4. Saves one PNG file per participant.

### `resolve_heatmap_path(heatmap_arg, source)`

Returns a concrete directory path or `None`, following the same convention as `resolve_summary_path`.

---

## Chart Output

**File:** `chart_output.py`

### `generate_run_charts(path, all_trackers, total_frames, fps, pid_map, data_plugins)`

Generates time-series charts for the completed run:

1. Iterates all trackers and calls `time_series_data()` on each.
2. Creates matplotlib subplots for each returned metric.
3. Supported chart types: **area**, **step**, **line**.

### `resolve_chart_path(charts_arg, source)`

Returns a concrete directory path or `None`, following the same convention as the other resolve functions.

---

## Matplotlib Dashboard

**File:** `dashboard_matplotlib.py`

Provides a matplotlib-based dashboard renderer used in headless and CLI modes (when a Qt display is unavailable). Queries each tracker's `dashboard_data()` method and renders the panels to a static image that is composited alongside the annotated frame, mirroring the layout of the Qt live dashboard. This module is selected automatically when the GUI is not running.

---

## How Plugins Contribute Data

Phenomena trackers and plugins extend data collection by implementing any combination of the following methods:

| Method | Return type | Used by |
|---|---|---|
| `csv_rows(total_frames, pid_map)` | List of rows | `csv_output.write_summary_csv()` |
| `dashboard_data(pid_map)` | Dict with `title`, `colour`, `rows` | `dashboard_output.compose_dashboard()` |
| `time_series_data()` | Dict of metric name to series data | `chart_output.generate_run_charts()` |
| `console_summary(total_frames, pid_map)` | String | `data_pipeline.finalize_run()` |

Each method is optional. A tracker that only cares about CSV output can implement `csv_rows` alone and ignore the rest.

---

## Extending Data Collection

For fully custom output (e.g. writing to a database, generating a PDF report), subclass `DataCollectionPlugin` and override its hooks. The plugin system will discover and invoke your subclass automatically.

See [Plugin System](plugin-system.md) for registration details and the full plugin lifecycle.
