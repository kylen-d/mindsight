# Data Collection Module

## Overview

The `mindsight/outputs/` module is responsible for all output generation in MindSight: tidy CSV tables, per-run provenance manifests, dashboard video overlays, heatmaps, time-series charts, and project-level CSV aggregation. It contains eight files:

| File | Purpose |
|---|---|
| `data_pipeline.py` | Pipeline step coordinator (`collect_frame_data`, `finalize_run`) |
| `csv_output.py` | Tidy long-format summary + per-stream table writer |
| `global_csv.py` | Project-level CSV aggregation and per-condition splitting |
| `provenance.py` | Per-run `manifest.json` (config dump, run-identity hash, weight hashes, environment) |
| `dashboard_output.py` | Frame overlay + dashboard compositor |
| `heatmap_output.py` | Per-participant heatmap generation |
| `chart_output.py` | Time-series chart generation |
| `dashboard_matplotlib.py` | Matplotlib-based dashboard rendering for headless/CLI runs |

> The per-frame **event CSV** (`--log`) is written by `mindsight/io/writers.py`
> (`open_event_log`), not by this module; its `t_seconds` column is populated by
> `collect_frame_data` below. The resume **ledger** lives in
> `mindsight/project/ledger.py` (see [Outputs reference](#outputs-reference)).

---

## Data Pipeline

**File:** `data_pipeline.py`

This file coordinates all data collection during and after a run.

### `collect_frame_data(ctx, *, log_csv, frame_no, hit_events, face_track_ids, persons_gaze, ...)`

Called once per frame. Every argument after `ctx` is **keyword-only**.
Responsibilities:

- Accumulates the `look_counts` dictionary, mapping `(face_idx, obj_cls)` pairs to frame counts.
- If a `log_csv` writer is provided, writes per-hit rows to the open event CSV. Each row carries a `t_seconds` column immediately after `frame`, computed as `frame_no / video_fps` (formatted to 3 decimal places; empty string when the source has no frame rate, e.g. a webcam).
- In project mode, prepends `video_name` and `conditions` columns to each CSV row (read from `ctx`). The `t_seconds` column stays in the core position, after the project prefix.
- If `heatmap_path` is set on the context, accumulates gaze endpoint coordinates for later heatmap generation.

### `finalize_run(ctx)`

Called once at the end of a run. Responsibilities:

1. Prints run statistics to the console (total frames processed, hit event count).
2. Writes the tidy summary + per-stream tables via `csv_output.write_summary_tables()`, passing `video_name`, `conditions`, and the real video frame rate (`ctx['video_fps']`) for seconds conversions and project-mode columns.
3. Generates heatmaps via `heatmap_output.save_heatmaps()`.
4. Generates charts via `chart_output.generate_run_charts()`, using the true `ctx['video_fps']` for the time axis.

The per-run provenance `manifest.json` is **not** written here -- it is written by
the orchestration layer (CLI `main`, `project/runner.run_project`, the GUI
workers) after the run returns, so a mid-run cancellation never leaves a partial
manifest. See [Provenance](#provenance).

---

## Summary tables (tidy long-format)

**File:** `csv_output.py`

MindSight writes **tidy, R-friendly** CSVs: one long-format scalar table plus one
typed file per event/timeseries stream. The old multi-section `#`-header summary
CSV (heterogeneous per-tracker blocks in a single file) is gone -- every column is
now typed, and project aggregation is a pure concatenation.

### `resolve_summary_path(summary_arg, source)`

Returns a concrete file path or `None` for the scalar table.

- If `summary_arg` is `True`, an automatic path is derived from `source` (`{stem}_summary.csv`).
- If `summary_arg` is a string, it is used as-is.

Per-stream files are written next to it as `{stem}_<stream>.csv`.

### `write_summary_tables(path, total_frames, fps, look_counts, all_trackers, pid_map, video_name, conditions)`

Writes the scalar summary file and any non-empty stream files.

**Scalar file `{stem}_summary.csv`** -- one long-format row per metric, header
identical in single and project mode (`video_name`/`conditions` are empty strings
in single mode, filled in project mode, so Global concatenation is trivial):

```csv
video_name,conditions,phenomenon,participant,partner,object,metric,value
trimmed,,object_look_time,P0,,dining table,frames_active,357
trimmed,,object_look_time,P0,,dining table,seconds_active,11.942
trimmed,,object_look_time,P0,,dining table,pct_of_video,41.0817
trimmed,,joint_attention,all,,,frames_active,143
trimmed,,gaze_following,P0,P1,,event_count,6
trimmed,,gaze_following,P0,P1,,mean_lag_frames,15.7
trimmed,,gaze_following,P0,P1,,mean_lag_seconds,0.525
```

Rows are sorted deterministically by `(phenomenon, participant, partner, object, metric)`.
Object look-time rows are emitted by the writer itself (it is not a tracker);
every other scalar row comes from a tracker's `summary_metrics()` hook. Units live
in the metric name (`*_frames`, `*_seconds`, `*_pct`); seconds values are formatted
to 3 decimal places, percentages to 4.

**`phenomenon` labels are prettified.** The value is the tracker's `summary_label`
attribute, which defaults to its `name` but is overridden by four terse trackers so
the analyst-facing labels read cleanly (the `name` attrs themselves are unchanged --
they key registries, flags, and dashboards):

| Tracker `name` | `summary_label` |
|---|---|
| `gaze_follow` | `gaze_following` |
| `gaze_leader` | `gaze_leadership` |
| `social_ref` | `social_referencing` |
| `attn_span` | `attention_span` |

All other labels (`joint_attention`, `gaze_aversion`, `scanpath`, `mutual_gaze`,
`novel_salience`, `eye_movement`, `pupillometry`, `object_look_time`) equal their
tracker name.

**Stream files** -- each tracker's `summary_tables()` hook returns
`{table_name: (header, rows)}`; each table becomes `{stem}_<table_name>.csv` (only
written when it has data), with `video_name,conditions` prepended by the writer. The
built-in streams and their typed headers:

| File | Contents |
|---|---|
| `{stem}_scanpath.csv` | One row per fixation (`participant,fixation_index,object`) |
| `{stem}_novel_salience_events.csv` | Per-saccade events (`frame,t_seconds,participant,direction,speed_px,speed_deg,delta_x,delta_y`) |
| `{stem}_eye_movement_events.csv` | Fixation/saccade events (`participant,event_type,start_frame,end_frame,start_seconds,end_seconds,duration_seconds,peak_velocity`) |
| `{stem}_pupillometry_timeseries.csv` | Per-frame pupil ratios (`frame,t_seconds,participant,pupil_iris_ratio,dilation_pct,valid,...`) |
| `{stem}_pupillometry_blinks.csv` | Detected blink events |

Per-participant summary statistics of the stream plugins (eye-movement means,
pupillometry baseline/mean ratios) fold into `{stem}_summary.csv` as scalar metrics.

**Merged episode stream `{stem}_phenomena_events.csv`.** In addition to the
per-tracker stream files above, `write_summary_tables` collects every tracker's
`episode_rows()` (`csv_output.py:151-166`) and merges them into a single
`{stem}_phenomena_events.csv` (columns `phenomenon,participant,partner,object,
frame_start,frame_end`, with `video_name,conditions` prepended and rows sorted by
`frame_start`). This is the tidy episode log for glances, aversion streaks,
mutual-gaze pairs, and the like.

**Legacy `csv_rows` passthrough.** A third-party plugin that overrides only the
legacy `csv_rows` hook (and neither tidy hook) still produces output: the writer
dumps its rows verbatim to `{stem}_plugin_{name}.csv`. See
[Plugin Base Classes](../reference/plugin-base-classes.md) and
[Writing a Plugin](writing-a-plugin.md).

---

## Global CSV Aggregation

**File:** `global_csv.py`

This module handles project-level CSV aggregation, called after all per-video processing is complete. Because every per-video table is now tidy long-format with an identical header across videos, aggregation is a pure header-once concatenation.

The `GLOBAL_TABLES` registry maps each per-video filename suffix to its global
output name, so every table type -- the scalar summary, the frame-level event log,
and every stream -- is aggregated:

| Per-video suffix | Global file |
|---|---|
| `_summary.csv` | `Global_summary.csv` |
| `_Events.csv` | `Global_Events.csv` |
| `_phenomena_events.csv` | `Global_phenomena_events.csv` |
| `_scanpath.csv` | `Global_scanpath.csv` |
| `_novel_salience_events.csv` | `Global_novel_salience_events.csv` |
| `_eye_movement_events.csv` | `Global_eye_movement_events.csv` |
| `_pupillometry_timeseries.csv` | `Global_pupillometry_timeseries.csv` |
| `_pupillometry_blinks.csv` | `Global_pupillometry_blinks.csv` |

### `generate_global_csv(csv_dir, suffix, out_name)`

Concatenates all per-video files matching `suffix` into `out_name`. The header is
written once; subsequent duplicate header rows are skipped. `Global_*` files are
excluded from their own inputs.

Returns the path to the written file, or `None` if no source files were found.

### `generate_condition_csvs(global_path, condition_dir, suffix)`

Splits a global CSV by the `conditions` column (column index 1). Each unique tag
gets its own file (`{tag}{suffix}`). A video with multiple pipe-delimited tags
(e.g., `"Emotional|Group A"`) appears under both `Emotional_summary.csv` and
`Group A_summary.csv`. Tag names are sanitized for filesystem safety.

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
| `summary_metrics(total_frames, fps, pid_map)` | List of scalar-metric dicts | `csv_output.write_summary_tables()` (scalar file) |
| `summary_tables(total_frames, fps, pid_map)` | Dict of `table_name -> (header, rows)` | `csv_output.write_summary_tables()` (stream files) |
| `episode_rows(total_frames, fps, pid_map)` | List of episode dicts | `csv_output.write_summary_tables()` (merged `{stem}_phenomena_events.csv`) |
| `finalize(frame_no)` | `None` | Pipeline run-end hook — closes in-flight episodes before summaries are written |
| `csv_rows(total_frames, pid_map)` | List of rows | *Legacy passthrough* -> `{stem}_plugin_{name}.csv` |
| `dashboard_data(pid_map)` | Dict with `title`, `colour`, `rows` | `dashboard_output.compose_dashboard()` |
| `time_series_data()` | Dict of metric name to series data | `chart_output.generate_run_charts()` |
| `console_summary(total_frames, pid_map)` | String | `data_pipeline.finalize_run()` |

Each method is optional. `finalize()` runs before the summary hooks so any
open-ended episode is closed and appears in `episode_rows()`. Scalar metrics go through `summary_metrics`; per-event or
per-frame streams go through `summary_tables`. `csv_rows` is retained only for
backward compatibility with third-party plugins written against the old paper
contract -- a plugin overriding only `csv_rows` gets its rows dumped verbatim to a
dedicated `{stem}_plugin_{name}.csv` file. New trackers should implement the tidy
hooks. See [Plugin Base Classes](../reference/plugin-base-classes.md).

---

## Provenance

**File:** `provenance.py`

Every run that writes at least one file output also writes a per-run
`manifest.json` capturing exactly what produced the data, so a stored CSV can
always be traced back to its config and model weights.

### `write_run_manifest(path, *, ns, config, source, output_paths, started, finished, status, error=None, meta=None)`

The optional `meta` argument carries per-run staging provenance (from a project
`run.yaml`), folded into the manifest when present.


Builds and atomically writes the manifest (temp file + `os.replace`). Called by the
orchestration layer, never by the pipeline generator. The manifest records:

- `config` -- the full `PipelineConfig` dump plus its `config_canonical_hash`.
- `run_identity` -- a sha256 over the processing config (minus output/project
  sections) + model-wiring inputs (`model`/`vp`/`classes`/`blacklist`/`device`) +
  plugin flag values + loaded-weight sha256s + `mindsight.__version__`. Two runs
  that would produce identical numeric output share a `run_identity`; changing the
  output path does **not** change it (this is the hash the resume ledger compares).
- `environment` -- Python, platform, and the versions of torch, ultralytics,
  onnxruntime, cv2, numpy, and mediapipe (`"absent"` when not importable).
- `weights` -- per-backend resolved weight paths with `size`, `mtime`, and sha256
  (or `"missing"` for auto-download names not present locally). Hashes are cached
  per `(path, size, mtime)` so a 30-video batch hashes each weight once.
- `source` -- the input file identity (size, mtime, sha256).
- `outputs`, `timestamps`, `status` (`done` / `error`), and any `error` string.

**Location** (per the Q4 ruling -- provenance travels with the data):

- Project mode: `Outputs/CSV Files/{stem}_manifest.json` (one per video; the ledger
  stores the path; the error branch writes `status: "error"`).
- Single-source mode: `{stem}_manifest.json` next to whichever CSV output is
  configured (summary preferred, else log, else beside `--save`). Pure display runs
  (no file output) write no manifest.

---

## Outputs reference

Everything a run can produce, and where it lands. In project mode all paths are
under the project's `Outputs/`; in single-source mode they follow the `--save` /
`--log` / `--summary` / `--heatmap` / `--charts` arguments.

**Per video:**

| Artifact | Path (project mode) | Written by |
|---|---|---|
| Annotated video | `Outputs/Videos/{stem}_Video_Output.mp4` | `io/writers.py` + `dashboard_output.py` |
| Per-frame event log | `Outputs/CSV Files/{stem}_Events.csv` | `io/writers.py` (`t_seconds` via `collect_frame_data`) |
| Scalar summary table | `Outputs/CSV Files/{stem}_summary.csv` | `csv_output.write_summary_tables()` |
| Per-stream tables | `Outputs/CSV Files/{stem}_<stream>.csv` | `csv_output.write_summary_tables()` |
| Merged episode log | `Outputs/CSV Files/{stem}_phenomena_events.csv` | `csv_output.write_summary_tables()` (each tracker's `episode_rows()`) |
| Legacy plugin passthrough | `Outputs/CSV Files/{stem}_plugin_{name}.csv` | `csv_output` (only if a plugin overrides only `csv_rows`) |
| Provenance manifest | `Outputs/CSV Files/{stem}_manifest.json` | `provenance.write_run_manifest()` |
| Heatmaps | `Outputs/{stem}_Heatmap/` | `heatmap_output.py` |
| Charts | chart directory (when `--charts`) | `chart_output.py` |

**Per project (aggregates):**

| Artifact | Path | Written by |
|---|---|---|
| Global tables | `Outputs/CSV Files/Global_*.csv` | `global_csv.generate_global_csv()` |
| Per-condition splits | `Outputs/CSV Files/By Condition/{tag}_*.csv` | `global_csv.generate_condition_csvs()` |

**Run bookkeeping (project mode):**

| Artifact | Path | Written by |
|---|---|---|
| Resume ledger | `Outputs/_run/ledger.json` | `mindsight/project/ledger.py` |
| Superseded outputs | `Outputs/_run/superseded/<UTC stamp>_<stem>/` | `ledger.archive()` on config change |

The ledger tracks each video's status (`in_progress` / `done` / `error`), its
`config_hash` (= the manifest `run_identity`) and a per-video `video_hash`. On a
resumed batch, `done` videos with matching hashes are skipped; a config change
archives the prior outputs to `_run/superseded/` before reprocessing. See
[Your First Project](../getting-started/first-project.md) for the resume workflow
and `--no-resume`.

---

## Extending Data Collection

For fully custom output (e.g. writing to a database, generating a PDF report), subclass `DataCollectionPlugin` and override its hooks. The plugin system will discover and invoke your subclass automatically.

See [Plugin System](plugin-system.md) for registration details and the full plugin lifecycle.
