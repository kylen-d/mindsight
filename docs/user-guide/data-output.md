# Data & Output

## Overview

MindSight can produce several types of output for research data collection:

- **Annotated video** with gaze overlays and bounding boxes
- **Per-frame CSV logs** recording every gaze-object hit
- **Post-run summary CSV** with aggregated metrics
- **Per-participant heatmaps** showing gaze distribution
- **Time-series charts** for phenomena metrics over time

All outputs are optional and enabled via CLI flags. By default, MindSight only displays the live preview window.

---

## Annotated Video Output

Use `--save` to write an annotated video to disk.

```bash
# Auto-named: Outputs/Video/[stem]_Video_Output.mp4
python MindSight.py --save video.mp4

# Custom path
python MindSight.py --save /path/to/output.mp4
```

Omit the path argument to use the automatic naming convention, which places the file under `Outputs/Video/` using the input filename stem.

The annotated video includes:

- Gaze rays drawn from each participant's face
- Object detection bounding boxes with class labels
- Joint attention markers when shared gaze is detected
- Dashboard panels (unless disabled with `--no-dashboard`)

---

## Per-Frame Event CSV

Use `--log` to write one row per gaze-object hit per frame.

```bash
python MindSight.py --log events.csv
```

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `video_name` | str | Source video filename stem (project mode only) |
| `conditions` | str | Pipe-delimited condition tags (project mode only) |
| `frame` | int | Frame number (0-indexed) |
| `face_idx` | int | Detected face index |
| `object` | str | Object class name |
| `object_conf` | float | Detection confidence score |
| `bbox_x1` | float | Bounding box left |
| `bbox_y1` | float | Bounding box top |
| `bbox_x2` | float | Bounding box right |
| `bbox_y2` | float | Bounding box bottom |
| `joint_attention` | int | 1 if raw joint attention detected this frame |
| `joint_attention_confirmed` | int | 1 if temporally confirmed joint attention |
| `participant_label` | str | Custom label from pid_map, or default P0, P1... |

The `video_name` and `conditions` columns are only present when running in project mode. In single-video mode, these columns are omitted.

---

## Post-Run Summary CSV

Use `--summary` to generate an aggregated summary after processing completes.

```bash
# Auto-named: Outputs/CSV Files/[stem]_Summary_Output.csv
python MindSight.py --summary

# Custom path
python MindSight.py --summary results.csv
```

The summary CSV contains three core sections in one file, plus additional rows from any active phenomena trackers.

### Core Sections

| Section | Columns | Description |
|---------|---------|-------------|
| `joint_attention` | category, participant, object, frames_active, total_frames, value_pct | Percentage of frames with a shared gaze target |
| `cosine_similarity` | category, participant, object, frames_active, total_frames, value_pct | Per-pair and overall mean cosine similarity of gaze directions |
| `object_look_time` | category, participant, object, frames_active, total_frames, value_pct | Per-(participant, object-class) frame count and percentage |

Additionally, each active phenomena tracker contributes its own rows via its `csv_rows()` method, using the same column structure.

In project mode, each data row is prepended with `video_name` and `conditions` columns, making the summary CSV self-describing and ready for cross-video aggregation.

---

## Heatmaps

Use `--heatmap` to generate per-participant gaze heatmaps.

```bash
python MindSight.py --heatmap
```

Heatmap generation works as follows:

- Gaze ray endpoints are accumulated across all frames for each participant
- A Gaussian blur (sigma=40) is applied for smooth visualization
- The heatmap is overlaid on a reference frame sampled from the video
- One PNG is saved per participant: `[stem]_Heatmap_P0.png`, `[stem]_Heatmap_P1.png`, etc.
- Default output location: `Outputs/heatmaps/`

---

## Time-Series Charts

Use `--charts` to generate matplotlib charts showing phenomena metrics over time.

```bash
python MindSight.py --charts
```

- Each phenomena tracker that implements `time_series_data()` contributes a subplot
- The x-axis represents frame number; the y-axis represents the metric value
- Charts are rendered with the matplotlib-based dashboard renderer
- Default output path: `Outputs/Charts/[stem]_Charts.png`

Post-run chart generation is triggered by the `--charts` flag and runs after all frames have been processed.

---

## On-Screen Dashboard

The live display includes a side dashboard panel with real-time metrics, rendered using the matplotlib-based dashboard renderer:

- **Top-left:** FPS counter and gaze-object hit count
- **Top-right:** Per-pair cosine similarity (instantaneous and running average)
- **Bottom-right:** Joint attention percentage and currently attended objects
- **Phenomena panels:** Each active tracker contributes additional panels via `dashboard_data()`

To disable the dashboard for maximum throughput:

```bash
python MindSight.py --no-dashboard
```

---

## Face Anonymization

Use `--anonymize` to obscure detected faces in the output video.

```bash
# Gaussian blur over face regions
python MindSight.py --anonymize blur --save output.mp4

# Solid black rectangle over face regions
python MindSight.py --anonymize black --save output.mp4
```

The `--anonymize-padding` flag controls how much margin is added around the detected face bounding box. The default value is `0.3`, meaning 30% of the bounding box size is added on each side.

```bash
python MindSight.py --anonymize blur --anonymize-padding 0.5
```

---

## Participant ID Mapping

By default, participants are labeled P0, P1, P2, etc. based on detection order. Custom labels can be assigned in two ways.

### Inline Mapping

Use `--participant-ids` with a comma-separated list. Labels are assigned positionally.

```bash
python MindSight.py --participant-ids S70,S71,S72
```

### CSV File Mapping

Use `--participant-csv` with a CSV file that maps video filenames to participant labels.

```bash
python MindSight.py --participant-csv ids.csv
```

Custom labels appear in CSV output, the on-screen dashboard, and console messages.

---

## Output Directory Structure

When using auto-naming (no explicit path), MindSight organizes outputs under an `Outputs/` directory:

```
Outputs/
├── CSV Files/
│   └── [stem]_Summary_Output.csv
├── Video/
│   └── [stem]_Video_Output.mp4
├── heatmaps/
│   └── [stem]_Heatmap_P0.png
└── Charts/
    └── [stem]_Charts.png
```

Where `[stem]` is the input video filename without its extension.

### Project Mode Output Structure

In project mode, outputs are organized per-video within the output root, plus global and per-condition aggregations:

```
Outputs/
├── CSV Files/
│   ├── session_001_Summary.csv       # per-video
│   ├── session_001_Events.csv
│   ├── session_002_Summary.csv
│   ├── session_002_Events.csv
│   ├── Global_Summary.csv            # all videos combined
│   └── Global_Events.csv
├── By Condition/                     # auto-split by condition tag
│   ├── Emotional Story_Summary.csv
│   ├── Emotional Story_Events.csv
│   ├── Group A_Summary.csv
│   └── Group A_Events.csv
├── Videos/
│   ├── session_001_Video_Output.mp4
│   └── session_002_Video_Output.mp4
└── session_001_Heatmap/
    └── ...
```

The **Global CSVs** are produced by `global_csv.py`, which concatenates all per-video CSVs into a single file with one header row. The **Per-Condition CSVs** are auto-generated when condition tags are defined, filtering the global CSV by each unique tag. A video with multiple tags (e.g., `Emotional Story | Group A`) appears in every matching condition file.

The output root directory can be customized via `project.yaml` or the GUI's Output Settings. See [Project Mode](project-mode.md) for details.

---

## Parameter Reference

| Flag | Argument | Description |
|------|----------|-------------|
| `--save` | `[path]` | Save annotated video. Omit path for auto-naming. |
| `--log` | `path` | Write per-frame event CSV. |
| `--summary` | `[path]` | Write post-run summary CSV. Omit path for auto-naming. |
| `--heatmap` | -- | Generate per-participant gaze heatmaps. |
| `--charts` | -- | Generate time-series charts for phenomena metrics. |
| `--no-dashboard` | -- | Disable on-screen dashboard overlay. |
| `--anonymize` | `blur` or `black` | Anonymize faces in output video. |
| `--anonymize-padding` | `float` | Padding fraction around face bbox (default: 0.3). |
| `--participant-ids` | `label,label,...` | Comma-separated participant labels (positional). |
| `--participant-csv` | `path` | CSV file mapping filenames to participant labels. |
