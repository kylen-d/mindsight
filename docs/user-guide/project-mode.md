# Project Mode

## Overview

Project Mode enables batch processing of multiple videos with a shared pipeline configuration and organized output. Instead of running MindSight once per video with individual CLI flags, you define a project directory containing your videos, prompts, and a pipeline YAML config. MindSight processes every video in the project and writes structured output automatically.

## Project Directory Structure

A MindSight project follows this layout:

```
MyProject/
├── Inputs/
│   ├── Videos/         # Drop video files here
│   └── Prompts/        # VP files (.vp.json)
├── Outputs/            # Auto-populated per-video
│   ├── CSV Files/      # Per-video + Global CSVs
│   ├── Videos/
│   ├── heatmaps/
│   └── By Condition/   # Auto-generated per-condition CSVs
├── Pipeline/
│   └── pipeline.yaml   # Project-specific processing config
├── project.yaml        # (optional) Study metadata: conditions, participants, output settings
└── participant_ids.csv  # (optional, legacy) Participant label mappings
```

A ready-to-use example is available at `Projects/ExampleProject/`.

- **Inputs/Videos/** -- Place all video files to be processed here. Supported formats include `.mp4`, `.avi`, `.mov`, and other formats supported by OpenCV.
- **Inputs/Prompts/** -- Store visual prompt `.vp.json` files here if using VP-based detection.
- **Outputs/** -- MindSight creates subdirectories for each output type. Results are organized per-video within each subdirectory. Can be customized to any directory via `project.yaml`.
- **Pipeline/pipeline.yaml** -- The pipeline configuration that applies to every video in the project (detection, gaze, phenomena settings).
- **project.yaml** -- Optional study-level metadata file for condition tags, participant labels, and output directory settings. See [Project YAML](#project-yaml) below.

## Pipeline YAML

The `pipeline.yaml` file defines all processing parameters for the project: detection model, gaze settings, enabled phenomena, output options, and more.

For the full schema reference, see [Pipeline YAML Schema](../reference/pipeline-yaml-schema.md).

A minimal example:

```yaml
detection:
  model: yolo11n.pt

gaze:
  ray_length: 300

phenomena:
  joint_attention: true

output:
  csv: true
  video: true
  heatmap: true
```

## Running

Run a project from the command line:

```bash
python MindSight.py --project Projects/MyProject/
```

MindSight will:

1. Load the pipeline configuration from `Pipeline/pipeline.yaml`.
2. Discover all video files in `Inputs/Videos/`.
3. Process each video sequentially using the shared pipeline config.
4. Write results to the `Outputs/` directory.

## Project YAML

The `project.yaml` file stores study-level metadata that is separate from pipeline processing parameters. It can be created manually or configured through the Project Mode GUI tab. Internally, the file is loaded into `ProjectConfig` and `ProjectOutputConfig` dataclasses, which validate the schema and provide typed access throughout the pipeline.

```yaml
version: 1

# Which pipeline.yaml to use (default: Pipeline/pipeline.yaml)
pipeline: "Pipeline/pipeline.yaml"

# Per-video condition tags (multiple tags per video supported)
conditions:
  session_001.mp4: ["Emotional Story", "Group A"]
  session_002.mp4: ["Non-fiction Story", "Group B"]
  session_003.mp4: ["Emotional Story", "Group B"]

# Per-video participant label mappings (replaces participant_ids.csv)
participants:
  session_001.mp4:
    0: "S70"
    1: "S71"
  session_002.mp4:
    0: "S72"
    1: "S73"

# Output directory customization
output:
  directory: "/path/to/custom/output"   # absolute or relative to project root
```

When `project.yaml` exists, it takes precedence over `participant_ids.csv` for participant labels. If no `project.yaml` is present, MindSight falls back to the legacy `participant_ids.csv` format and default output paths.

## Per-Video Output

Each video processed by the project generates its own set of outputs:

- **Summary CSV** -- A CSV file in `Outputs/CSV Files/` containing aggregated phenomena metrics for that video.
- **Events CSV** -- A per-frame log in `Outputs/CSV Files/` with gaze hit events, joint attention flags, and participant labels.
- **Annotated Video** -- A rendered video in `Outputs/Videos/` with gaze rays, bounding boxes, and phenomena overlays drawn on each frame.
- **Heatmaps** -- Gaze heatmap images in `Outputs/heatmaps/` showing spatial attention distribution across the video duration.

In project mode, per-video CSVs include `video_name` and `conditions` columns so each file is self-describing and ready for aggregation.

## Global CSV Output

After all videos are processed, MindSight automatically generates two global CSV files:

- **Global_Summary.csv** -- Combines all per-video Summary CSVs into one file.
- **Global_Events.csv** -- Combines all per-video Events CSVs into one file.

These files include `video_name` and `conditions` columns, making them immediately usable for cross-video analysis in R, Python, or other tools. Global CSV aggregation is handled by `global_csv.py`.

## Per-Condition CSV Output

When condition tags are defined in `project.yaml`, MindSight automatically splits the Global CSVs by condition into the `By Condition/` folder:

```
Outputs/By Condition/
├── Emotional Story_Summary.csv
├── Emotional Story_Events.csv
├── Non-fiction Story_Summary.csv
├── Non-fiction Story_Events.csv
├── Group A_Summary.csv
├── Group A_Events.csv
├── Group B_Summary.csv
└── Group B_Events.csv
```

A video tagged with multiple conditions (e.g., `["Emotional Story", "Group A"]`) will appear in each matching condition file. In CSV output, multiple condition tags are stored as pipe-delimited values (e.g., `Emotional Story|Group A`).

## Participant Labels

Participant labels can be defined in `project.yaml` (recommended) or via the legacy `participant_ids.csv` file. The `participants` section is loaded internally as `pid_map` -- a per-video mapping of track IDs to participant labels.

### Via project.yaml (recommended)

See the `participants` section in the [Project YAML](#project-yaml) example above.

### Via participant_ids.csv (legacy)

```csv
video_filename,track_id,participant_label
session_001.mp4,0,S70
session_001.mp4,1,S71
session_002.mp4,0,S72
```

Participant IDs then appear in all output CSVs and filenames.

## GUI Configuration

The Project Mode tab in the MindSight GUI provides a full visual interface for configuring all project metadata without manually editing YAML or CSV files. The tab is split into a **Configuration Panel** (left) and a **Monitoring Panel** (right).

### Pipeline Section

- **Pipeline file selector** -- Browse to select which `pipeline.yaml` to use. Defaults to `Pipeline/pipeline.yaml`.
- **Read-only summary** -- Displays the pipeline's key settings at a glance: detection model, confidence threshold, active phenomena, and gaze parameters. Updates automatically when a new pipeline file is selected.
- **Import from Gaze Tab** -- Exports all current settings from the Gaze Tracker tab (detection model, gaze backend and parameters, phenomena toggles and parameters, performance flags) directly into the project's `pipeline.yaml`. This is the recommended way to configure processing parameters: set them up in the Gaze Tracker tab using its full UI, then import them into the project with one click. Prompts before overwriting an existing file.

### Participants Table

An editable table for mapping video filenames to participant track IDs and labels.

- **Video column** -- Dropdown selector populated with all discovered video filenames. You can also type a custom name.
- **Track ID column** -- Integer face track index (0, 1, 2, ...).
- **Label column** -- Custom participant label (e.g., `S70`, `P01`). If left empty, defaults to `P{track_id}`.
- **+ Add Row** -- Adds a single new participant row.
- **- Remove Row** -- Removes selected rows.
- **Auto-populate** -- Creates one default participant row (track_id=0, label=P0) per discovered video. Useful when starting a new project. Prompts before overwriting existing rows.
- **+ Track to All** -- Adds a new participant track to every discovered video at once. Automatically increments the track ID and label based on existing entries (e.g., if videos already have P0, this adds P1 to all of them).

The table is auto-populated when a project is loaded -- from `project.yaml` if it exists, falling back to `participant_ids.csv`, or pre-filled with defaults if neither exists.

### Conditions Table

An editable table for assigning study condition tags to each video. Each video appears as one row with its filename (read-only) and an editable conditions cell.

- **Direct editing** -- Click any cell in the Conditions column and type tags separated by `|` (e.g., `Emotional | Group A`).
- **Apply to Selected** -- Type a tag in the input field, select one or more rows in the Conditions table or the Sources table on the right, then click "Apply to Selected" to add that tag to all selected videos.
- **Remove Tag** -- Type a tag name and click "Remove Tag" to remove that specific tag from all selected rows, leaving other tags intact.
- **Clear All** -- Removes all tags from selected rows.

Changes in the Conditions table are automatically mirrored to the read-only Sources table on the right side.

### Output Settings

- **Output Root** -- Browse to choose a custom output directory. Leave empty to use the default `project/Outputs/` location. Supports both absolute and relative paths.
- **Resolved path** -- Shows the full resolved path so you can verify where files will go.
- **Output preview** -- Displays a summary of what will be generated: the number of per-video CSVs, whether Global CSVs will be created, and how many per-condition CSV files will be produced based on the currently defined condition tags.

### Saving and Running

- **Save project.yaml** -- Persists all GUI configuration (pipeline path, participants, conditions, output settings) to the project directory. The button shows an asterisk (`*`) when unsaved changes exist.
- **Unsaved changes prompt** -- When clicking "Run Project" with unsaved changes, MindSight prompts you to save first.
- **Run Project / Stop** -- Located in the status bar. Runs the full batch processing pipeline using the GUI-configured settings.

### Monitoring Panel (right side)

- **Sources table** -- Read-only table showing all discovered video filenames and their assigned condition tags. Supports multi-row selection for use with the "Apply to Selected" button.
- **Preview** -- Displays live video frames during processing.
- **Progress bar** -- Shows current video number and name during batch processing.
- **Log** -- Scrolling log output with processing messages, errors, and completion status.

## Auxiliary Streams

The `aux_streams` section in the pipeline YAML allows you to specify additional per-participant video feeds (e.g., a scene camera alongside an eye-tracking camera):

```yaml
aux_streams:
  scene_camera:
    P01: aux/P01_scene.mp4
    P02: aux/P02_scene.mp4
```

Auxiliary streams are synchronized with the primary video and can be used by phenomena trackers or included in annotated output.
