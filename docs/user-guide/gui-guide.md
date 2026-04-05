# GUI Guide

## Launch

Start the MindSight GUI with:

```bash
mindsight-gui                # console command (after pip install -e .)
python MindSight_GUI.py      # or run the wrapper script directly
```

PyQt6 is required. Install it with `pip install PyQt6` if not already available.

## Main Window

The GUI uses a 3-tab layout with a menu bar at the top:

- **Gaze Tracker** -- The primary tab for configuring and running gaze tracking on a single video.
- **VP Builder** -- Create and test visual prompt files for custom object detection.
- **Project Mode** -- Set up and run batch processing across multiple videos.

The menu bar provides access to file operations, settings, and help.

<!-- screenshot: main window -->

## Gaze Tracker Tab

This is the main working area for single-video analysis. It contains the following panels:

<!-- screenshot: gaze tab -->

**Source Panel** -- Select the input video file or webcam feed. A **device selector** dropdown lets you choose between available capture devices (webcams, capture cards).

**Detection Mode** -- Choose between standard YOLO text-class detection or visual-prompt-based YOLOE detection. When VP mode is selected, a file picker for the `.vp.json` file appears.

**Gaze Backend** -- A dropdown selector for the gaze estimation backend (MGaze, L2CS-Net, UniGaze, Gazelle). Backend-specific options appear based on the selection.

**Gaze Parameters** -- Configure gaze ray length, smoothing, and other gaze processing settings.

**Phenomena Panel** -- A list of all available phenomena with a checkbox to enable each one. When a phenomenon is enabled, its configurable parameters (thresholds, durations, etc.) appear as editable fields below the checkbox.

**Plugin Panel** -- Automatically generated from each loaded plugin's `add_arguments` method. Any CLI arguments a plugin exposes are rendered as GUI controls here.

**Output Paths** -- Set output directories for CSV files, annotated videos, and heatmaps.

**Start/Stop Buttons** -- Begin or halt the tracking pipeline.

**Live Preview** -- A video preview panel that shows the annotated output in real time as tracking runs.

**Log Console** -- Displays log messages, warnings, and errors from the pipeline.

**Live Dashboard Charts** -- Real-time matplotlib charts showing tracker metrics (see Live Dashboard section below).

## VP Builder Tab

The VP Builder tab provides a graphical interface for creating visual prompt files.

<!-- screenshot: VP tab -->

- **Reference Image Loading** -- Add one or more reference images using the file browser.
- **Bounding Box Drawing** -- Click and drag on the image canvas to draw bounding boxes around target objects.
- **Class Management** -- Create, rename, and delete object classes. Assign classes to drawn bounding boxes.
- **Save/Test** -- Save the visual prompt as a `.vp.json` file and run a test inference to verify detection quality.

For more detail on the VP workflow, see [Visual Prompts](visual-prompts.md).

## Project Mode Tab

The Project Mode tab has been rebuilt with a pipeline YAML loader, an editable participants table, and integrated live matplotlib dashboard.

<!-- screenshot: project tab -->

The tab is split into two panels: a **Configuration Panel** on the left and a **Monitoring Panel** on the right.

**Configuration Panel (left):**

- **Pipeline** -- Select a `pipeline.yaml` file and see a read-only summary of its settings. Use **Import from Gaze Tab** to export the Gaze Tracker tab's current detection, gaze, and phenomena settings directly into the project's pipeline.
- **Participants** -- Editable table mapping videos to participant track IDs and labels. The Video column provides a dropdown of discovered filenames. Use **Auto-populate** to create default rows for all videos, or **+ Track to All** to bulk-add a new participant across every video.
- **Conditions** -- Assign per-video study condition tags (e.g., "Emotional Story", "Group A"). Tags can be edited directly in cells, or applied/removed in bulk via the action buttons. Multiple tags per video are separated with `|`.
- **Output Settings** -- Choose a custom output root directory. An info label shows what files will be generated (per-video CSVs, Global CSVs, and per-condition CSVs).

**Monitoring Panel (right):**

- **Sources** -- Read-only table of discovered videos with their condition tags.
- **Preview** -- Live video frame display during processing.
- **Progress** -- Per-video progress bar and current file indicator.
- **Log** -- Scrolling processing log.

Click **Save project.yaml** to persist all settings. The button shows `*` when unsaved changes exist.

For more detail on project structure, condition tags, and CSV output, see [Project Mode](project-mode.md).

## Settings Persistence

MindSight uses `settings_manager.py` to save and restore your last-used settings between sessions. When you close the GUI, the current state of all configuration fields is persisted. On the next launch, those values are restored so you can pick up where you left off.

Settings are stored per-user and include source paths, model selections, gaze parameters, enabled phenomena, and output directories.

### Presets

The presets system lets you save and recall named configurations. Use the Presets menu to save the current GUI state under a custom name, or load a previously saved preset to instantly restore all fields. Presets are stored alongside user settings and are available across sessions.

## Pipeline Dialog

The Pipeline Dialog (accessible from the menu bar or toolbar) allows you to:

- **Load** a previously saved `pipeline.yaml` file and apply its settings to the current GUI state.
- **Save** the current GUI configuration as a `pipeline.yaml` file for reuse or for Project Mode.

This is useful for switching between different analysis configurations without manually adjusting every field.

## Live Dashboard

When tracking is active, the Live Dashboard displays real-time matplotlib charts showing key metrics:

- Gaze hit counts per object over time
- Phenomena event timelines
- Per-frame processing statistics

The dashboard updates continuously as frames are processed and provides immediate visual feedback on the tracking session. Charts are rendered with matplotlib and embedded directly in the GUI via the Qt backend.
