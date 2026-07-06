# Quickstart: GUI

This guide walks through MindSight's graphical interface, covering each tab and its key controls.

---

## 1. Launching

```bash
python MindSight_GUI.py
```

Or use the console command: `mindsight-gui`

Requires **PyQt6**. Install it with `pip install PyQt6` if not already present.

The window has four tabs: **Analyze Footage** (the home screen for running
studies), **VP Builder**, **Gaze Tuning**, and **Models**.

---

## 2. Tab 1: Analyze Footage

The home screen for research assistants: open a project, check it is ready, and
batch-process every run with resume support.

### Workflow

1. **Open a project** -- type or paste the project folder path (press Enter or
   click **Open**), click **Browse...**, or pick from the **Recent projects**
   dropdown. Both project layouts are supported: flat `Inputs/Videos/` and
   per-run `Inputs/Runs/<run_id>/` folders.
2. **Read the preflight checklist** -- every time a project opens (or when you
   click **Re-run preflight**), MindSight checks project structure, pipeline
   config validity, model weights, the visual prompt file, discovered runs,
   per-run metadata, participant/condition coverage, compute device, disk
   space, and plugin load errors. Each line shows OK / WARN / FAIL with a fix
   hint. If required weights are missing and downloadable, a one-click
   **Download missing weights** button appears.
3. **Review the runs table** -- one row per run showing its source,
   participants, conditions, ledger status, and the resume **plan**
   ("done → will skip", "will process", "changed → re-run + archive") before
   anything runs.
4. **Run** -- click **Run** in the status bar. Rows update live with progress
   and per-run states (skipped / archived / done / error); the preview pane
   shows annotated frames; **Stop** cancels after the current video finalizes
   cleanly.

### Resume controls

Resume is always on -- finished runs with an unchanged configuration are
skipped automatically. To reprocess:

- **Re-run all** -- confirms, then reprocesses every run (ignores the ledger).
- **Re-run this run** -- right-click a row to invalidate just that run.

### Editing run metadata

Right-click a run and choose **Edit run...** to set its participants
(`0:S70, 1:S71`) and conditions before running. The edit is written where that
project keeps its metadata: `run.yaml` for run-folder projects, `project.yaml`
for flat projects.

### Study setup

The collapsible **Study setup** area holds the study-wide configuration:

- **Pipeline** -- pick the project's `pipeline.yaml`, or click **Import from
  Gaze Tuning** to write your current Gaze Tuning settings into it.
- **Participants / Conditions tables** -- study-wide labels and tags, saved to
  `project.yaml`.
- **Anonymize Footage** -- tick to obscure faces (blur or black) in every
  run's output video. Off by default; when unchecked, runs are exactly what
  they always were.
- **Output root** and **Save project.yaml**.

### Adding a single run

**Add single run...** stages one video manually: pick the file, enter
participants/conditions (or import a `participant_ids.csv`), then either
**Run now** (a one-off run to a directory of your choice, no ledger) or
**Save to project...** (creates a new run folder; copies the video by default,
or moves it if you tick *Move original*).

### Output panel

The bottom-right panel has three tabs:

- **Log** -- status messages from the batch.
- **Charts** -- in-GUI phenomena charts (object look-time, gaze-target
  timeline) rendered from the CSVs each run has already written -- current and
  previous runs both.
- **Output CSVs** -- a read-only viewer over any run's Events / summary /
  stream CSVs.

<!-- screenshot: Analyze Footage tab -->

---

## 3. Tab 2: VP Builder

Create and test visual prompts for YOLOE open-vocabulary detection.

### Workflow

1. **Add reference images** -- load one or more images that contain the objects you want to detect.
2. **Draw bounding boxes** -- click and drag on the canvas to mark object regions.
3. **Assign classes** -- select a class from the class list for each bounding box.
4. **Save** -- export the prompt as a `.vp.json` file.
5. **Test inference** -- load a YOLOE model and run detection on a test image to verify the prompt works as expected.

Click **Use saved VP in Gaze Tuning** to hand the saved prompt straight to the
Gaze Tuning tab's VP field.

<!-- screenshot: VP Builder tab -->

---

## 4. Tab 3: Gaze Tuning

The tuning surface for a single source -- configure the pipeline, watch the
live preview and dashboard, then export the settings for project use.

### Source, detection, and backend

- **Source** -- webcam index, video file, or image.
- **Detection mode** -- YOLO (text classes) or YOLOE visual prompt.
- **Gaze backend** -- **MobileGaze** (default; auto-detects ONNX or PyTorch
  from the model file extension) or **Gaze-LLE**.
- **Device** -- auto, CPU, CUDA, or MPS.

### Parameter panels

The tuning parameters (ray geometry, Gaze-LLE Blend, adaptive snap, smoothing,
fixation lock-on, hit detection, depth, performance, phenomena) are generated
from the configuration schema, so every control maps 1:1 to a CLI flag and
pipeline YAML key. The **Show advanced** toggle reveals the deep-tuning tier
(snap weights, filter cutoffs, fixation thresholds, and similar).

### Plugin panel and outputs

- **Plugin panel** -- controls auto-generated from installed plugins'
  `add_arguments`.
- **Output settings** -- annotated video, CSV log, summary, heatmaps, charts,
  anonymization.

### Presets and pipeline files

Save/load named **presets**, or use **Export Pipeline** / **Import Pipeline**
(File menu and tab buttons) to round-trip the full configuration as a
`pipeline.yaml`.

### Running

Click **Start** to begin processing and **Stop** to halt. The live preview
shows annotated frames; the live dashboard shows real-time gaze statistics,
hit counts, and phenomenon events; the log console reports status.

<!-- screenshot: Gaze Tuning tab -->

---

## 5. Tab 4: Models

A manifest-driven manager for model weights. Every weight MindSight can use is
listed with its backend, whether the *current* configuration needs it, its
on-disk state, and size. From here you can **install** missing weights,
**verify** files against the published checksums, or **re-download** a file
that no longer matches. The preflight checklist on Analyze Footage uses the
same manifest.

<!-- screenshot: Models tab -->

---

## 6. Loading and Saving Settings

GUI settings persist between sessions automatically. When you close and reopen
the application, your last-used source, backend, tuning parameters, and output
paths are restored, and recently opened projects appear in the Analyze Footage
**Recent projects** dropdown.
