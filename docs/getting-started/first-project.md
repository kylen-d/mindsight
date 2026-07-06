# Your First Project

This tutorial walks you through creating and running your first MindSight project from scratch.

## Step 1: Create the Project Directory

Set up the standard project folder structure:

```bash
mkdir -p Projects/MyFirstProject/Inputs/Videos
mkdir -p Projects/MyFirstProject/Inputs/Prompts
mkdir -p Projects/MyFirstProject/Outputs
mkdir -p Projects/MyFirstProject/Pipeline
```

## Step 2: Write a Pipeline Configuration

Create `Projects/MyFirstProject/Pipeline/pipeline.yaml` with a minimal configuration:

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

This configures MindSight to:

- Use the YOLOv11 nano model for object detection.
- Draw gaze rays with a length of 300 pixels.
- Track Joint Attention events.
- Generate CSV, annotated video, and heatmap outputs.

## Step 3: Add Video Files

Copy one or more video files into the `Inputs/Videos/` directory:

```bash
cp ~/recordings/session_001.mp4 Projects/MyFirstProject/Inputs/Videos/
cp ~/recordings/session_002.mp4 Projects/MyFirstProject/Inputs/Videos/
```

The videos should contain people whose gaze you want to track. Standard formats (`.mp4`, `.avi`, `.mov`) are supported.

### Alternative: one folder per run

Instead of a flat `Inputs/Videos/` directory, a project can stage each recording as
its own **run folder** under `Inputs/Runs/`:

```
Inputs/Runs/
├── dyad07_collab/
│   ├── session.mp4          # exactly ONE primary video per folder
│   └── run.yaml             # optional per-run metadata
└── kitchenA_solo/
    └── session.mp4          # a bare folder (no run.yaml) just works
```

The folder name is the **run id** — it names that run's outputs and keys the resume
ledger. The optional `run.yaml` carries the run's metadata:

```yaml
participants: {0: S70, 1: S71}   # track_id -> label
conditions: [collab, kitchenA]   # condition tags (string or list)
date: 2026-07-02                 # recorded in the manifest only
session: dyad-07                 # free-form, manifest only
notes: "camera bumped at ~03:10" # manifest only
extra: {experimenter: KD}        # free-form dict, manifest only
```

All keys are optional. Only `participants` and `conditions` influence processing and
CSV columns; the rest travels into the run's provenance manifest (its `run_meta`
field), so notes stay with the data without changing any CSV format. Metadata
precedence per run is `run.yaml` > `project.yaml` > a root `participant_ids.csv`.
An optional `aux/<type>/` subdirectory per run folder holds that run's auxiliary
streams.

Layout detection is automatic — no switch in `project.yaml`. A project with BOTH
`Inputs/Runs/` and `Inputs/Videos/` populated is ambiguous and refuses to run
(preflight reports it); use one layout per project. The flat layout remains the
default and is unchanged.

Run-folder projects also **mirror their outputs per run**: everything a run produces
(annotated video, Events/summary CSVs, manifest, heatmaps) lands in
`Outputs/Runs/<run_id>/`, named by the run id. Cross-run aggregates (`Global_*.csv`
and `By Condition/`) stay in `Outputs/CSV Files/` in both layouts, so the analyst
hand-off files are identical either way.

In the GUI's **Analyze Footage** tab, *Add single run...* stages a video into a new
run folder for you — copying the file by default (a self-contained, portable
project) or moving it if you tick *Move original*.

## Step 4: Check readiness with preflight

Before a long batch, ask MindSight whether everything is in place:

```bash
python MindSight.py --project Projects/MyFirstProject/ --preflight
```

This prints a checklist — project structure, pipeline config validity, model weights
(with checksums), visual prompt file, discovered runs, per-run metadata,
participant/condition coverage, compute device, disk space, and plugin load errors —
each line `OK`/`WARN`/`FAIL` with a fix hint. Exit code 0 means no failures. The
same checklist appears at the top of the GUI's Analyze Footage tab whenever you open
a project.

## Step 5: Run the Project

Process all videos in the project:

```bash
python MindSight.py --project Projects/MyFirstProject/
```

MindSight will load the pipeline configuration, discover all videos in `Inputs/Videos/`, and process each one sequentially. Progress is logged to the console.

### Resuming an interrupted batch

Project runs **resume by default**. MindSight keeps a per-batch ledger at
`Outputs/_run/ledger.json` recording each video's status. If a batch is interrupted
(a crash, a `kill`, or a machine reboot), just re-run the same command: videos that
already finished with an unchanged configuration are skipped, and processing picks
up where it left off. You will see a line like `[3/30] Skipping session_003.mp4
(done, config unchanged)` for each skipped video.

If you change the pipeline configuration (or the input video), the affected videos
are reprocessed automatically -- their previous outputs are moved into
`Outputs/_run/superseded/<timestamp>_<video>/` first, so nothing is silently
overwritten.

To force a full reprocess regardless of the ledger, pass `--no-resume`:

```bash
python MindSight.py --project Projects/MyFirstProject/ --no-resume
```

`--no-resume` reprocesses every video in place and does **not** archive prior
outputs -- it is the "I know what I'm doing" escape hatch.

## Step 6: Inspect the Outputs

After processing completes, the `Outputs/` directory contains:

- **CSV Files/** -- per video: a tidy scalar summary (`{stem}_summary.csv`), a
  per-frame event log (`{stem}_Events.csv`), any per-stream tables
  (`{stem}_scanpath.csv`, `{stem}_novel_salience_events.csv`, etc.), and a
  provenance manifest (`{stem}_manifest.json`). Across the batch: `Global_*.csv`
  aggregates and a `By Condition/` split for each study condition tag.
- **Videos/** -- Annotated versions of each input video with gaze rays, bounding boxes, and phenomenon overlays rendered on every frame.
- **{stem}_Heatmap/** -- Gaze heatmap images showing where attention was concentrated across each video.
- **_run/** -- Batch bookkeeping: the resume `ledger.json` and any `superseded/`
  outputs from reprocessed videos (see [Resuming an interrupted batch](#resuming-an-interrupted-batch)).

(Run-folder projects gather all of the per-run files above under
`Outputs/Runs/<run_id>/` instead, named by run id; `Global_*.csv`, `By Condition/`,
and `_run/` stay where they are.)

Every CSV is tidy long-format (one typed column per field), so it loads directly into
R or pandas. The `{stem}_manifest.json` records exactly what produced each video's
data -- the full config, a run-identity hash, model-weight hashes, and library
versions -- so results are always traceable. Open a summary file to examine it:

```bash
head -20 Projects/MyFirstProject/Outputs/CSV\ Files/session_001_summary.csv
```

Play an annotated video to visually verify the tracking results.

## Next Steps

Now that you have a working project, try expanding it:

- **Add more phenomena** -- Edit `pipeline.yaml` to enable `mutual_gaze`, `gaze_following`, `attention_span`, or use `all_phenomena: true` to enable everything. See [Phenomena Tracking](../user-guide/phenomena-overview.md).
- **Create visual prompts** -- Build a `.vp.json` file to detect custom objects that standard YOLO classes do not cover. See [Visual Prompts](../user-guide/visual-prompts.md).
- **Customize the pipeline** -- Adjust gaze parameters, detection thresholds, and output settings in `pipeline.yaml`. See [Pipeline YAML Schema](../reference/pipeline-yaml-schema.md).
- **Use the GUI** -- Launch `mindsight-gui` (or `python MindSight_GUI.py`) for a graphical interface to configure and run tracking. See [GUI Guide](../user-guide/gui-guide.md).
- **Assign participant IDs** -- Map track IDs to participant labels and tag videos with study conditions, either per run (`run.yaml`) or study-wide (`project.yaml` / `participant_ids.csv`). The GUI's **Analyze Footage** tab edits both for you: the *Study setup* area writes `project.yaml`, and right-clicking a run offers *Edit run...*.
