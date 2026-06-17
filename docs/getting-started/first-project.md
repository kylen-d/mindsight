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

## Step 4: Run the Project

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

## Step 5: Inspect the Outputs

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
- **Assign participant IDs** -- Create a `project.yaml` in the project root to map video filenames to participant IDs and tag videos with study conditions. The GUI's Project Mode tab can generate this file for you, or you can write it by hand. See [Project Mode](../user-guide/project-mode.md).
