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

## Step 5: Inspect the Outputs

After processing completes, the `Outputs/` directory contains:

- **CSV Files/** -- One summary CSV per video with per-frame gaze data, object detections, hit events, and Joint Attention occurrences.
- **Videos/** -- Annotated versions of each input video with gaze rays, bounding boxes, and phenomenon overlays rendered on every frame.
- **heatmaps/** -- Gaze heatmap images showing where attention was concentrated across each video.

Open a CSV file to examine the raw data:

```bash
head -20 Projects/MyFirstProject/Outputs/CSV\ Files/session_001.csv
```

Play an annotated video to visually verify the tracking results.

## Next Steps

Now that you have a working project, try expanding it:

- **Add more phenomena** -- Edit `pipeline.yaml` to enable `mutual_gaze`, `gaze_following`, `attention_span`, or use `all_phenomena: true` to enable everything. See [Phenomena Tracking](../user-guide/phenomena-overview.md).
- **Create visual prompts** -- Build a `.vp.json` file to detect custom objects that standard YOLO classes do not cover. See [Visual Prompts](../user-guide/visual-prompts.md).
- **Customize the pipeline** -- Adjust gaze parameters, detection thresholds, and output settings in `pipeline.yaml`. See [Pipeline YAML Schema](../reference/pipeline-yaml-schema.md).
- **Use the GUI** -- Launch `mindsight-gui` (or `python MindSight_GUI.py`) for a graphical interface to configure and run tracking. See [GUI Guide](../user-guide/gui-guide.md).
- **Assign participant IDs** -- Create a `project.yaml` in the project root to map video filenames to participant IDs and tag videos with study conditions. The GUI's Project Mode tab can generate this file for you, or you can write it by hand. See [Project Mode](../user-guide/project-mode.md).
