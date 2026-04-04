# Quickstart: GUI

This guide walks through MindSight's graphical interface, covering each tab and its key controls.

---

## 1. Launching

```bash
python -m GUI.main_window
```

Requires **PyQt6**. Install it with `pip install PyQt6` if not already present.

---

## 2. Tab 1: Gaze Tracker

The main tracking interface. Configure and run gaze analysis from a single screen.

### Source Selection

Choose your input at the top of the tab:

- **Webcam** -- enter a device index (e.g., `0` for the default camera).
- **Video file** -- browse to an `.mp4`, `.avi`, or other supported format.
- **Image** -- browse to a single image file.

### Detection Mode

- **YOLO** -- enter object class names as a comma-separated list (e.g., `person, knife, cup`).
- **YOLOE** -- select a visual prompt `.vp.json` file for open-vocabulary detection.

### Gaze Backend

Select from the dropdown:

- **MGaze** (default — auto-detects ONNX or PyTorch from model file extension)
- **Gazelle**
- **L2CS**
- **UniGaze**

### Device Selector

Choose the inference device from the device dropdown:

- **Auto** -- automatically selects the best available device.
- **CPU** -- force CPU inference.
- **CUDA** -- use an NVIDIA GPU.
- **MPS** -- use Apple Silicon GPU acceleration.

### Gaze Parameters

Adjust tracking behaviour:

- **Ray length** -- multiplier for the rendered gaze ray.
- **Snap mode** -- adaptive ray snapping to detected objects, with configurable snap distance.
- **Lock-on** -- hold gaze target for a specified number of dwell frames.
- **Gaze cone** -- widen the hit-test angle (degrees).

### Phenomena Panel

Enable phenomena via checkboxes:

- Joint Attention
- Mutual Gaze
- Gaze Following
- Gaze Aversion
- Gaze Leadership
- Social Referencing
- Attention Span
- Scanpath

Each phenomenon exposes its own parameter fields (e.g., temporal window size, threshold) when enabled.

### Plugin Panel

Plugins that define `add_arguments` will have their controls auto-generated here. Adjust plugin-specific settings before starting a run.

### Output Settings

Configure what gets saved:

- **Save video** -- path for the annotated output video.
- **CSV log** -- per-frame event log path.
- **Summary** -- post-run summary CSV path.
- **Heatmap** -- output path for gaze heatmap images.
- **Charts** -- output path for generated charts.

### Presets

Use the **Presets** dropdown to save and load named configurations. This lets you quickly switch between different experimental setups without reconfiguring every parameter.

### Running

Click **Start** to begin processing. Click **Stop** to halt at any time.

- The **live preview** displays annotated frames in real time.
- The **live dashboard** shows real-time gaze statistics, hit counts, and phenomenon events during processing.
- The **log console** at the bottom shows status messages, warnings, and errors.

<!-- screenshot: Gaze Tracker tab -->

---

## 3. Tab 2: VP Builder

Create and test visual prompts for YOLOE open-vocabulary detection.

### Workflow

1. **Add reference images** -- load one or more images that contain the objects you want to detect.
2. **Draw bounding boxes** -- click and drag on the canvas to mark object regions.
3. **Assign classes** -- select a class from the class list for each bounding box.
4. **Save** -- export the prompt as a `.vp.json` file.
5. **Test inference** -- load a YOLOE model and run detection on a test image to verify the prompt works as expected.

<!-- screenshot: VP Builder tab -->

---

## 4. Tab 3: Project Mode

Batch-process multiple videos with a shared pipeline configuration, study condition tags, and participant labels.

### Workflow

1. **Select a project directory** -- Browse to a directory with `Inputs/Videos/` containing your video files.
2. **Configure the pipeline** -- Either browse to an existing `pipeline.yaml`, or click **Import from Gaze Tab** to export your current Gaze Tracker settings directly into the project's pipeline file.
3. **Set up participants** -- Use the Participants table to map video filenames to participant labels. Click **Auto-populate** to quickly generate default entries for all videos, or **+ Track to All** to add additional participants.
4. **Assign conditions** -- In the Conditions table, tag each video with study conditions (e.g., "Emotional Story", "Group A"). Select rows and use **Apply to Selected** to bulk-tag, or edit cells directly.
5. **Save** -- Click **Save project.yaml** to persist your configuration.
6. **Run** -- Click **Run Project** in the status bar. MindSight processes every video and generates per-video CSVs, Global CSVs, and per-condition CSVs automatically.

<!-- screenshot: Project Mode tab -->

---

## 5. Loading and Saving Settings

GUI settings persist between sessions automatically via `settings_manager.py`. When you close and reopen the application, your last-used source, backend, gaze parameters, and output paths are restored.

Pipeline configurations can be loaded and saved independently through the **Pipeline dialog**, accessible from the Project Mode tab or the menu bar. This lets you maintain multiple named configs and switch between them as needed.
