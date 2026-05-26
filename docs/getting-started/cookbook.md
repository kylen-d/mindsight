# MindSight Cookbook

A practical guide for getting MindSight running on lab footage, with parameter recommendations drawn from my own testing on KITCO 3 video. For background on *why* any of this works, see the MindSight paper; this doc is the "what to actually click" companion. Parameter names below match the **GUI labels** -- if you're using the CLI or a YAML pipeline file, the equivalent flag/key may differ slightly.

> **Terminology note:** This guide uses the paper's names -- **Gaze-LLE** and **MobileGaze** -- for the two model families it discusses. The current GUI labels them as **"Gazelle"** and **"MGaze"** respectively, and the same shorter names appear in CLI flags (`--mgaze-model`, `--gazelle-model`), directory paths (`Weights/Gazelle/`, `Weights/MGaze/`), and model filenames (`gazelle_dinov2_vitb14.pt`). These are the same thing; treat the two name forms as interchangeable.

## Overview

MindSight predicts and analyses visual attention from standard monocular video. It takes a video file as input, runs gaze estimation and object detection in parallel, and post-processes the combined outputs to produce structured attention data analogous to manual video coding -- but in minutes rather than hours, and without inter-rater variability.

The system is built around a four-stage pipeline: **Detection → Gaze Estimation → Ray Forming & Intersection → Phenomena & Data Collection**. Most of the tuning effort in this cookbook lives in stages 2 and 3, since those are where the system goes from raw model output to "Person A is looking at Object B."

---

## Quick Start

### 1. Install the program and its dependencies

Follow the installation steps in the [README](../../README.md). The short version:

```bash
git clone https://github.com/kylen-d/mindsight.git
cd mindsight
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Install the desired models

#### Gaze backends

In my testing, **MobileGaze with the ResNet50 architecture** performed the best as a per-face backend. The ResNet50 weights are heavier and benefit from reasonably modern hardware; use the `.pt` weights for CUDA devices (NVIDIA GPUs) and the ONNX weights for Apple Silicon or other non-NVIDIA hardware (ONNX runs on CoreML/CPU as a fallback).

**Installing Gaze-LLE is highly recommended** as the secondary model for periodic ray forming. The combination -- a per-face backend running every frame plus Gaze-LLE running every N frames for spatial correction -- is what the GUI calls *Gazelle Blend (Ray Forming)*, and what the paper formalises as **Gaze-LLE Blend**. It is by a wide margin the most accurate ray-forming mode currently available in the system (see Sections 7.3 and 11.3 of the paper).

#### Object detection

**YOLOE models** are recommended due to their support for visual prompting (which lets you define study-specific object classes without retraining). They can be downloaded automatically by Ultralytics on first use. In my testing, the leaner `yoloe-v8-seg` models worked better than the newer `v26` variants on KITCO footage, and the medium and large weights (`yoloe-v8m-seg` / `yoloe-v8l-seg`) hit the best accuracy/speed tradeoff.

Place all weights in `Weights/` under the appropriate backend subdirectory (e.g. `Weights/MGaze/`, `Weights/Gazelle/`, `Weights/YOLO/`).

---

## Recommended Parameters

A default YAML pipeline file lives at `[path to your lab's preset]`. The breakdown below explains what each section does, organised in the same order as the GUI's **Gaze Tracker** tab.

### Object Detection

Switch the mode radio to **YOLOE Visual Prompt** for any study with custom object classes.

| GUI field | Value | Notes |
|---|---|---|
| **YOLOE model** | `yoloe-v8l-seg.pt` | Paired with a study-specific `.vp.json` |
| **VP file** | your `.vp.json` | Built in the VP Builder tab (see below) |
| **Conf** | `0.20--0.30` | With Merge Overlaps on, lower is generally better until non-objects start slipping through |

**Merge Overlaps**

| GUI field | Value | Notes |
|---|---|---|
| **Enable** | checked | |
| **Strategy** | `filter` or `dynamic` | `filter` keeps the highest-confidence box per cluster; `dynamic` chooses per-cluster based on confidence and size |
| **Threshold** | `0.50--0.60` | Lower than the GUI default (0.70) -- tuned for visual-prompt detection, where you want tighter clustering |

### Gaze Backend

- **Backend:** MobileGaze (GUI radio: *MGaze*)
- **Model:** path to `resnet50_gaze.pt` (CUDA) or `resnet50_gaze.onnx` (Apple Silicon / CPU)
- **Arch:** `resnet50` (only shown when the model path ends in `.pt`)

### Gaze Ray Geometry

| GUI field | Value | Notes |
|---|---|---|
| **Ray length** | `1.25--1.50` | Default starting point; with Gaze-LLE Blend on, length is dynamically corrected anyway |
| **Gaze cone** | `~5.0°` | Study-dependent. Standard precision/recall tradeoff -- a tight cone gave me the cleanest hits |
| **Forward threshold (°)** | `5.0` (default) | Pitch/yaw below this are treated as "looking at camera"; leave default unless you have a reason |

### Gaze-LLE Blend (GUI group: *Gazelle Blend (Ray Forming)*)

This is the single most impactful subsystem in the program. A per-face backend produces smooth, temporally consistent pitch/yaw vectors but knows nothing about scene context; Gaze-LLE produces scene-aware heatmaps but jumps between regions instead of tracking smooth pursuit. Blending them gives you the strengths of both at a fraction of the cost of running pure Gaze-LLE every frame.

| GUI field | Value | Notes |
|---|---|---|
| **Model** | path to `gazelle_dinov2_vitb14.pt` | The standard checkpoint |
| **Variant** | `gazelle_dinov2_vitb14` | The `_inout` variants add an "in-frame" confidence score that can suppress noisy heatmaps |
| **Min call gap** | `10--30` frames | Minimum frames between Gaze-LLE inference calls (the GUI field formerly labelled *Inference interval*). Lower = more frequent corrections at the cost of throughput. See **Belief Map Tuning** below |

**Belief Map Tuning** (3 default-visible + 4 Advanced knobs)

The redesigned blender uses a fixation-aware scheduler to decide *when* Gaze-LLE inferences are applied to each participant, and a One Euro Filter to adaptively smooth the output direction and length. Gaze-LLE fires only while a participant is genuinely fixating -- during smooth pursuit or a head turn the ray tracks pitch/yaw directly, so mid-motion inferences no longer drag the ray onto a stale region. The default-visible knobs are what you tune per study; the Advanced group is for fine adjustment.

*Default-visible:*

| GUI field | Value | What it controls |
|---|---|---|
| **Min call gap (frames)** | `10--30` | Minimum frames between Gaze-LLE inference calls. Lower = more scene corrections per second (higher GPU cost). The scheduler additionally requires at least one participant to be fixating; if nobody is, no inference fires regardless. |
| **Direction responsiveness** | `0.5` (default) | One Euro beta for direction. Higher = direction snaps to fast head/eye motion; lower = smoother, more latency. |
| **Length responsiveness** | `0.3` (default) | One Euro beta for length. Kept lower than direction by default -- length should hold more steadily during smooth motion. |

*Advanced (collapsible):*

| GUI field | Value | What it controls |
|---|---|---|
| **Fixation v-threshold (rad/frame)** | `0.02` | Smoothed pitch/yaw velocity at which a face is treated as 50% fixating. Lower = safer anchoring; higher = more inferences accepted. |
| **Fixation d-threshold (rad)** | `0.10` | Windowed pitch/yaw dispersion at 50% fixation likelihood. Lower = tighter fixation criterion. |
| **Direction min-cutoff (Hz)** | `1.0` | Direction smoother floor cutoff. Lower = smoother at rest. |
| **Length min-cutoff (Hz)** | `1.0` | Length smoother floor cutoff. |

### Migrating from the legacy Gazelle Blend

If you have pipelines or notes referencing the pre-redesign knobs, these are removed and have no 1:1 replacement:

- `direction_blend`, `length_blend`, `length_only`, `direction_decay`, `length_decay`, `diffusion_sigma`, `blend_conf_scale`, `belief_min_peak`, `inout_threshold`

Practical migration:

1. Start from the new defaults -- they encode the intended behaviour (direction defers to pitch/yaw quickly out of fixation; length holds through a fixation).
2. If you previously lowered `length_decay` to make ray reach persist, that persistence is now automatic -- length holds while a participant fixates and collapses when they leave. Leave **Length responsiveness** at its default.
3. If you previously raised `direction_decay` for stable fixations, that is also automatic now -- keep **Direction responsiveness** at its default and lower **Direction min-cutoff** (Advanced) if you want stronger jitter suppression at rest.
4. `rf_gazelle_interval` still parses; it now populates `min_call_gap`.

### Adaptive Snap

The legacy ray-forming mode -- scores nearby objects against the gaze vector via a weighted sum, then snaps or extends the ray to the winner. It works, but requires a *ton* of per-study tuning, falls apart in object-sparse scenes, and is generally outclassed by Gaze-LLE Blend. I'd leave the **Adaptive Snap** group unchecked unless you have a specific reason not to use Gaze-LLE.

### Ray Forming Smoothing

| GUI field | Value | Notes |
|---|---|---|
| **Smooth targets** | `All` | Smooths all ray endpoint transitions |
| **Smooth alpha** | `0.40--0.60` | EMA rate -- lower values smooth more heavily, but very low values (near 0) cause the endpoint to lag indefinitely. This range gives meaningful smoothing without freezing the ray |

### Hit Detection

If you're using Gaze-LLE Blend, set both **Hit conf gate** and **Extend detection (px)** to `0` -- the blend already places the ray endpoint on the right region, so no extra extension is needed. If you find you have too many noisy detections, bump **Hit conf gate** up. **Extend detection** is mostly useful for debugging.

### Depth Estimation

Work-in-progress. Leave the **Depth Estimation** group unchecked for production runs.

### Performance & Tracking

| GUI field | Value | Notes |
|---|---|---|
| **Skip frames** | `1--5` | Run full detection every N frames. Dramatic speedup at the cost of moving-object tracking fidelity. `1` = every frame |
| **Detect scale** | `0.85--0.95` | For cropped KITCO footage. Useful when your VP reference images and your video footage have mismatched resolutions |
| **ReID grace (s)** | `4--5` | Seconds to keep a lost track before dropping it -- the re-identification window after brief occlusion or head turns |
| **Obj persistence** | `0` | Frames to persist objects after disappearance; off by default |
| **Skip phenomena** | `0` | Run phenomena trackers every frame |

Leave the following **unchecked** for normal runs:

- **Fast mode (bundle perf optimizations)**
- **Lite overlay (minimal drawing)**
- **Include dashboard in video output**
- **Profile (per-stage timing)**

### Phenomena

Enable whichever phenomena trackers your study cares about in the Phenomena panel. Joint Attention is the standard one; Gaze Leadership and Social Referencing turned out to be unexpectedly informative on KITCO data, so consider enabling them even if they weren't in your original study design.

### Plugins

- **Iris Refined Gaze** -- worth enabling if your participants have clearly visible eyes/pupils. Improves per-face gaze on HD footage with visible irises; less useful at lower resolutions.
- **Gaze Boost** -- useful if you're seeing task-relevant objects flicker in and out of detection. Boosts the confidence of objects currently in a participant's line of sight, which helps keep handheld items detected.

---

## Visual Prompt Builder (VP Builder Tab)

`.vp.json` files encode the reference images and bounding-box annotations YOLOE uses for visual class prompting. The in-GUI VP Builder (Tab 2) is the intended workflow for creating them.

### Workflow

1. **Add Images.** Select one or more frames containing the objects you want to detect. Sampling from your actual study footage works better than using stock photos.
2. **Add Class** for each object category your study cares about.
3. **Draw bounding boxes.** Click-and-drag on the canvas. Each box is assigned to whichever class is currently selected on the right.
4. **Save VP File** to write the `.vp.json`.
5. **Test inference.** Select a YOLOE model, point at a folder of test images, and hit **Test** to preview detections.

### Best practices

Drawn from Section 11.6 of the paper plus my own experience:

- **Sample from study footage.** Visual prompts defined from your actual recordings consistently beat prompts built from external reference images.
- **Match prompt resolution to video resolution.** YOLOE encodes class embeddings from the *pixel size* of the example objects, not their semantic identity. If your prompt images are 4K and your video is 720p, objects will appear at different scales and detection confidence will tank. Normalising prompt resolution to roughly match expected video resolution makes a noticeable difference.
- **Watch for low colour contrast.** Objects that don't visually pop against their background are the most common detection failure. If you're sourcing study materials, prefer items that clearly contrast with the table/floor/walls they'll sit on.
- **Lower confidence than you'd expect.** Visual prompts often need a **Conf** around `0.20--0.30` for consistent detection, whereas text-class YOLO is happy around `0.35--0.50`. Combine the lower threshold with Merge Overlaps to suppress duplicates.
- **Larger / less occluded is more reliable.** Small, ambiguous, or frequently-occluded objects are inherently harder. Where possible, design the scene so task-relevant items stay visually distinct and minimally occluded.

---

## Data Outputs

MindSight produces several complementary output formats, configured in the GUI's **Output** section (or via the `output:` block in `pipeline.yaml`). They're all independent -- enable only what you need.

### Per-frame event CSV (GUI: *Event log*)

One row per gaze-object hit per frame. If a participant's gaze ray intersects two objects in the same frame, that's two rows. If no participant hits any object on a given frame, no rows are written for that frame.

**Columns:** `frame, face_idx, object, object_conf, bbox_x1, bbox_y1, bbox_x2, bbox_y2, joint_attention, joint_attention_confirmed, participant_label`. In project mode, `video_name` and `conditions` are prepended.

**When to use it.** Anything that needs frame-level granularity -- custom downstream analysis, validation against human coding, fixation/transition extraction, etc. If you only care about aggregate look-time per object, the Summary CSV is enough.

### Summary CSV (GUI: *Summary CSV*)

Post-run aggregations, one multi-section CSV. Each section is preceded by a `#`-prefixed header line. The built-in **Object Look Time** section gives one row per `(participant, object)` pair with `frames_active`, `total_frames`, and `value_pct`. Every active phenomena tracker contributes its own section -- joint attention proportions, mutual gaze pair counts, gaze leadership credits, etc.

**When to use it.** Almost always. This is the file most analyses will pull from.

### Heatmaps (GUI: *Heatmap*)

One PNG per participant, generated from the participant's gaze ray endpoints over the full video, Gaussian-blurred and blended over the middle frame as background. Defaults: `sigma = 40 px`, `alpha = 0.65`. Output lands in `Outputs/heatmaps/[video_stem]_Heatmap_Output/`.

**Important caveats:**
- If a participant has no gaze data collected (face never tracked, or all hits filtered out), no heatmap is generated for them. Check the console.
- The background is *always* the middle frame -- if your scene changes substantially over the video, the heatmap won't reflect that.
- When **Anonymize faces** is on, the heatmap background is anonymized too.

### Time-series charts (GUI: *Generate post-run charts*)

One multi-panel PNG per video (`Outputs/Charts/[video_stem]_Charts.png`) plus individual per-tracker PNGs. Each chart shows a phenomena metric over time -- joint attention %, gaze following events, etc. Only trackers that implement `time_series_data()` contribute charts; some lighter trackers don't.

**When to use it.** Quick visual sanity-check on a run, or when you want to see *when* in a video the phenomena spiked. For statistical analysis, use the underlying CSVs.

### Annotated video (GUI: *Save annotated video*)

The source video re-rendered with overlays: object bounding boxes (colour-coded by attention state), face rectangles with participant labels, gaze rays with state-aware endpoints (locked / snapped / extended), dwell-progress arcs, and the live phenomena dashboard panels on either side. Saved as `Outputs/Video/[video_stem]_Video_Output.mp4`.

Recorded as mp4v then automatically remuxed to H.264 post-run if `ffmpeg` is available (which gives you QuickTime compatibility); falls back to raw mp4v otherwise (still playable in VLC).

**When to use it.** Sanity-checking that detection and ray forming are doing what you think they're doing. Indispensable during parameter tuning; turn it off for production batches once you trust the pipeline, since it's the slowest output by a wide margin. The **Include dashboard in video output** checkbox in Performance & Tracking adds the side panels (~44% extra width) -- leave it off unless you specifically need it.

### Face anonymization (GUI: *Anonymize faces*)

Blurs or blackouts faces in saved video output (and heatmap backgrounds). The bounding box is padded by `0.3` (30%) of face dimensions by default, and the anonymization box uses temporal smoothing -- it grows instantly but shrinks over ~15 frames, so brief detection flicker doesn't show up as visible holes in the blur.

**Important:** anonymization happens *before* overlay rendering, which means gaze rays and bounding boxes are drawn *on top of* the blurred faces. The faces themselves are hidden, but participants can still be visually distinguished by which gaze ray they own. If you need fully de-anonymized output for sharing, plan for that at the analysis stage too.

### Participant IDs (GUI: *Participant IDs*)

Comma-separated, **positional** mapping. `S70,S71,S72` maps to face tracks 0, 1, 2 in the order they're first detected. Labels show up in event CSVs and on-screen overlays. Unmapped faces fall back to `P0, P1, ...`.

Note that ReID is per-video -- if you need consistent participant labels across multiple videos in the same study, use the project-mode `participants` mapping (see Batch Processing below) rather than this field.

### Output directory structure

By default, all outputs land in `Outputs/` at the project root:

```
Outputs/
├── CSV Files/
│   ├── [video_stem]_Summary_Output.csv
│   ├── [video_stem]_Events.csv
│   ├── Global_Summary.csv       # project mode only
│   ├── Global_Events.csv        # project mode only
│   └── By Condition/            # project mode only, if conditions defined
├── Video/
│   └── [video_stem]_Video_Output.mp4
├── heatmaps/
│   └── [video_stem]_Heatmap_Output/
│       ├── [video_stem]_S70_heatmap.png
│       └── [video_stem]_S71_heatmap.png
└── Charts/
    ├── [video_stem]_Charts.png
    └── [video_stem]_joint_attention.png
```

---

## Batch Processing (Project Mode)

> **Heads up:** Project mode is still actively under development at the time of writing. The directory layout, YAML schema, and aggregate-output behaviours described in this section reflect the current state of the codebase but may shift in future releases, and some edge cases I haven't personally exercised may not behave exactly as described. Treat this section as a working guide rather than authoritative documentation, and double-check against the GUI / a small test project before kicking off a long batch run.

For anything beyond a one-off run -- e.g. processing all 30+ KITCO sessions against the same pipeline -- use **Project mode**. A project is a directory with a standard layout: drop your videos into `Inputs/Videos/`, drop your visual prompt into `Inputs/Prompts/`, write a `pipeline.yaml` for your parameter set, and MindSight processes all videos sequentially with unified model loading and automatic per-condition aggregation.

### Directory layout

```
MyProject/
├── Inputs/
│   ├── Videos/           # All source videos go here (auto-discovered)
│   ├── Prompts/          # VP files (first .vp.json auto-loaded if no override)
│   └── AuxStreams/       # Optional -- eye_only / face_closeup / wide_closeup / custom
├── Outputs/              # Auto-created; per-video + Global + By Condition outputs
├── Pipeline/
│   └── pipeline.yaml     # Required -- the parameter set for this project
└── project.yaml          # Optional -- conditions, participant labels, output overrides
```

Missing subdirectories are auto-created on first run.

### `pipeline.yaml` -- the parameter set

This is where every value from the Recommended Parameters section above lives in YAML form. Sections mirror the GUI tabs (`detection`, `gaze`, `phenomena`, `output`, `performance`, `depth`, `plugins`). YAML keys are the underscore versions of the CLI flags. A minimal example:

```yaml
detection:
  vp_file: Inputs/Prompts/kitco.vp.json
  vp_model: yoloe-v8l-seg.pt
  conf: 0.25
  merge_overlaps: true
  merge_overlap_threshold: 0.55
gaze:
  ray_length: 1.5
  gaze_cone: 5.0
  rf_gazelle_model: Weights/Gazelle/gazelle_dinov2_vitb14.pt
  min_call_gap: 10
  dir_beta: 0.5
  len_beta: 0.3
phenomena:
  - joint_attention: {ja_window: 10, ja_quorum: 1.0}
  - mutual_gaze: {}
  - social_referencing: {window: 60}
  - gaze_following: {}
  - gaze_leadership: {}
output:
  save_video: true
  log_csv: true
  summary_csv: true
  heatmaps: true
performance:
  fast: false
```

You can build this file once in the GUI (using the Gaze Tracker tab with everything tuned the way you want), then *Export Pipeline* from the top toolbar to write it to disk. Any CLI flag passed at run time overrides the YAML value.

> For the full schema, see the [pipeline YAML reference](../reference/pipeline-yaml-schema.md).

### `project.yaml` -- conditions and participants

Optional file, lives at the project root. Defines per-video metadata that gets folded into output CSVs. Example:

```yaml
version: 1
pipeline: Pipeline/pipeline.yaml
conditions:
  P12_session1.mp4: [Emotional]
  P12_session2.mp4: [Neutral]
  P15_session1.mp4: [Emotional]
participants:
  P12_session1.mp4:
    0: S70
    1: S71
  P12_session2.mp4:
    0: S70          # Same subject across sessions -- reuse the label
    1: S71
output:
  directory: CustomOutput   # Optional; absolute or relative to project root
```

**Conditions** are pipe-delimited tags. Each output CSV row gets a `conditions` column (e.g. `"Emotional"`), and a `By Condition/` directory is auto-generated with separate aggregate CSVs per unique tag. Videos with multiple tags appear in each corresponding file.

**Participants** is the cross-video version of the GUI's Participant IDs field. Use this instead of the GUI field when the same subject appears in multiple videos -- it's how you keep labels consistent across the study.

#### Defining conditions in the GUI

Hand-writing YAML is fine for one project, but for studies with many videos the **Project Mode** tab gives you a faster workflow. After pointing the tab at your project directory, the **Conditions** section presents a two-column table (Video | Conditions) with one row per discovered video.

Two ways to tag:

1. **Direct edit.** Click into the Conditions cell for a row and type tags separated by `|`. For example, `Emotional` for a single tag, or `Emotional | Followup` if you have multiple.
2. **Bulk apply.** Select one or more rows (Cmd/Ctrl-click for multi-select), type a tag like `Neutral` into the **Tag to apply/remove** field, then click **Apply to Selected**. The same row supports **Remove Tag** (strips that specific tag from the selected rows) and **Clear All** (wipes all tags from the selected rows).

For a typical Emotional-vs-Neutral split, the workflow is: select all the emotional-condition videos → type `Emotional` → Apply to Selected → select the rest → type `Neutral` → Apply to Selected. Hit **Save project.yaml** in the top toolbar when you're done to persist the tags to disk.

#### Generating project.yaml from R

If your study metadata already lives in a CSV (which it usually does), you can generate a valid `project.yaml` programmatically rather than hand-tagging in the GUI. The worked example below uses base R plus the [`yaml`](https://cran.r-project.org/package=yaml) package, and assumes a dyadic study layout -- **two participants per video** -- since that matches the KITCO setup and most joint-attention paradigms.

Assume one master CSV at the project root with one row per session:

**`study_metadata.csv`**

```
video,participant_0,participant_1,condition
P12_session1.mp4,S70,S71,Emotional
P12_session2.mp4,S70,S71,Neutral
P15_session1.mp4,S82,S83,Emotional
P15_session2.mp4,S82,S83,Neutral
```

The cleanest workflow splits this into the two formats MindSight prefers natively: a `participant_ids.csv` for the participant mapping (auto-loaded from the project root) and a `project.yaml` for the conditions block.

```r
library(yaml)

# 1. Read the master CSV
meta <- read.csv("study_metadata.csv", stringsAsFactors = FALSE)

# 2. Participants → native participant_ids.csv
#    Pivot the wide (P0/P1) layout into the long format MindSight expects:
#    one row per (video, track_id, participant_label) triple.
parts <- rbind(
  data.frame(video_filename    = meta$video,
             track_id          = 0L,
             participant_label = meta$participant_0),
  data.frame(video_filename    = meta$video,
             track_id          = 1L,
             participant_label = meta$participant_1)
)
parts <- parts[order(parts$video_filename, parts$track_id), ]
write.csv(parts, "Projects/MyProject/participant_ids.csv",
          row.names = FALSE)

# 3. Conditions → project.yaml
#    Each video appears once in the master CSV, so split() gives a
#    length-1 vector per video -- which is exactly what we want.
conditions_block <- split(meta$condition, meta$video)

project <- list(
  version    = 1L,
  pipeline   = "Pipeline/pipeline.yaml",
  conditions = conditions_block
)
write_yaml(project, "Projects/MyProject/project.yaml")
```

If you'd rather keep everything in a single `project.yaml` -- e.g. because you want to commit one configuration file to version control rather than two -- swap step 2 for an in-YAML participants block:

```r
participants_block <- setNames(
  Map(function(p0, p1) list("0" = p0, "1" = p1),
      meta$participant_0, meta$participant_1),
  meta$video
)

project <- list(
  version      = 1L,
  pipeline     = "Pipeline/pipeline.yaml",
  conditions   = conditions_block,
  participants = participants_block
)
write_yaml(project, "Projects/MyProject/project.yaml")
```

The split-CSV approach is the cleaner default though -- participant mappings are inherently a flat table (a natural CSV fit), while conditions are inherently a per-video tag list (a natural YAML fit), and using each format for what it's best at sidesteps R's habit of quoting integer-looking YAML keys.

### Running a project

**CLI:**

```bash
python MindSight.py --project Projects/MyProject/
# or
mindsight --project Projects/MyProject/
```

**GUI:** Open the **Project Mode** tab (Tab 3), point it at your project directory, and use the live monitoring panel to watch progress.

The GUI has tables for editing the participant and condition mappings in place -- handy for iterating without hand-editing YAML. Use *Save project.yaml* once you're done to persist changes.

### Aggregate outputs

In addition to per-video outputs (same as single-video mode), project runs automatically produce:

- **`Outputs/CSV Files/Global_Summary.csv`** -- every video's summary stacked into one file
- **`Outputs/CSV Files/Global_Events.csv`** -- every video's per-frame events stacked into one file
- **`Outputs/By Condition/[condition]_Summary.csv`** and **`[condition]_Events.csv`** -- one pair per unique condition tag

These are what most cross-video analyses should pull from.

### Auxiliary streams (advanced)

If your study has supplementary footage -- close-up eye cameras, face-only crops, wide shots, etc. -- you can wire those into a project via `Inputs/AuxStreams/`:

```
Inputs/AuxStreams/
├── eye_only/S70.mp4              # Per-participant
├── face_closeup/S70+S71.mp4      # Multi-participant (+ separator)
└── wide_closeup/session1.mp4
```

Aux streams are opened once and frame-synchronized with the main video. Plugins can request frames from them via `find_aux_frame()`. **Heads up:** if an aux stream's FPS differs from the main video by more than 1 fps, you'll get a console warning and the streams will drift over time -- match FPS at capture time when possible.

### Pitfalls

- **VP file discovery.** If you don't specify `vp_file` in `pipeline.yaml` and there's a `.vp.json` in `Inputs/Prompts/`, the first one alphabetically gets auto-loaded. Keep one VP file per project to avoid surprises.
- **ReID is per-video, not per-project.** Use the `participants` block in `project.yaml` to enforce consistent labels across videos in the same study.
- **Condition aggregation needs consistent tag spelling.** `"Emotional"` and `"emotional"` are different tags. Pick a convention and stick to it.
- **CLI flags override `pipeline.yaml`.** This is occasionally useful (one-off tweaks) and occasionally confusing (you forgot you passed `--conf 0.5` last time). When in doubt, run from a clean shell.
- **Annotated video generation is slow.** For large batches, turn `output.save_video` off in `pipeline.yaml` once you trust the pipeline -- you can always re-run a single problematic video with video output on later.

---

## Pitfalls & Things to Watch For

A short list of things the paper flags that are easy to trip over in practice:

- **Camera geometry matters.** A camera placed roughly 45° above and angled down works best -- you get partial depth cues, faces stay forward-facing, and extreme occlusion is rare. Directly overhead or horizontally offset footage gives substantially worse results from face-detection failures and extreme-angle gaze degradation (Section 11.5).
- **Post-processing is not optional.** With post-processing disabled, raw per-face output is essentially unusable -- either you set the ray short and miss real fixations, or you extend it and accumulate false positives. The tuning of the post-processing chain is the primary determinant of output quality, not backend choice (Section 11.2).
- **Per-face backends are 2D and depth-blind.** A raw pitch/yaw vector has no information about how *far* away the object is. This is what Gaze-LLE Blend's **length channel** is fixing -- the scheduler feeds scene-aware length corrections while a participant fixates, and the One Euro length smoother holds that reach steady between inferences.
- **Adaptive Snap needs dense scenes.** The legacy adaptive-snap mode assumes roughly 4--8 tracked objects near the task area. In object-sparse scenes it has nothing to score against and falls back to raw rays. Gaze-LLE Blend doesn't have this failure mode.
- **MindSight is not a hardware replacement.** Per-face backends report 4--11° mean angular error vs sub-degree precision for Tobii/EyeLink. For studies where "who is looking at what" matters more than sub-degree precision, this is fine; for fixation/saccade dynamics or reading paradigms, it isn't.
