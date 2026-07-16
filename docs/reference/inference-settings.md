# Inference Settings

**Inference Settings** is the control panel for *how* footage is processed. Every
run launched from Analyze Footage -- project runs, quick runs, and camera runs --
reads its models, gaze-estimation behaviour, object detection, phenomena,
outputs, and performance knobs from this one dialog. It does not choose *what* to
process (source video, output folder, participants, conditions): those are set
per run by the project or quick-run layer.

Open it from the **Inference Settings...** button on every Analyze Footage mode,
or from the **Tools** menu. The dialog is a modal panel with a vertical list of
seven tabs on the left and a scrollable page on the right, plus **OK**,
**Cancel**, and **Apply**. When a project is open you also get **Save to project
pipeline...**, and you can round-trip the full configuration through **Import /
Export YAML**.

The header names the preset the settings came from. On a fresh install that is
**KG_Standard**, the shipped known-good preset; once you change anything the
header gains a **(modified)** suffix, and after **Save to project pipeline...**
it reflects the open project's pipeline. Numeric settings pair a slider (which
bounds the *recommended* range) with a typeable value box (which is
authoritative). Typing a value beyond the slider's range never clamps it: the
slider greys or pins and the value turns **amber** with a tooltip noting it is
outside the usual range.

!!! note "Inference Tuning is decoupled"
    The **Inference Tuning** tab is a live playground for *finding* good
    values; by design nothing you try there changes your study's runs. Only
    this dialog's stored settings drive runs. When an Inference Tuning
    experiment is worth keeping, bring the values across with **Import from
    Inference Tuning** in this dialog. See the
    [Inference Tuning](../studies/run-a-study-tutorial.md#inference-tuning)
    section of the Run a Study tutorial.

Throughout, weight paths display as model family names or bare filenames and
resolve against the shared `Weights/` folder -- you never type absolute paths.

---

## Models & Device

Which models run and where. The tab links to the Models manager for downloading
and verifying weights rather than duplicating it.

| Setting | What it does |
|---|---|
| Compute device | Where models run: Auto picks the best available (NVIDIA GPU > Apple GPU > CPU). |
| No object detection | Run without an object-detection model: faces, gaze rays, and gaze-tip phenomena only. Object hits and object lock-on are disabled -- for lightweight attention studies. Not compatible with a visual prompt file. |
| Object detection model | The YOLO model that finds objects and people in each frame. Smaller = faster, larger = more accurate. |
| Visual prompt model | The YOLOE model used when a visual prompt file teaches the detector your study's custom objects. |
| Visual prompt file | The `.vp.json` from the VP Builder describing your study's objects. Empty = standard classes only. |
| Gaze model (MobileGaze) | The per-face gaze direction model. Family name (e.g. "resnet50") auto-selects the right build for your device. |
| Gaze-LLE model | The scene-aware model used by Gaze-LLE Correction (Gaze Estimation tab) to periodically correct gaze rays. |

### Advanced

| Setting | What it does |
|---|---|
| MobileGaze architecture | Architecture name, required for `.pt` MobileGaze weights (e.g. resnet50). |
| MobileGaze dataset key | Dataset the gaze model was trained on (default: gaze360). |
| Gaze-LLE variant | The Gaze-LLE model variant used for correction (default: gazelle_dinov2_vitb14). The `vitl14` variants are markedly heavier per correction (roughly 3-4x the latency spike) for a modest accuracy gain -- consider them only for offline batch runs where per-correction stalls don't matter. With the in/out gate active (v1.1), a checkpoint carrying the in/out head auto-upgrades to its `_inout` architecture unless a variant is set explicitly. |

---

## Gaze Estimation

How gaze rays are drawn, corrected, and matched to objects.

### Gaze rays

| Setting | What it does |
|---|---|
| Ray length | How far the drawn gaze ray reaches, as a multiplier of face size. |
| Looking-at-camera threshold (deg) | Pitch/yaw below this counts as "looking at the camera", not at the scene. 0 disables. |
| Gaze cone (deg) | Replace the thin ray with a vision cone of this angle. 0 = ray. Cones catch more objects, less precisely. |

**Advanced**

| Setting | What it does |
|---|---|
| Confidence-scaled ray length | Shorten the ray when the gaze model is unsure. |

### Gaze-LLE Blend

The primary validated correction mode, **on in KG_Standard**. It periodically
runs the scene-aware Gaze-LLE model to nudge each person's gaze ray toward what
they are actually fixating.

| Setting | What it does |
|---|---|
| Enable Gaze-LLE Blend | Periodically run the scene-aware Gaze-LLE model to correct each person's gaze ray toward what they're actually fixating. The primary validated mode. |
| Correction interval (frames) | Minimum frames between corrections. Lower = more corrections, slower processing. |
| Direction responsiveness | How quickly the corrected ray direction follows fast motion. |
| Length responsiveness | Same, for corrected ray length. |
| Length hold (s) | How long a corrected ray length persists before decaying to baseline. |

**Advanced**

| Setting | What it does |
|---|---|
| Fixation velocity threshold | Smoothed pitch/yaw velocity (rad/frame) at which a face is treated as 50% fixating. Lower anchors corrections more cautiously. |
| Fixation dispersion threshold | Windowed pitch/yaw dispersion (rad) at which a face is treated as 50% fixating. |
| Direction smoother floor (Hz) | One-Euro smoother floor cutoff for corrected ray direction. Lower = smoother at rest. |
| Length smoother floor (Hz) | One-Euro smoother floor cutoff for corrected ray length. |

### Object lock-on

| Setting | What it does |
|---|---|
| Lock rays onto objects | When a ray passes near a detected object: "reach toward object" extends it; "lock onto object" pins the endpoint to the object. |
| Lock-on distance (px) | How close a ray must pass to an object to lock on. |
| Smooth lock-on movement | Glide the ray toward its target instead of jumping: objects / gaze tips / all. |
| Smoothing rate | How fast the ray glides toward its target. Lower = smoother/slower; higher = faster/more responsive. |

**Advanced -- lock-on scoring** (how candidate objects are ranked and accepted)

| Setting | What it does |
|---|---|
| Bbox scale | Fraction of the object's bbox half-diagonal added to the lock-on radius. |
| Distance weight | Scoring weight for how close the ray passes to the object. |
| Angle weight | Scoring weight for angular deviation between ray and object. |
| Size weight | Scoring reward for larger objects. |
| Intersection weight | Scoring bonus when the ray actually crosses the object box. |
| Stickiness weight | Scoring bonus for staying on the previous frame's target. |
| Gate angle (deg) | Hard angular cutoff -- objects beyond this angle from the blended gaze+head direction are never candidates. |
| Head-direction blend | How much head orientation (vs pure gaze direction) drives angular scoring: 0 = gaze only, 1 = head only. |
| Quality gate | Maximum score allowed to accept a lock-on match; lower rejects poorer matches. |
| Tip distance (px) | Lock-on distance threshold for gaze tips (-1 = use Lock-on distance). |
| Tip quality gate | Quality threshold for tip lock-on (-1 = use Quality gate). |
| Release frames | Frames without a match before a held lock-on is released. |
| Engage frames | Frames of consistent match required before locking on (0 = instant). |

### Gaze tips

| Setting | What it does |
|---|---|
| Gaze tips (virtual targets) | Mark each ray's endpoint with a circular target so two people's gaze can meet in empty space -- tip convergence counts as joint attention. |
| Tip radius (px) | Size of the endpoint target. |

### Gaze-object hits

| Setting | What it does |
|---|---|
| Hit confidence gate | Ignore gaze-object hits from faces with weaker gaze estimates than this. 0 = off. |

**Advanced**

| Setting | What it does |
|---|---|
| Extend hit reach (px) | Count objects up to N px past the visible ray end as hits. |
| Extended reach applies to | Object hits, phenomena, or both. |

---

## Object Detection

How the detector accepts and de-duplicates boxes, and how long it remembers
objects and people.

| Setting | What it does |
|---|---|
| Detection confidence | Minimum confidence to accept a detection. Lower = more objects, more false positives. |
| Merge overlapping detections | Combine duplicate boxes on the same object. |
| Merge strategy | Keep best box / merge boxes / decide per case ("dynamic"). |
| Merge threshold | How much boxes must overlap before merging. |

### Advanced

| Setting | What it does |
|---|---|
| Object classes | Restrict detected classes. |
| Class blacklist | Exclude detected classes. |
| Keep lost objects (frames) | Keep an object alive N frames after the detector loses it. |
| Track re-ID grace (s) | How long a lost person track can reappear with the same identity. |
| Gaze-guided detection boost | Boost detector confidence for objects near where people look. |
| — Boost factor | Multiplier applied to detector confidence near gaze. |
| — Boost radius (px) | Pixel radius around gaze endpoints where the boost applies. |
| — Min confidence | Lowest raw detector confidence eligible for a boost. |
| — Max confidence | Cap on the boosted confidence. |
| — Boost classes | Restrict the boost to these object classes (default: all non-person classes). |

---

## Phenomena

One checkable group per phenomenon: the checkbox enables the tracker, and its
parameters sit beneath it. An **Enable all phenomena** button at the top is a
bulk action, not a stored setting. The output pages -- for example
[Joint Attention](../phenomena/joint-attention.md) -- explain what each
phenomenon measures.

!!! note "Tip convergence is joint attention"
    Tip convergence *is* joint attention (a per-frame union, never
    double-counted). Enabling Gaze tips extends joint attention to gaze that
    meets in empty space.

| Setting | What it does |
|---|---|
| Joint Attention | Enable joint-attention tracking. Parameters: **Consistency window (frames, 0 = off)** -- frames a joint-attention episode must hold to count; **Window threshold** -- fraction of the window that must agree before it counts; **Participant quorum** -- fraction of participants that must share a target. |
| Mutual Gaze | Enable mutual-gaze tracking. |
| Social Referencing | Enable social-referencing tracking. **Window (frames)** -- frames over which a look-back is counted. |
| Gaze Following | Enable gaze-following tracking. **Max follow lag (frames)** -- longest delay still counted as one person following another's gaze. |
| Gaze Leadership | Enable gaze-leadership tracking. **Count tip convergence** -- also detect leadership via gaze-tip convergence (needs Gaze tips); **Tip lookback (frames)** -- lookback window for tip-arrival priority. |
| Gaze Aversion | Enable gaze-aversion tracking. **Window** -- frames over which aversion is measured; **Confidence** -- minimum gaze confidence for an aversion to count. |
| Scanpath | Enable scanpath tracking. **Min dwell (frames)** -- minimum frames on a target before it enters the scanpath. |
| Attention Span | Enable attention-span tracking. |
| Eye Movement Classification | Enable eye-movement classification. Fixation/saccade parameters are Advanced (below). |
| Novel Salience | Enable novel-salience tracking. Sensitivity parameters are Advanced (below). |
| Pupillometry | Enable pupillometry. Measurement and filtering parameters are Advanced (below). |

### Advanced

**Eye Movement Classification**

| Setting | What it does |
|---|---|
| Velocity source | Signal the classifier measures velocity from (default: gaze). |
| Saccade threshold | Velocity (px/frame) above which motion counts as a saccade. |
| Fixation threshold | Velocity (px/frame) below which motion counts as a fixation. |
| Min fixation frames | Shortest run of frames that counts as a fixation. |
| Velocity window | Median-filter window for smoothing velocity. |

**Novel Salience**

| Setting | What it does |
|---|---|
| Speed threshold | Gaze-endpoint speed (px/frame) that flags an event. Lower = more sensitive. |
| Cooldown (frames) | Minimum frames between events for the same face. |
| History | Sliding-window depth for velocity smoothing. |
| Flash (frames) | How long the on-video saccade indicator persists after an event. |

**Pupillometry**

| Setting | What it does |
|---|---|
| Measurement mode | How pupil size is measured (default: rgb). |
| Baseline frames | Frames used to calibrate a baseline. |
| Upscale | Upscale factor for the eye crop in RGB mode. |
| Filter | Smoothing filter applied to the pupil ratio (default: kalman). |
| EMA alpha | Smoothing strength when the filter is EMA. |
| Kalman measurement noise | Higher values smooth more aggressively (Kalman filter only). |
| Kalman process noise | How fast the Kalman filter adapts to changes. |
| Blink frames | Consecutive low-eye-aspect-ratio frames that count as a blink. |
| Eye-aspect-ratio threshold | EAR below which a blink is detected. |
| IR threshold | Brightness cutoff for dark-pupil segmentation in IR mode. |
| Outlier window | Window size for the Hampel outlier filter. |
| Per-eye | Report left and right pupils separately. |

---

## Output

The events CSV and summary CSV are **always** written -- they are not toggles,
and their paths come from the project or quick-run layer, so no path fields
appear here. These settings govern the optional extras.

| Setting | What it does |
|---|---|
| Save annotated video | Record the video with overlays drawn. |
| Gaze heatmaps | Per-participant heatmap images after each run. |
| Post-run charts | Time-series charts per phenomenon appear in the Charts tab after a run (no separate chart files are written to disk for GUI runs). |
| Anonymize faces | Blur or black-box faces in the output video. |
| Overlay detail | Full overlays vs minimal (no cones, markers, debug text). Checked = minimal. |

!!! note "Anonymize on project runs"
    On project runs the study-setup **Anonymize Footage** checkbox overrides this
    value at launch; the dialog value governs quick runs and YAML exports.
    Padding is honoured everywhere.

### Advanced

| Setting | What it does |
|---|---|
| Show dashboard panels | Compose the side dashboard onto processed frames (off = fastest; the GUI Live tab works regardless). |
| Padding | Extra margin around each face box when anonymizing, as a fraction of face size. |

---

## Performance

Speed-versus-accuracy trade-offs.

| Setting | What it does |
|---|---|
| Detect every Nth frame | Run object detection every N frames; tracking fills gaps. Higher = faster, less accurate. |
| Detection scale | Downscale frames before detection. 1.0 = full resolution. |
| Fast mode | Bundled speed optimizations (skip phenomena on non-detection frames, throttle previews). |

### Advanced

| Setting | What it does |
|---|---|
| Phenomena every Nth frame | Run phenomena trackers every N frames. 0 = every frame. |

!!! warning "Low Power preset"
    On weak hardware you can load the **Low Power** preset -- but it is
    UNVALIDATED for research conclusions.

---

## Advanced & Experimental

Whole unvalidated or half-wired features, grouped so a novice never mistakes them
for supported controls. Each is off by default. Handle with care -- do not rely
on any of these for research conclusions.

| Setting | What it does |
|---|---|
| Depth estimation (EXPERIMENTAL) | Estimate scene depth for depth-aware lock-on scoring. |
| — Backend | Depth model backend (default: midas_small). |
| — Input size (px) | Depth model input resolution. |
| — Skip frames | Run depth every N detection cycles. |
| — Depth-aware lock-on scoring | Fold depth agreement into lock-on scoring. |
| — Depth weight | Weight of the depth match in lock-on scoring. |
| — Sample radius | Half-size of the patch sampled for depth. |
| Depth-scaled ray length (EXPERIMENTAL) | Scale ray length from scene depth. |
| — Min multiplier | Ray length multiplier at the nearest depth. |
| — Max multiplier | Ray length multiplier at the farthest depth. |
| — Belief boost | How much depth agreement boosts Gaze-LLE heatmap confidence during correction. |
| Iris refinement (EXPERIMENTAL) | Wrap the gaze backend with iris-based correction. |
| — Weight | Blend weight for the iris correction. |
| — Upscale | Upscale face crops before iris extraction. |
| Gaze-LLE-only backend (EXPERIMENTAL) | A heatmap-only gaze backend, distinct from the validated MobileGaze + Gaze-LLE Blend path. |
| — Variant | The Gaze-LLE model variant (e.g. gazelle_dinov2_vitb14). |
| — In/out threshold | In/out-of-view confidence threshold for the `*_inout` variants. |
| — Device | Compute device for this backend (auto, cpu, cuda, or mps). |
| — Skip frames | Reuse the previous gaze result for N frames between inference runs. |
| — FP16 | Half-precision inference on CUDA/MPS. |
| — Compile | Use `torch.compile()` (PyTorch 2.0+ only). |
| Debug overlay | Pitch/yaw debug text drawn on the video. |

---

## Presets and persistence

- **Seeded from KG_Standard.** A fresh install seeds the dialog from the shipped
  known-good preset, KG_Standard. "Default" in the UI means the preset value, not
  the raw code default.
- **Reset to preset.** The header's reset control returns every setting to the
  active preset, clearing the **(modified)** state.
- **Save to project pipeline.** With a project open, **Save to project
  pipeline...** writes the current settings into that project's pipeline preset,
  so the whole study runs with them. Coordinate with your study lead before
  changing a running study's settings.
- **Settings persist per user.** Your choices are remembered across sessions in
  your user profile (`~/.mindsight/run_settings.json`) and reload the next time
  you open the dialog.
- **Weights resolve globally.** Bare weight names and family names resolve
  against the shared `Weights/` folder; weight resolution is global, not
  per-project.

See the [Inference Settings](../studies/run-a-study-tutorial.md#inference-settings)
walkthrough in the Run a Study tutorial for this dialog in context.
