# Validation & testing

The **Inference Tuning** tab's bottom-right quadrant is the Validation &
Testing workbench: label a handful of frames from your own footage once, and
every settings experiment can be *scored* against them — mean pixel error,
hit rate, angular error, off-screen detection, and processing speed — instead
of eyeballed. It closes the tune → run → squint loop into tune → **Validate**
→ read a number.

!!! note "Where it fits"
    The workbench scores the settings **currently dialed into the tuning
    tab** (the same decoupled sandbox described in
    [Inference settings and tuning](inference-settings-and-tuning.md)).
    Nothing it does affects study runs; when a validated configuration wins,
    bring it across with **Import from Inference Tuning...** in the Inference
    Settings dialog, exactly as you would after any tuning experiment.

---

## Validation sets

A **validation set** is a named collection of ground-truth labels for one
video: for a sample of frames, *where each participant is actually looking*
(a clicked point, or an off-screen / uncertain / skip ruling).

Create one with **New...** — a three-step wizard:

1. **Set** — name the set and pick its source video. The set is saved as soon
   as you continue, so nothing is lost if you stop early.
2. **Frames** — sample frames from the clip. The spinner is "sample every N
   frames" and translates itself live ("= every 1.0 s at 30 fps → adds ~29
   frames"); you can also add single frames by number. Labeling unlocks once
   the set has frames.
3. **Label** — for each frame, click where each participant is looking; the
   wizard advances through participants and frames as you click (`n` / `b`
   step forward and back). Use **Looking off-screen** when the gaze
   target is outside the frame, **Uncertain** when you cannot tell, and
   **Skip participant** to exclude someone from that frame. Every click
   autosaves.

**Annotate...** reopens the wizard for an existing set (straight to labeling,
or to the frames page while the set is still empty); **Delete** removes a
set's labels but keeps its past run results.

!!! tip "How many labels?"
    Even 20–50 labeled points give a usable signal for comparing settings —
    the point is *relative* scoring between runs, not a publication-grade
    benchmark. Spread samples across the clip rather than labeling one dense
    burst.

Set files are ordinary JSON in `<project>/validation/` (or
`~/.mindsight/validation/` outside a project), and they are valid
eval-harness label files — `scripts/eval_gaze.py score` reads them unchanged.

---

## Validate

**▶ Validate (current settings)** runs the full pipeline over the set's video
with the tab's current settings — live frame counter, fps, ETA and progress
bar, with **Cancel** always available (a cancelled run still scores whatever
it processed). When it finishes, the metrics table fills in, side by side
with the previous run:

| Metric | Meaning |
|--------|---------|
| mean / median / p95 px | Distance between each predicted ray endpoint and the labeled gaze point. |
| gaze hit rate | Fraction of labeled points within the hit radius (80 px). |
| MAE (degrees) | Mean angular error between the predicted ray and the origin→label direction. |
| off-screen AUC | How well the in/out-of-frame score separates your off-screen labels from on-screen ones (needs off-screen labels and an inout-capable blend model). |
| avg fps | Processing throughput of the run — accuracy changes that cost speed show up here. |

Every run is preserved under `validation/.runs/<set>/run-NNN/` with its
streams, `score.json`, and a `settings.json` snapshot of the namespace that
produced it.

- **History...** lists every scored run for the set, newest first, with a
  *changed vs previous* column — the settings diff between consecutive runs,
  so "what did I change to get this number" is always answerable.
- **Embed...** writes the set's latest score into a pipeline YAML's
  `validation:` block — metadata only (it never affects runs, resume, or the
  config hash), so a shared pipeline file can carry the evidence for its own
  settings.

---

## Auto-tune

**Auto-tune...** automates the loop for one or two knobs at a time: pick the
knobs, give each a comma-separated value list, and every combination runs
sequentially through the same runner as Validate (up to 12 combinations; the
dialog shows a live count and a time estimate based on the set's last
measured fps).

The curated knob list: ray length gain, min call gap, length refresh gap,
detection confidence, snap quality threshold, length slew, and detection
scale.

Results land in a table sorted by mean pixel error with the winner in bold,
and **Apply best to tab** writes the winning values back into the tuning tab
— pressing Validate immediately reproduces the winning run. Each combination
is an ordinary scored run, so History shows sweeps with their settings diffs
like any other runs. Sweeps persist (`validation/.runs/<set>/sweep-NNN.json`);
reopening the dialog shows the set's last sweep, and cancelling keeps every
completed combination's score.

!!! tip "Sweep small"
    Two knobs × three values each = nine full pipeline runs. Start from the
    knob you suspect matters, sweep it alone with 3–4 values, then sweep the
    runner-up against the winner. The 12-combination cap is there to keep
    sweeps answerable-in-one-coffee, not to be filled.
