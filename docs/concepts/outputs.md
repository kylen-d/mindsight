# Understanding the Outputs

A MindSight run produces several complementary files. They are independent --
you enable only the ones you need -- and they land under an `Outputs/`
directory at the project root. This page explains what each file is, when to
use it, and which files an analyst actually needs.

If you just want the short answer: the **summary CSV** is the file most
analyses pull from, and the **provenance manifest** is what makes a result
traceable. Everything else is either finer-grained (the event CSV) or a visual
aid (heatmaps, charts, annotated video).

## Per-frame event CSV

One row per gaze-object hit per frame. If a participant's gaze ray intersects
two objects in the same frame, that is two rows; if nobody hits any object on a
frame, that frame contributes no rows.

Core columns include the frame number, the tracked `face_idx`, the
`participant_label` (if participants were mapped), the `object` class and its
`object_conf`, the object bounding box (`bbox_x1, bbox_y1, bbox_x2, bbox_y2`),
and whether joint attention was active on that frame. In project mode a
`video_name` and `conditions` column are prepended so rows from many videos can
be stacked.

**When to use it.** Anything needing frame-level granularity -- custom
downstream analysis, validation against human coding, fixation or transition
extraction. If you only care about aggregate look-time per object, the summary
CSV already has it.

## Summary CSV

The post-run aggregation, written as one multi-section CSV. Each section is
preceded by a `#`-prefixed header line. The built-in **object look-time**
section gives one row per `(participant, object)` pair with the number of
active frames, the total frames, and a percentage. Every active phenomena
tracker appends its own section -- joint attention proportions, mutual-gaze
pair counts, gaze-leadership credits, and so on.

**When to use it.** Almost always. This is the file most statistical analyses
start from. Every CSV MindSight writes is tidy long-format (one typed column
per field), so it loads directly into R or pandas without reshaping.

## Provenance manifest

Alongside the summary, each run writes a `{stem}_manifest.json`. It records
exactly what produced that run's data: the full resolved configuration, a
run-identity hash, the model-weight hashes, and the library versions in play.
Because everything that affects the result is captured, a run is always
reproducible and auditable after the fact -- if two runs of the "same" study
disagree, the manifests tell you what actually differed.

**When to use it.** Keep it. You may never open it by hand, but it is what lets
you (or a reviewer) trust a result months later.

## Heatmaps

One image per participant, accumulated from that participant's gaze-ray
endpoints over the whole video, Gaussian-blurred and blended over a background
frame. A participant with no collected gaze data (face never tracked, or all
hits filtered out) produces no heatmap -- check the console if one is missing.
The background is a single frame, so a scene that changes substantially over
the video will not be fully reflected in the heatmap.

**When to use it.** A quick, human-legible picture of where attention
concentrated. Not a substitute for the CSVs in analysis.

## Time-series charts

Post-run charts showing phenomena metrics over time -- joint-attention
percentage, gaze-following events, and so on -- one multi-panel image per
video plus per-tracker images. Only trackers that expose time-series data
contribute; some lighter trackers do not.

**When to use it.** A visual sanity check, or to see *when* in a video a
phenomenon spiked. For statistics, use the underlying CSVs.

## Annotated video

The source video re-rendered with overlays: object bounding boxes colour-coded
by attention state, face rectangles with participant labels, gaze rays with
state-aware endpoints, and phenomena dashboard panels. It is the slowest output
by a wide margin.

**When to use it.** To confirm that detection and ray forming are doing what
you think they are -- indispensable during parameter tuning. Turn it off for
production batches once you trust the pipeline; you can always re-render a
single problematic video later. Face anonymization, when enabled, blurs or
blacks out faces in this output (and in heatmap backgrounds).

## Project-mode aggregates

When you process a whole project rather than a single video, MindSight
additionally produces batch-level files:

- **`Global_Summary.csv`** and **`Global_Events.csv`** -- every video's
  summary (or events) stacked into one file.
- **`By Condition/`** -- one summary/events pair per unique condition tag, for
  studies that tag videos with experimental conditions.

These are what most cross-video analyses should pull from.

## The resume ledger

Project runs resume by default. MindSight keeps a per-batch ledger at
`Outputs/_run/ledger.json` recording each video's status, so an interrupted or
re-launched batch skips work that is already complete and only processes what
remains. Forcing a full reprocess (`--no-resume`) reprocesses every video in
place. The ledger is bookkeeping, not analysis data -- you do not hand it to an
analyst.

## Output directory at a glance

```
Outputs/
├── CSV Files/
│   ├── {stem}_summary.csv        # tidy scalar summary (per video)
│   ├── {stem}_Events.csv         # per-frame hits (per video)
│   ├── {stem}_manifest.json      # provenance manifest (per video)
│   ├── Global_Summary.csv        # project mode only
│   ├── Global_Events.csv         # project mode only
│   └── By Condition/             # project mode only, if conditions defined
├── Videos/                       # annotated videos
├── {stem}_Heatmap/               # per-participant heatmap images
├── Charts/                       # time-series charts
└── _run/
    └── ledger.json               # resume bookkeeping
```

Run-folder projects gather the per-run files under `Outputs/Runs/<run_id>/`
instead, named by run id; the `Global_*.csv`, `By Condition/`, and `_run/`
locations stay the same.

## What to hand to an analyst

For most studies the handoff is small: the **summary CSV** (or the project-mode
`Global_Summary.csv` / `By Condition/` files), plus the **provenance
manifest** so the numbers are traceable. Include the per-frame **event CSV**
only when the analysis needs frame-level detail. Heatmaps, charts, and the
annotated video are for review and communication, not for statistics.
