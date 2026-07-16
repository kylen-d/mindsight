# Understanding the Outputs

A MindSight run produces several complementary files. They are independent --
you enable only the ones you need -- and they land under an `Outputs/`
directory. This page explains what each file is, when to use it, and how to load
it into an analysis.

If you just want the short answer: the **summary CSV** is the file most analyses
pull from, the **phenomena-events CSV** is where every timed episode lives, and
the **provenance manifest** is what makes a result traceable. Everything else is
either finer-grained (the per-frame events CSV, per-tracker streams) or a visual
aid (heatmaps, charts, annotated video).

Every CSV MindSight writes is **tidy long-format**: one typed value per row, one
column per field, and no `#`-prefixed section headers. Files load directly into
R (`read.csv`) or pandas (`pd.read_csv`) without any reshaping.

## Summary CSV -- one tidy table

The post-run aggregation is a single file, `{stem}_summary.csv`, with exactly
one header and one row per scalar metric:

```
video_name,conditions,phenomenon,participant,partner,object,metric,value
```

Every enabled tracker contributes rows into this one table -- there are no
per-tracker sections. The `phenomenon` column tells you which measure a row
belongs to, and `metric`/`value` carry the number. `participant`, `partner`, and
`object` are filled in only when a metric is scoped to them (they are empty
strings otherwise). In single-video mode `video_name` and `conditions` are empty
strings; in project mode they identify the source, so many videos stack into one
file with no format change.

A few example rows from a run with joint attention, mutual gaze, and object
look-time enabled (single-video mode, so the first two columns are blank):

```
video_name,conditions,phenomenon,participant,partner,object,metric,value
,,joint_attention,all,,,frames_active,842
,,joint_attention,all,,,pct_of_video,46.7813
,,mutual_gaze,P0,P1,,frames_active,315
,,mutual_gaze,P0,P1,,pct_of_video,17.5000
,,object_look_time,P0,,cup,seconds_active,12.500
```

**How to filter it.** Because every measure lives in the same table, you select
what you want with an ordinary row filter rather than by reading a particular
section. In R:

```r
df <- read.csv("trimmed_summary.csv")

# All joint-attention active-frame counts:
subset(df, phenomenon == "joint_attention" & metric == "frames_active")

# One participant's look-time across objects:
subset(df, phenomenon == "object_look_time" & participant == "P0")

# Pivot a metric wide if you prefer a matrix:
library(tidyr)
pivot_wider(df, names_from = metric, values_from = value)
```

Note that `value` is read as text (some metrics are integer frame counts, others
are formatted seconds or percentages); coerce with `as.numeric(df$value)` after
filtering to a single metric.

**When to use it.** Almost always. This is the file most statistical analyses
start from.

## Phenomena-events CSV -- the episode stream

Alongside the summary, a run writes `{stem}_phenomena_events.csv`: one merged,
time-ordered stream of every *episode* every tracker recorded -- each
mutual-gaze pair while it was active, each aversion streak, each confirmed
joint-attention span, each point event (a gaze-following or social-referencing
moment). The summary tells you *how much*; this file tells you *when*.

Its header is:

```
video_name,conditions,phenomenon,participant,partner,object,frame_start,frame_end,t_start,t_end,duration_s
```

Frame bounds are always present; `t_start`, `t_end`, and `duration_s` are
seconds (blank only when the frame rate is unknown). Point events have
`frame_start == frame_end` and a zero duration. Example rows:

```
video_name,conditions,phenomenon,participant,partner,object,frame_start,frame_end,t_start,t_end,duration_s
,,mutual_gaze,P0,P1,,120,168,4.000,5.600,1.600
,,joint_attention,all,,cup,300,354,10.000,11.800,1.800
,,tip_convergence,P0+P1,,,410,470,13.667,15.667,2.000
,,social_referencing,P2,,cup,502,502,16.733,16.733,0.000
```

**When to use it.** Any analysis of timing, duration, or sequence -- episode
counts and lengths, overlap between phenomena, alignment against an external
event track. It is written whenever at least one tracker logged an episode.
(The scanpath tracker records no episodes, so it never appears here; its
fixations live in its own stream, below.)

## Per-tracker stream tables

Some trackers emit a row-per-event table of their own, written alongside the
summary as `{stem}_<stream>.csv`. The built-in one is the scanpath stream,
`{stem}_scanpath.csv`, with one row per confirmed fixation:

```
video_name,conditions,participant,fixation_index,object
,,P0,0,cup
,,P0,1,plate
,,P0,2,knife
```

`fixation_index` is that participant's fixation order (0-based), and `object` is
the fixated class. Feature plugins add their own streams the same way when
enabled -- for example `{stem}_novel_salience_events.csv`,
`{stem}_eye_movement_events.csv`, `{stem}_pupillometry_timeseries.csv`, and
`{stem}_pupillometry_blinks.csv`. Each is written only when it has data.

## Per-frame events CSV

`{stem}_Events.csv` is one row per gaze-object hit per frame. If a
participant's gaze ray intersects two objects in the same frame, that is two
rows; if nobody hits any object on a frame, that frame contributes no rows.
Its columns are:

```
frame,t_seconds,face_idx,object,object_conf,bbox_x1,bbox_y1,bbox_x2,bbox_y2,joint_attention,joint_attention_confirmed,participant_label,gaze_conf,gaze_pitch,gaze_yaw,ray_end_x,ray_end_y,depth_at_hit,ray_snapped,ray_extended
```

`t_seconds` is the frame's timestamp, and `joint_attention` /
`joint_attention_confirmed` are per-frame flags for raw and temporally-confirmed
joint attention. In project mode a `video_name` and `conditions` column are
prepended so rows from many videos stack.

The columns after `participant_label` are new in v1.1 and strictly
**additive** -- the original twelve columns are unchanged, so positional
consumers of 1.0 files keep working. `gaze_conf` is the per-face gaze
confidence, `gaze_pitch`/`gaze_yaw` are the smoothed gaze angles in degrees,
`ray_end_x`/`ray_end_y` is the finalized ray endpoint in pixels,
`depth_at_hit` is the normalized scene depth sampled at the endpoint (empty
unless depth estimation is on), and `ray_snapped`/`ray_extended` flag whether
object snap or reach extension shaped that ray.

**When to use it.** Anything needing frame-level hit granularity -- custom
downstream analysis, validation against human coding, transition extraction.
If you only care about aggregate look-time per object, the summary CSV already
has it. In project mode this file is written automatically; in CLI
single-video mode you request it with `--log PATH`.

## Per-frame gaze stream

`{stem}_gaze.csv` (new in v1.1) is the densest table: **one row per visible
face per frame, hits or not** -- the record of where every gaze ray pointed
even when it hit nothing. It is written alongside the summary whenever the
summary is enabled.

```
video_name,conditions,frame,t_seconds,face_idx,participant_label,gaze_conf,gaze_pitch,gaze_yaw,origin_x,origin_y,ray_end_x,ray_end_y,ray_snapped,ray_extended,trust,accepted_inference,inout_score,depth_at_end,hit_objects
```

`origin_x/y` is the ray origin (eye centre), `ray_end_x/y` the finalized
endpoint. The three blend-telemetry columns expose the Gaze-LLE blend's
internals per face and frame: `trust` is the fixation likelihood driving the
blend, `accepted_inference` is 1 on the frames where a fresh Gaze-LLE heatmap
was accepted for that face, and `inout_score` is the model's
in/out-of-frame estimate for the most recent inference (1.0 when the loaded
variant has no in/out head). `hit_objects` lists the classes hit that frame,
semicolon-joined, empty when the gaze landed on nothing.

**When to use it.** Gaze-trajectory analysis, accuracy evaluation against
hand-labeled frames, and any question about frames *without* hits (aversion,
off-target scanning). The blend columns are also the first place to look when
tuning: they show when and how much the Gaze-LLE signal was trusted.

## Provenance manifest

Each run writes a `{stem}_manifest.json` beside the summary. It records exactly
what produced that run's data: the full resolved configuration, a run-identity
hash, the model-weight hashes, and the library versions in play. Because
everything that affects the result is captured, a run is auditable after the
fact -- if two runs of the "same" study disagree, the manifests tell you what
actually differed.

The manifest is written **only when at least one file output is configured**. A
pure on-screen preview run (no summary, log, video, heatmap, or charts) writes
no manifest, because there is no data file to make traceable.

**When to use it.** Keep it. You may never open it by hand, but it is what lets
you (or a reviewer) trust a result months later.

## Heatmaps

One image per participant, accumulated from that participant's gaze-ray
endpoints over the whole video, Gaussian-blurred and blended over a background
frame. A participant with no collected gaze data produces no heatmap -- check
the console if one is missing. The background is a single frame, so a scene that
changes substantially over the video will not be fully reflected.

**When to use it.** A quick, human-legible picture of where attention
concentrated. Not a substitute for the CSVs in analysis.

## Time-series charts

Post-run charts showing phenomena metrics over time -- joint-attention
percentage, gaze-following events, and so on -- one multi-panel image per video
plus per-tracker images. Only trackers that expose time-series data contribute;
some lighter trackers do not.

**When to use it.** A visual sanity check, or to see *when* in a video a
phenomenon spiked. For statistics, use the underlying CSVs.

## Annotated video

The source video re-rendered with overlays: object bounding boxes colour-coded
by attention state, face rectangles with participant labels, gaze rays with
state-aware endpoints, and phenomena dashboard panels. It is the slowest output
by a wide margin.

**When to use it.** To confirm that detection and ray forming are doing what you
think -- indispensable during parameter tuning. Turn it off for production
batches once you trust the pipeline. Face anonymization, when enabled, blurs or
blacks out faces here (and in heatmap backgrounds).

## Project-mode aggregates

When you process a whole project rather than a single video, MindSight also
concatenates each per-video tidy table into a project-level file and, if your
videos carry condition tags, splits each one by condition. The global files
(written into `Outputs/CSV Files/`) are:

- `Global_summary.csv` -- every video's summary stacked.
- `Global_Events.csv` -- every video's per-frame events stacked.
- `Global_gaze.csv` -- every video's per-frame gaze stream stacked.
- `Global_phenomena_events.csv` -- every video's episode stream stacked.
- `Global_scanpath.csv` -- every video's fixation stream stacked.
- `Global_novel_salience_events.csv`, `Global_eye_movement_events.csv`,
  `Global_pupillometry_timeseries.csv`, `Global_pupillometry_blinks.csv` --
  the corresponding feature-plugin streams, each written only when at least one
  video produced that stream.

Because every tidy table already carries `video_name` and `conditions` as its
first two columns, concatenation is a pure append with one header. When
conditions are defined, `Outputs/By Condition/` additionally holds one
`{tag}<suffix>.csv` per unique condition tag (e.g. `Emotional_summary.csv`), a
video with several pipe-delimited tags appearing under each. These are what most
cross-video analyses should pull from.

## The resume ledger

Project runs resume by default. MindSight keeps a per-batch ledger at
`Outputs/_run/ledger.json` recording each video's status, so an interrupted or
re-launched batch skips work that is already complete. Forcing a full reprocess
(`--no-resume`) reprocesses every video in place. The ledger is bookkeeping, not
analysis data -- you do not hand it to an analyst.

## Where the files land

The default directory layout differs slightly between a CLI single-video run and
a project-mode run. Both are shown below.

**CLI single-video run** (paths from the bare `--summary` / `--save` /
`--heatmap` defaults; `--log` needs an explicit path):

```
Outputs/
├── CSV Files/
│   ├── {stem}_summary.csv              # tidy summary
│   ├── {stem}_phenomena_events.csv     # merged episode stream
│   ├── {stem}_scanpath.csv             # per-tracker streams (when produced)
│   └── {stem}_manifest.json            # provenance manifest
├── Video/                              # annotated video ({stem}_Video_Output.mp4)
├── heatmaps/
│   └── {stem}_Heatmap_Output/          # one PNG per participant
└── Charts/                             # time-series charts
```

**Project-mode run** (under the project's `Outputs/`):

```
Outputs/
├── CSV Files/
│   ├── {stem}_summary.csv              # per video
│   ├── {stem}_Events.csv               # per video (per-frame events)
│   ├── {stem}_phenomena_events.csv     # per video (episode stream)
│   ├── {stem}_scanpath.csv             # per video (when produced)
│   ├── {stem}_manifest.json            # per video
│   ├── Global_summary.csv              # project aggregate
│   ├── Global_Events.csv
│   ├── Global_phenomena_events.csv
│   └── Global_scanpath.csv             # (+ feature-plugin globals)
├── Videos/                             # annotated videos (plural)
├── {stem}_Heatmap/                     # per-participant heatmaps (one dir per video)
├── Charts/                             # time-series charts
├── By Condition/                       # per-condition splits (when conditions defined)
└── _run/
    └── ledger.json                     # resume bookkeeping
```

!!! note
    The two key differences are the annotated-video directory (`Video/` singular
    in CLI mode, `Videos/` plural in project mode) and the heatmap directory
    (`heatmaps/{stem}_Heatmap_Output/` in CLI mode, `{stem}_Heatmap/` directly
    under `Outputs/` in project mode). Run-folder projects gather each run's
    per-video files under `Outputs/Runs/<run_id>/` instead, named by run id; the
    `Global_*.csv`, `By Condition/`, and `_run/` locations are unchanged.

## What to hand to an analyst

For most studies the handoff is small: the **summary CSV** (or the project-mode
`Global_summary.csv` / `By Condition/` files), the **phenomena-events CSV** when
timing or duration matters, plus the **provenance manifest** so the numbers are
traceable. Include the per-frame **events CSV** only when the analysis needs
frame-level detail. Heatmaps, charts, and the annotated video are for review and
communication, not for statistics.
