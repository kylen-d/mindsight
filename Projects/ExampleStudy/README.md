# Example Study

A minimal, generic study-project skeleton showing the folder shape MindSight
expects. Copy this folder, rename it for your study, and fill in the parts
marked "you provide".

## What's here

```
ExampleStudy/
  project.yaml              # points at the shipped known-good pipeline preset
  Inputs/
    Runs/                   # one sub-folder per recording session (run-folder layout)
    Prompts/                # your study's visual prompt (.vp.json) goes here
  Outputs/                  # MindSight writes CSVs and videos here (created on run)
```

## What you provide

- **Videos.** This skeleton uses the run-folder layout. For each recording
  session, create `Inputs/Runs/<run_id>/` and place exactly one primary video
  inside it. (The flat layout -- dropping videos straight into
  `Inputs/Videos/` -- also works, but use one layout or the other, not both.)

- **Visual prompt.** The known-good preset uses a YOLOE open-vocabulary
  detector, which is meant to run with a visual prompt. Export your study's
  prompt from the VP Builder and save it as a `.vp.json` file in
  `Inputs/Prompts/`. The prompt's reference image should be at the same
  resolution as your videos. Without a prompt the pipeline still runs, but
  detection quality on classroom footage will be lower.

## Pipeline preset

`project.yaml` references `configs/pipeline_known_good.yaml` -- a preset whose
values were validated on classroom-style dyadic footage (see the header of
that file and `configs/KNOWN_GOOD.md`). CLI flags override anything set there.

## Checking your setup

Run a read-only preflight before a real run:

```
python MindSight.py --project Projects/ExampleStudy/ --preflight
```

On this empty skeleton, preflight reports the missing study data (no runs yet,
no visual prompt) -- that is expected. Once you add a run folder with a video
and a `.vp.json` prompt, those rows clear.
