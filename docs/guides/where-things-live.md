# Where things live

MindSight keeps files in **two distinct places**, and knowing which is which
saves a lot of confusion. This page is the map.

!!! note "Two locations, don't mix them up"
    - **The install root** (`~/MindSight` on macOS, `%LOCALAPPDATA%\MindSight` on
      Windows) holds the **application itself** -- its private Python, the model
      weights, and the default data folders.
    - **The app-state directory** (`~/.mindsight`, on both platforms) holds your
      **personal app settings** -- presets, last-used session, recent projects.

    They are separate on purpose: reinstalling or moving the app touches the
    install root; it does not disturb your settings in `~/.mindsight`.

---

## The install root

The installer puts everything under one folder:

- **macOS** -- `~/MindSight` (your home folder's `MindSight` directory).
- **Windows** -- `%LOCALAPPDATA%\MindSight` (usually
  `C:\Users\<you>\AppData\Local\MindSight`).

```
MindSight/                 # the install root
├── venv/                  # a private Python + all of MindSight's dependencies
└── app/                   # the application data home
    ├── Weights/           # model weights, one subfolder per backend (Weights/YOLO/, Weights/MGaze/, ...)
    ├── Outputs/           # default output location for runs that don't target a project
    ├── Projects/          # the sample ExampleStudy and any projects you keep here
    └── ...                # the app source and assets
```

Weight resolution is **global**: a bare filename like `yolov8n.pt` resolves to
`Weights/<backend>/yolov8n.pt` for every project on the machine, which is why the
**Models** tab and preflight talk about one shared `Weights/` folder rather than
a per-project one.

---

## The app-state directory

Your personal settings live in `~/.mindsight` (a hidden folder in your home
directory, on both macOS and Windows):

| Path | What it holds |
|------|---------------|
| `~/.mindsight/presets/` | Named setting bundles saved via **File > Save Preset...**. |
| `~/.mindsight/last_used.json` | The last session's settings, restored on the next launch. |
| `~/.mindsight/recent_projects.json` | The recent-projects list shown on the Projects tab. |
| `~/.mindsight/run_settings.json` | The persisted Inference Settings store (the dialog's state). |

Because this is separate from the install root, your presets and recent-projects
list survive an app reinstall.

---

## A project folder

A study project is a self-contained folder (see
[Projects and sessions](projects-and-sessions.md)). The sample
`Projects/ExampleStudy/` shows the shape:

```
MyProject/
├── project.yaml            # points at the pipeline preset; holds participants/conditions
├── notes.md                # free-form study notes (shown on the Projects tab)
├── Pipeline/
│   └── pipeline.yaml       # the pipeline preset this study runs with
├── Inputs/
│   ├── Runs/               # one subfolder per session: Runs/<run_id>/ with one video + run.yaml
│   │                       #   (a Runs/<run_id>/ with run.yaml but NO video is a planned session)
│   ├── Videos/             # (legacy flat layout) recordings directly in a folder
│   └── Prompts/            # the study's .vp.json visual prompt, if used
└── Outputs/                # created when you Run
    ├── Runs/<run_id>/      # per-run CSVs and outputs (run-folder projects)
    ├── CSV Files/          # per-video CSVs + the project-wide Global_* aggregates (flat projects)
    ├── Videos/             # annotated output videos
    ├── heatmaps/           # heatmap images
    └── _run/
        └── ledger.json     # the resume ledger: what's done, and archived prior outputs
```

A project uses **one** input layout: **run-folder** (`Inputs/Runs/<run_id>/`,
what the wizard builds and the recommended layout) **or** the legacy **flat**
layout (`Inputs/Videos/`). Do not mix both -- preflight stops you if a project
has videos in both places.

The `_run/ledger.json` file is what makes runs resumable: it records, per
recording, whether it is done and unchanged, and it archives superseded outputs
rather than deleting them.

---

## Quick-analysis outputs

Runs done **without a project** -- Video File mode, or a Camera quick-run -- do
not use the project tree. They write to a **folder you choose** (prefilled next
to the source in Video File mode, and editable). A Camera quick-run also drops a
`<run_id>_session.yaml` sidecar next to its outputs, carrying the session's
metadata so the recording can be imported into a project later.

---

## See also

- [Projects and sessions](projects-and-sessions.md) -- creating and managing the
  project folders described above.
- [Understanding the outputs](../concepts/outputs.md) -- what each output file
  contains.
- [Analyze footage](analyze-footage.md) -- where quick-analysis outputs come
  from.
