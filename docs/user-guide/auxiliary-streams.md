# Auxiliary Streams

Auxiliary streams are optional per-participant video feeds -- such as a dedicated eye-tracking camera or a first-person-view recording -- that run alongside the main source video. They are frame-synchronised with the main video and exposed to plugins via the FrameContext, but are **not** processed by any built-in pipeline stage.

---

## When to Use Auxiliary Streams

Auxiliary streams are useful when your experiment captures multiple camera angles per participant. Common examples:

- **Eye camera** -- A close-up camera recording a participant's eye movements.
- **First-person view** -- A head-mounted camera showing what the participant sees.
- **Scene camera** -- An additional wide-angle view of the environment.

Plugins can read these frames to perform specialised analysis (e.g., pupil dilation measurement from an eye camera feed).

---

## Configuration

There are four ways to define auxiliary streams, listed from simplest to most flexible.

### CLI Flags

Use `--aux-stream` (repeatable) with format `PID:TYPE:SOURCE`:

```bash
python MindSight.py --source main_video.mp4 \
  --aux-stream "S70:eye_camera:/data/s70_eye.mp4" \
  --aux-stream "S71:first_person_view:/data/s71_fpv.mp4"
```

| Component | Description |
|-----------|-------------|
| `PID` | Participant label (must match your participant ID assignments) |
| `TYPE` | Stream purpose tag (arbitrary string, e.g. `eye_camera`, `first_person_view`) |
| `SOURCE` | File path to the auxiliary video |

### Pipeline YAML

Add an `aux_streams` section to your `pipeline.yaml`:

```yaml
aux_streams:
  - pid: "S70"
    stream_type: eye_camera
    source: /data/s70_eye.mp4
  - pid: "S71"
    stream_type: first_person_view
    source: /data/s71_fpv.mp4
```

### Participant CSV

Include optional `aux_stream_type` and `aux_stream_source` columns in your `participant_ids.csv`:

```csv
track_id,participant_label,aux_stream_type,aux_stream_source
1,S70,eye_camera,/data/s70_eye.mp4
2,S71,first_person_view,/data/s71_fpv.mp4
```

### Auto-Discovery (Project Mode)

In project mode, place auxiliary videos in a conventional directory structure under your project's `Inputs/` folder:

```
Inputs/AuxStreams/
├── eye_camera/
│   ├── S70.mp4
│   └── S71.mp4
└── first_person_view/
    └── S70.mp4
```

The subdirectory name becomes the `stream_type` and the file stem becomes the `pid`. All four sources are merged -- CLI flags override YAML values when both are provided.

---

## Synchronisation

Auxiliary streams are read **every frame** alongside the main video to stay in sync. MindSight assumes aux streams have the same frame rate as the main source. If the FPS differs by more than 1.0, a warning is logged:

```
Warning: aux stream S70:eye_camera FPS (60.0) differs from main (30.0) -- frames may drift
```

If an aux stream ends before the main video, a one-time warning is printed and subsequent reads for that stream return `None`.

---

## Accessing Aux Frames in Plugins

Auxiliary frames are available in the FrameContext under the `aux_frames` key:

```python
def process_frame(frame, aux_frames, **kwargs):
    # Access a specific stream
    eye_frame = aux_frames.get(("S70", "eye_camera"))
    if eye_frame is not None:
        # Process the eye camera frame
        ...

    # Iterate all streams
    for (pid, stream_type), frame_data in aux_frames.items():
        if frame_data is not None:
            ...
```

The key is a `(pid, stream_type)` tuple and the value is either a NumPy ndarray (the video frame) or `None` if the stream has ended or failed to read.

!!! note
    Built-in pipeline stages (detection, gaze, phenomena, data collection) do not read auxiliary frames. Only plugins that explicitly request `aux_frames` will receive them.

---

## Error Handling

| Condition | Behaviour |
|-----------|-----------|
| Aux stream file cannot be opened | Warning logged, stream skipped |
| FPS mismatch with main video | Warning logged, processing continues |
| Aux stream ends before main video | `None` returned for that stream, one-time warning |
| Invalid `--aux-stream` format | `ValueError` raised at startup |
