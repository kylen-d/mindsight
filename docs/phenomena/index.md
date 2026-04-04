# Phenomena Reference

MindSight can detect and quantify a set of **gaze-based social phenomena** from video. Each phenomenon is an observable pattern of visual attention -- such as two people making eye contact or a group fixating on the same object -- that carries meaning in social cognition research.

Phenomena tracking runs on top of the core gaze-estimation and object-detection pipelines. You enable individual phenomena with their CLI flags, or activate all of them at once with `--all-phenomena`.

## Phenomena Overview

| Phenomenon | Description | CLI Flag | Page |
|---|---|---|---|
| Joint Attention | Multiple participants simultaneously fixating on the same object | `--joint-attention` | [joint-attention.md](joint-attention.md) |
| Mutual Gaze | Two participants looking directly at each other (eye contact) | `--mutual-gaze` | [mutual-gaze.md](mutual-gaze.md) |
| Social Referencing | A participant looks at another person's face then redirects gaze to an object | `--social-ref` | [social-referencing.md](social-referencing.md) |
| Gaze Following | A participant shifts gaze in the same direction another person is looking | `--gaze-following` | gaze-following.md |
| Gaze Leadership | Identifying which participant initiates shared attention episodes | `--gaze-leadership` | gaze-leadership.md |
| Gaze Aversion | A participant actively avoids eye contact by looking away | `--gaze-aversion` | gaze-aversion.md |
| Attention Span | Duration a participant sustains fixation on a single target | `--attention-span` | attention-span.md |
| Scanpath | The sequential trajectory of a participant's gaze across objects | `--scanpath` | scanpath.md |

## Enabling Phenomena

Enable a single phenomenon:

```bash
python MindSight.py --source video.mp4 --joint-attention
```

Enable several at once:

```bash
python MindSight.py --source video.mp4 --mutual-gaze --social-ref
```

Enable every phenomenon:

```bash
python MindSight.py --source video.mp4 --all-phenomena
```

Each phenomenon may expose additional tuning parameters (window sizes, thresholds, quorum fractions). See the individual pages for details.
