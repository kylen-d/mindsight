# Testing

## Test Suite

MindSight uses **pytest** for its test suite. All test files live in the `tests/`
directory at the project root — there are **more than 60** of them, from geometry
unit tests through full-pipeline integration runs.

Heavy tests that load real model weights or run whole clips are tagged with the
`slow` marker (declared in `pyproject.toml` under `[tool.pytest.ini_options]`).
Deselect them for a fast inner-loop run with `-m "not slow"`.

## Representative Tests

A few of the many test files, to show the range:

| File | What it covers |
|------|----------------|
| `test_geometry.py` | Ray intersection math, pitch/yaw conversions, and coordinate transforms |
| `test_frame_context.py` | FrameContext API -- creation, attribute access, and data attachment |
| `test_config_compat.py` | YAML loading, key mapping, CLI-override precedence |
| `test_phenomena_trackers.py` | Tracker `update()` calls, output format, and edge cases |

## Running Tests

Run the full suite:

```bash
pytest tests/
```

Skip the heavy integration tests for a fast inner loop:

```bash
pytest tests/ -m "not slow"
```

Run a single file with verbose output:

```bash
pytest tests/test_geometry.py -v
```

Run a specific test by name:

```bash
pytest tests/test_phenomena_trackers.py -k "test_mutual_gaze" -v
```

## Writing Tests for Plugins

Start from the copy-and-edit skeleton at `Plugins/TEMPLATE/my_plugin.py`, then
create a test such as `tests/test_my_plugin.py`. Use the **real** `update()` kwarg
names (the phenomena engine passes `frame_no`, `persons_gaze`, `dets`, etc. — see
[Plugin Base Classes](../reference/plugin-base-classes.md)), not the pre-1.0
`frame_idx` / `person_gazes` / `detections` vocabulary:

```python
import numpy as np
import pytest
from Plugins.Phenomena.MyPlugin.my_plugin import MyPhenomenonTracker  # Plugins stay top-level


@pytest.fixture
def tracker():
    """Construct the tracker with test-friendly parameters."""
    return MyPhenomenonTracker(threshold=0.5, window=5)


def test_update_returns_expected_keys(tracker):
    """Call update() with minimal real kwargs and check the output dict."""
    result = tracker.update(
        frame_no=0,
        persons_gaze=[(np.array([100.0, 200.0]),      # origin
                       np.array([300.0, 220.0]),      # ray_end
                       (0.3, -0.1))],                 # (pitch, yaw)
        face_track_ids=[0],
        hits=set(),
        dets=[],
        n_faces=1,
    )
    assert isinstance(result, dict)


def test_no_crash_on_empty_input(tracker):
    """Ensure the tracker handles an empty frame gracefully."""
    result = tracker.update(
        frame_no=0, persons_gaze=[], face_track_ids=[],
        hits=set(), dets=[], n_faces=0,
    )
    assert result is not None
```

Key points:

- Import your plugin class directly.
- Construct it with explicit test parameters so tests are deterministic.
- Call `update()` with keyword arguments that mirror what the phenomena engine
  actually provides (`frame_no`, `persons_gaze` as `(origin, ray_end, angles)`
  tuples, `hits`, `dets`, `n_faces`, `face_track_ids`, ...).
- Assert on the shape and content of the returned data, not on internal state.
