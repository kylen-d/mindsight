# Testing

## Test Suite

MindSight uses **pytest** for its test suite. All test files live in the `tests/` directory at the project root.

## Available Tests

| File | What it covers |
|------|----------------|
| `test_geometry.py` | Ray intersection math, pitch/yaw conversions, and coordinate transforms |
| `test_frame_context.py` | FrameContext API -- creation, attribute access, and data attachment |
| `test_pipeline_loader.py` | YAML loading, key mapping, CLI-override precedence |
| `test_phenomena_trackers.py` | Tracker `update()` calls, output format, and edge cases |

## Running Tests

Run the full suite:

```bash
pytest tests/
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

To test a custom plugin, create a file such as `tests/test_my_plugin.py` and follow this pattern:

```python
import pytest
from Plugins.Phenomena.MyPlugin.my_plugin import MyPhenomenonTracker  # Plugins stay top-level


@pytest.fixture
def tracker():
    """Construct the tracker with test-friendly parameters."""
    return MyPhenomenonTracker(threshold=0.5, window=5)


def test_update_returns_expected_keys(tracker):
    """Call update() with minimal kwargs and check the output dict."""
    result = tracker.update(
        frame_idx=0,
        person_gazes={1: (100, 200, 0.3, -0.1)},
        detections=[],
    )
    assert isinstance(result, dict)
    assert "metric" in result


def test_no_crash_on_empty_input(tracker):
    """Ensure the tracker handles an empty frame gracefully."""
    result = tracker.update(
        frame_idx=0,
        person_gazes={},
        detections=[],
    )
    assert result is not None
```

Key points:

- Import your plugin class directly.
- Construct it with explicit test parameters so tests are deterministic.
- Call `update()` with mock keyword arguments that mirror what the phenomena engine provides.
- Assert on the shape and content of the returned data, not on internal state.
