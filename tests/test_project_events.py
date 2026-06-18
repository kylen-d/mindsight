"""Fast constructor + immutability tests for the ProjectEvent union (SP3.1 D1)."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mindsight.project.events import (
    BatchDone,
    BatchStarted,
    ProjectEvent,
    VideoArchived,
    VideoDone,
    VideoError,
    VideoFrame,
    VideoSkipped,
    VideoStarted,
)


def test_batch_started_fields():
    ev = BatchStarted(total=3, out_root=Path("/tmp/Out"))
    assert ev.total == 3
    assert ev.out_root == Path("/tmp/Out")


def test_video_started_fields():
    src = Path("/proj/Inputs/Videos/a.mp4")
    ev = VideoStarted(index=1, total=2, run_id="a.mp4", source=src)
    assert (ev.index, ev.total, ev.run_id, ev.source) == (1, 2, "a.mp4", src)


def test_video_frame_carries_result():
    sentinel = object()
    ev = VideoFrame(run_id="a.mp4", result=sentinel)
    assert ev.run_id == "a.mp4"
    assert ev.result is sentinel


def test_video_skipped_reason():
    ev = VideoSkipped(run_id="a.mp4", reason="done, config unchanged")
    assert ev.reason == "done, config unchanged"


def test_video_archived_dest_may_be_none():
    assert VideoArchived(run_id="a.mp4", dest=None).dest is None
    dest = Path("/proj/Outputs/_run/superseded/x_a")
    assert VideoArchived(run_id="a.mp4", dest=dest).dest == dest


def test_video_done_manifest():
    ev = VideoDone(run_id="a.mp4", manifest_path="/proj/a_manifest.json")
    assert ev.manifest_path.endswith("a_manifest.json")


def test_video_error_text():
    assert VideoError(run_id="a.mp4", error="boom").error == "boom"


def test_batch_done_out_root():
    assert BatchDone(out_root=Path("/tmp/Out")).out_root == Path("/tmp/Out")


def test_events_are_frozen():
    ev = BatchStarted(total=1, out_root=Path("/tmp"))
    with pytest.raises(FrozenInstanceError):
        ev.total = 2  # type: ignore[misc]


def test_union_membership():
    events = [
        BatchStarted(1, Path(".")),
        VideoStarted(1, 1, "a", Path("a")),
        VideoFrame("a", None),
        VideoSkipped("a", "r"),
        VideoArchived("a", None),
        VideoDone("a", "m"),
        VideoError("a", "e"),
        BatchDone(Path(".")),
    ]
    # ProjectEvent is a typing Union; every member is one of its arms.
    arms = ProjectEvent.__args__
    for ev in events:
        assert type(ev) in arms
