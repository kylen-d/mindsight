"""SP2.1 Batch E: the resume ledger (mindsight.project.ledger).

Covers the decide matrix (skip / redo / redo_archive), atomic-write resilience
to a torn/garbage tmp or a corrupt ledger.json, supersede archiving moving the
right files, and video_hash sensitivity to mtime / pid_map / conditions / aux.
All fast (pure stdlib, tmp files).
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

from mindsight.project.ledger import Ledger, compute_video_hash

H1 = ("cfg_aaa", "vid_111")


def _src(tmp_path, name="a.mp4", content=b"video-bytes"):
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# decide matrix
# ══════════════════════════════════════════════════════════════════════════════

def test_decide_absent_is_redo(tmp_path):
    led = Ledger.load(tmp_path)
    assert led.decide("a.mp4", H1) == "redo"


def test_decide_in_progress_is_redo(tmp_path):
    led = Ledger.load(tmp_path)
    led.mark_started("a.mp4", H1, {"summary": str(tmp_path / "a.csv")})
    # in_progress (a kill -9 mid-run) must reprocess, never skip.
    assert led.decide("a.mp4", H1) == "redo"


def test_decide_error_is_redo(tmp_path):
    led = Ledger.load(tmp_path)
    led.mark_started("a.mp4", H1, {})
    led.mark_error("a.mp4", "boom")
    assert led.decide("a.mp4", H1) == "redo"


def test_decide_done_match_is_skip(tmp_path):
    led = Ledger.load(tmp_path)
    led.mark_started("a.mp4", H1, {})
    led.mark_done("a.mp4", str(tmp_path / "a_manifest.json"))
    assert led.decide("a.mp4", H1) == "skip"


def test_decide_done_config_mismatch_is_redo_archive(tmp_path):
    led = Ledger.load(tmp_path)
    led.mark_started("a.mp4", H1, {})
    led.mark_done("a.mp4")
    assert led.decide("a.mp4", ("cfg_DIFFERENT", "vid_111")) == "redo_archive"


def test_decide_done_video_mismatch_is_redo_archive(tmp_path):
    led = Ledger.load(tmp_path)
    led.mark_started("a.mp4", H1, {})
    led.mark_done("a.mp4")
    assert led.decide("a.mp4", ("cfg_aaa", "vid_DIFFERENT")) == "redo_archive"


# ══════════════════════════════════════════════════════════════════════════════
# atomic persistence
# ══════════════════════════════════════════════════════════════════════════════

def test_every_transition_leaves_valid_ledger(tmp_path):
    led = Ledger.load(tmp_path)
    path = tmp_path / "_run" / "ledger.json"
    led.mark_started("a.mp4", H1, {"summary": "x"})
    data = json.loads(path.read_text())
    assert data["ledger_version"] == 1
    assert data["videos"]["a.mp4"]["status"] == "in_progress"
    led.mark_done("a.mp4", "m.json")
    assert json.loads(path.read_text())["videos"]["a.mp4"]["status"] == "done"


def test_no_tmp_litter_after_write(tmp_path):
    led = Ledger.load(tmp_path)
    led.mark_started("a.mp4", H1, {})
    run_dir = tmp_path / "_run"
    leftovers = [p for p in run_dir.iterdir() if ".tmp." in p.name]
    assert leftovers == []


def test_load_survives_garbage_tmp(tmp_path):
    """A torn tmp from a killed process must not corrupt a good ledger.json."""
    led = Ledger.load(tmp_path)
    led.mark_started("a.mp4", H1, {})
    led.mark_done("a.mp4")
    run_dir = tmp_path / "_run"
    # Simulate a process killed BETWEEN writing the tmp and os.replace.
    (run_dir / f"ledger.json.tmp.{os.getpid()}").write_text("{ NOT valid json")
    reloaded = Ledger.load(tmp_path)
    assert reloaded.decide("a.mp4", H1) == "skip"  # good ledger.json still read


def test_load_corrupt_ledger_starts_clean(tmp_path):
    run_dir = tmp_path / "_run"
    run_dir.mkdir(parents=True)
    (run_dir / "ledger.json").write_text("{ half-written")
    led = Ledger.load(tmp_path)
    assert led.decide("a.mp4", H1) == "redo"  # empty -> reprocess everything


# ══════════════════════════════════════════════════════════════════════════════
# supersede archiving
# ══════════════════════════════════════════════════════════════════════════════

def test_archive_moves_outputs_and_manifest(tmp_path):
    csv_dir = tmp_path / "CSV Files"
    csv_dir.mkdir()
    summary = csv_dir / "a_summary.csv"
    log = csv_dir / "a_Events.csv"
    heat = tmp_path / "a_Heatmap"
    heat.mkdir()
    (heat / "a_P0_heatmap.png").write_bytes(b"png")
    manifest = csv_dir / "a_manifest.json"
    for f in (summary, log, manifest):
        f.write_text("data")

    led = Ledger.load(tmp_path)
    led.mark_started("a.mp4", H1, {
        "summary": str(summary), "log": str(log), "heatmap": str(heat)})
    led.mark_done("a.mp4", str(manifest))

    dest = led.archive("a.mp4")
    assert dest is not None
    # Originals gone, copies present under superseded/<stamp>_a/.
    assert not summary.exists() and not log.exists() and not heat.exists()
    assert not manifest.exists()
    assert (dest / "a_summary.csv").read_text() == "data"
    assert (dest / "a_Events.csv").exists()
    assert (dest / "a_manifest.json").exists()
    assert (dest / "a_Heatmap" / "a_P0_heatmap.png").exists()
    assert dest.parent.name == "superseded"


def test_archive_absent_video_returns_none(tmp_path):
    led = Ledger.load(tmp_path)
    assert led.archive("ghost.mp4") is None


def test_archive_skips_missing_files(tmp_path):
    led = Ledger.load(tmp_path)
    led.mark_started("a.mp4", H1, {"summary": str(tmp_path / "gone.csv")})
    led.mark_done("a.mp4")
    # No output file exists -> nothing to move.
    assert led.archive("a.mp4") is None


# ══════════════════════════════════════════════════════════════════════════════
# video_hash sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def test_video_hash_stable(tmp_path):
    src = _src(tmp_path)
    a = compute_video_hash(src, pid_map={0: "P0"}, conditions="GroupA")
    b = compute_video_hash(src, pid_map={0: "P0"}, conditions="GroupA")
    assert a == b


def test_video_hash_changes_on_mtime(tmp_path):
    src = _src(tmp_path)
    before = compute_video_hash(src)
    # Bump mtime_ns (and size) by rewriting -> re-encode invalidates the skip.
    os.utime(src, ns=(0, 0))
    after = compute_video_hash(src)
    assert before != after


def test_video_hash_changes_on_pid_map(tmp_path):
    src = _src(tmp_path)
    assert (compute_video_hash(src, pid_map={0: "P0"})
            != compute_video_hash(src, pid_map={0: "PX"}))


def test_video_hash_changes_on_conditions(tmp_path):
    src = _src(tmp_path)
    assert (compute_video_hash(src, conditions="GroupA")
            != compute_video_hash(src, conditions="GroupB"))


def test_video_hash_changes_on_aux_streams(tmp_path):
    src = _src(tmp_path)
    aux = [SimpleNamespace(source="e.mp4", video_type="eye_only",
                           stream_label="eye_S70", participants=["S70"])]
    assert compute_video_hash(src) != compute_video_hash(src, aux_streams=aux)
