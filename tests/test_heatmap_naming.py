"""Heatmap inner-PNG naming (SP3.1 P3 blessing amendment).

The user-ruled run-folder output shape requires the PNGs INSIDE
``Outputs/Runs/<run_id>/<run_id>_Heatmap/`` to use the run's output stem
(the run_id), not the source-video stem.  ``save_heatmaps`` grew an
explicit ``stem`` parameter; the legacy/single-source fallback (source
stem) must stay byte-compatible.
"""
import numpy as np

from mindsight.outputs.heatmap_output import save_heatmaps

BG = np.zeros((32, 32, 3), dtype=np.uint8)
GAZE = {0: [(5, 5), (6, 6)], 1: [(10, 10)]}


def _names(dirpath):
    return sorted(p.name for p in dirpath.iterdir())


def test_directory_case_default_uses_source_stem(tmp_path):
    out = tmp_path / "session_Heatmap"
    save_heatmaps(str(out), "/videos/session.mp4", BG, GAZE)
    assert _names(out) == ["session_P0_heatmap.png", "session_P1_heatmap.png"]


def test_directory_case_explicit_stem_wins(tmp_path):
    out = tmp_path / "dyad07_collab_Heatmap"
    save_heatmaps(str(out), "/runs/dyad07_collab/session.mp4", BG, GAZE,
                  pid_map={0: "S70", 1: "S71"}, stem="dyad07_collab")
    assert _names(out) == ["dyad07_collab_S70_heatmap.png",
                           "dyad07_collab_S71_heatmap.png"]


def test_empty_stem_falls_back_to_source(tmp_path):
    out = tmp_path / "a_Heatmap"
    save_heatmaps(str(out), "/videos/a.mp4", BG, GAZE, stem="")
    assert _names(out) == ["a_P0_heatmap.png", "a_P1_heatmap.png"]


def test_legacy_project_stem_equals_source_stem_is_noop(tmp_path):
    # Legacy project mode passes stem == run_output_stem == source stem;
    # names must be identical to the no-stem call.
    out1 = tmp_path / "one" / "a_Heatmap"
    out2 = tmp_path / "two" / "a_Heatmap"
    save_heatmaps(str(out1), "/videos/a.mp4", BG, GAZE)
    save_heatmaps(str(out2), "/videos/a.mp4", BG, GAZE, stem="a")
    assert _names(out1) == _names(out2)


def test_webcam_source_fallback(tmp_path):
    out = tmp_path / "webcam_Heatmap"
    save_heatmaps(str(out), 0, BG, {0: [(1, 1)]})
    assert _names(out) == ["webcam_P0_heatmap.png"]
