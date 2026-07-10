"""Analyze Footage output panel (SP3.1 Batch G fix-forward, G-ENH-4).

Pure readers (mindsight/GUI/run_outputs.py, no Qt) + the tabbed Log | Charts |
Output CSVs panel.  Everything renders from CSVs the pipeline ALREADY wrote
(D11 consume-don't-compute); nothing here may add files to a project's
Outputs/ tree (P2/P3 identity, T1).
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_SUMMARY = """\
video_name,conditions,phenomenon,participant,partner,object,metric,value
a,GroupA,object_look_time,S70,,chair,frames_active,28
a,GroupA,object_look_time,S70,,chair,pct_of_video,3.2
a,GroupA,object_look_time,S70,,dining table,pct_of_video,41.1
a,GroupA,object_look_time,S71,,chair,pct_of_video,23.7
a,GroupA,object_look_time,S71,,cup,pct_of_video,0.9
a,GroupA,other_phenomenon,S70,,chair,pct_of_video,99.0
"""

_EVENTS = """\
video_name,conditions,frame,t_seconds,face_idx,object,object_conf,bbox_x1,bbox_y1,bbox_x2,bbox_y2,joint_attention,joint_attention_confirmed,participant_label
a,GroupA,6,0.201,0,dining table,0.446,2,426,1054,668,0,0,S70
a,GroupA,7,0.234,0,chair,0.5,2,426,1054,668,0,0,S70
a,GroupA,8,0.268,1,cup,0.5,2,426,1054,668,0,0,S71
"""


def _write_run_csvs(csv_dir: Path, stem: str = "a"):
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / f"{stem}_summary.csv").write_text(_SUMMARY)
    (csv_dir / f"{stem}_Events.csv").write_text(_EVENTS)
    (csv_dir / f"{stem}_scanpath.csv").write_text("a,b\n1,2\n")


def _make_project(tmp_path, *, with_outputs=True):
    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    (proj / "Inputs" / "Videos" / "a.mp4").write_bytes(b"\x00" * 32)
    if with_outputs:
        _write_run_csvs(proj / "Outputs" / "CSV Files")
    return proj


# ── Pure readers ─────────────────────────────────────────────────────────────

def test_discover_run_outputs_legacy(tmp_path):
    from mindsight.project.project import Project
    from mindsight.GUI.run_outputs import discover_run_outputs
    proj = _make_project(tmp_path)
    outs = discover_run_outputs(Project.open(proj).runs())
    assert len(outs) == 1
    out = outs[0]
    assert out.run_id == "a.mp4" and out.stem == "a"
    assert out.summary is not None and out.events is not None
    assert [p.name for p in out.csv_paths] == [
        "a_Events.csv", "a_scanpath.csv", "a_summary.csv"]


def test_discover_skips_runs_without_csvs(tmp_path):
    from mindsight.project.project import Project
    from mindsight.GUI.run_outputs import discover_run_outputs
    proj = _make_project(tmp_path, with_outputs=False)
    assert discover_run_outputs(Project.open(proj).runs()) == []


def test_look_time_table(tmp_path):
    from mindsight.GUI.run_outputs import look_time_table
    p = tmp_path / "s.csv"
    p.write_text(_SUMMARY)
    table = look_time_table(p)
    # only object_look_time + pct_of_video rows contribute
    assert table == {
        "S70": {"chair": 3.2, "dining table": 41.1},
        "S71": {"chair": 23.7, "cup": 0.9},
    }


def test_gaze_timeline(tmp_path):
    from mindsight.GUI.run_outputs import gaze_timeline
    p = tmp_path / "e.csv"
    p.write_text(_EVENTS)
    objects, per = gaze_timeline(p)
    assert objects == ["dining table", "chair", "cup"]
    assert per["S70"] == ([0.201, 0.234], [0, 1])
    assert per["S71"] == ([0.268], [2])


def test_load_csv_rows_truncates(tmp_path):
    from mindsight.GUI.run_outputs import load_csv_rows
    p = tmp_path / "big.csv"
    p.write_text("h1,h2\n" + "\n".join(f"{i},{i}" for i in range(50)))
    header, rows, total_rows = load_csv_rows(p, max_rows=10)
    assert header == ["h1", "h2"]
    # capped view, but the FULL row count is still reported (all 50 rows).
    assert len(rows) == 10 and total_rows == 50


def test_load_csv_rows_untruncated_reports_full_count(tmp_path):
    from mindsight.GUI.run_outputs import load_csv_rows
    p = tmp_path / "small.csv"
    p.write_text("h1,h2\n" + "\n".join(f"{i},{i}" for i in range(5)))
    header, rows, total_rows = load_csv_rows(p, max_rows=10)
    assert len(rows) == 5 and total_rows == 5


def test_discover_run_outputs_stem_no_prefix_collision(tmp_path):
    """stem 'video1' must NOT swallow 'video10_*.csv' in a shared flat dir."""
    from mindsight.GUI.run_outputs import discover_run_outputs

    csv_dir = tmp_path / "Outputs" / "CSV Files"
    csv_dir.mkdir(parents=True)
    for stem in ("video1", "video10"):
        (csv_dir / f"{stem}_summary.csv").write_text(_SUMMARY)
        (csv_dir / f"{stem}_Events.csv").write_text(_EVENTS)
        (csv_dir / f"{stem}_scanpath.csv").write_text("a,b\n1,2\n")

    class _Spec:
        def __init__(self, stem):
            self.run_id = f"{stem}.mp4"
            self.source = f"{stem}.mp4"
            self.output_paths = {
                "summary": str(csv_dir / f"{stem}_summary.csv"),
                "log": str(csv_dir / f"{stem}_Events.csv"),
            }

    outs = {o.run_id: o for o in discover_run_outputs(
        [_Spec("video1"), _Spec("video10")])}
    names = [p.name for p in outs["video1.mp4"].csv_paths]
    assert names == ["video1_Events.csv", "video1_scanpath.csv",
                     "video1_summary.csv"]
    assert not any("video10" in n for n in names)


def test_discover_global_outputs(tmp_path):
    from mindsight.GUI.run_outputs import discover_global_outputs

    out_root = tmp_path / "Outputs"
    csv_dir = out_root / "CSV Files"
    cond_dir = out_root / "By Condition"
    csv_dir.mkdir(parents=True)
    cond_dir.mkdir(parents=True)
    (csv_dir / "Global_summary.csv").write_text(_SUMMARY)
    (csv_dir / "Global_Events.csv").write_text(_EVENTS)
    (csv_dir / "Global_scanpath.csv").write_text("a,b\n1,2\n")
    # a per-run file in the same dir must NOT be picked up as a global.
    (csv_dir / "a_summary.csv").write_text(_SUMMARY)
    (cond_dir / "GroupA_summary.csv").write_text(_SUMMARY)

    out = discover_global_outputs(out_root)
    assert out is not None
    assert out.run_id == "Global (project)" and out.stem == "Global"
    assert out.summary.name == "Global_summary.csv"
    assert out.events.name == "Global_Events.csv"
    names = [p.name for p in out.csv_paths]
    assert names == ["GroupA_summary.csv", "Global_Events.csv",
                     "Global_scanpath.csv", "Global_summary.csv"]
    assert "a_summary.csv" not in names


def test_discover_global_outputs_absent(tmp_path):
    from mindsight.GUI.run_outputs import discover_global_outputs
    assert discover_global_outputs(tmp_path / "Outputs") is None


# ── Qt panel ─────────────────────────────────────────────────────────────────

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_output_tabs_render_from_written_csvs(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import RunStudyTab
    proj = _make_project(tmp_path)
    before = sorted(p.name for p in (proj / "Outputs" / "CSV Files").iterdir())

    tab = RunStudyTab()
    tab._open_project(str(proj))

    # three tabs: Log | Charts | Output CSVs
    labels = [tab._output_tabs.tabText(i)
              for i in range(tab._output_tabs.count())]
    assert labels == ["Log", "Charts", "Output CSVs"]

    # charts: the run is selectable and a canvas rendered
    assert tab._charts_run.currentText() == "a.mp4"
    assert tab._chart_canvas is not None
    assert tab._chart_canvas.isVisibleTo(tab._output_tabs.widget(1))
    assert not tab._chart_placeholder.isVisibleTo(tab._output_tabs.widget(1))

    # CSV viewer: file combo lists the run's CSVs; table populated read-only
    files = [tab._csv_file.itemText(i) for i in range(tab._csv_file.count())]
    assert files == ["a_Events.csv", "a_scanpath.csv", "a_summary.csv"]
    assert tab._csv_table.columnCount() == 14      # Events header width
    assert tab._csv_table.rowCount() == 3
    assert tab._csv_table.item(0, 5).text() == "dining table"

    # HARD CONSTRAINT: rendering wrote NOTHING into the Outputs tree
    after = sorted(p.name for p in (proj / "Outputs" / "CSV Files").iterdir())
    assert after == before


def test_output_tabs_include_global_entry(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import RunStudyTab
    proj = _make_project(tmp_path)
    csv_dir = proj / "Outputs" / "CSV Files"
    (csv_dir / "Global_summary.csv").write_text(_SUMMARY)
    (csv_dir / "Global_Events.csv").write_text(_EVENTS)
    cond = proj / "Outputs" / "By Condition"
    cond.mkdir(parents=True)
    (cond / "GroupA_summary.csv").write_text(_SUMMARY)
    before = sorted(str(p.relative_to(proj))
                    for p in (proj / "Outputs").rglob("*") if p.is_file())

    tab = RunStudyTab()
    tab._open_project(str(proj))

    # Both selectors list the project-level aggregate entry (after the runs).
    runs = [tab._csv_run.itemText(i) for i in range(tab._csv_run.count())]
    assert "a.mp4" in runs and "Global (project)" in runs

    idx = tab._csv_run.findText("Global (project)")
    tab._csv_run.setCurrentIndex(idx)
    files = [tab._csv_file.itemText(i) for i in range(tab._csv_file.count())]
    assert "Global_summary.csv" in files
    assert "Global_Events.csv" in files
    assert "GroupA_summary.csv" in files

    # Charts render from the Global summary/Events (prefixed columns tolerated).
    cidx = tab._charts_run.findText("Global (project)")
    tab._charts_run.setCurrentIndex(cidx)
    assert tab._chart_canvas is not None
    assert tab._chart_canvas.isVisibleTo(tab._output_tabs.widget(1))

    # HARD CONSTRAINT: surfacing the aggregates wrote NOTHING new to Outputs.
    after = sorted(str(p.relative_to(proj))
                   for p in (proj / "Outputs").rglob("*") if p.is_file())
    assert after == before


def test_output_tabs_placeholder_without_csvs(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import RunStudyTab
    proj = _make_project(tmp_path, with_outputs=False)
    tab = RunStudyTab()
    tab._open_project(str(proj))
    assert tab._charts_run.count() == 0
    assert tab._chart_placeholder.isVisibleTo(tab._output_tabs.widget(1))
    assert tab._csv_file.count() == 0
    assert tab._csv_table.rowCount() == 0
