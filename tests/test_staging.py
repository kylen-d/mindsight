"""SP3.1 Batch E Step 9: RunSpec staging + discovery (fast, no models).

Covers layout detection (Q1), run.yaml parsing + validation (Q2), the legacy
and run-folder producers, metadata precedence, and ledger-key compatibility so
a pre-SP3 ledger keeps resuming.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mindsight.pipeline_config import ProjectConfig, ProjectOutputConfig
from mindsight.project.ledger import Ledger, compute_video_hash
from mindsight.project.staging import (
    AMBIGUOUS,
    LEGACY,
    RUN_FOLDER,
    RunSpec,
    detect_layout,
    discover_run_specs,
    inspect_run_folders,
    parse_run_yaml,
    run_display_name,
    run_folder_output_paths,
)


# ── fixtures ────────────────────────────────────────────────────────────────

def _touch(path: Path, data: bytes = b"\x00\x00") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _flat_project(tmp_path, videos=("a.mp4", "b.mp4")):
    proj = tmp_path / "proj"
    for name in videos:
        _touch(proj / "Inputs" / "Videos" / name)
    return proj


def _run_folder(proj, run_id, *, video="video.mp4", run_yaml=None,
                data=b"\x00\x00"):
    folder = proj / "Inputs" / "Runs" / run_id
    _touch(folder / video, data)
    if run_yaml is not None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run.yaml").write_text(run_yaml)
    return folder


# ── layout detection (Q1) ───────────────────────────────────────────────────

def test_layout_legacy(tmp_path):
    assert detect_layout(_flat_project(tmp_path)) == LEGACY


def test_layout_run_folder(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "run01")
    assert detect_layout(proj) == RUN_FOLDER


def test_layout_ambiguous(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    _run_folder(proj, "run01")
    assert detect_layout(proj) == AMBIGUOUS


def test_layout_empty_is_legacy(tmp_path):
    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    assert detect_layout(proj) == LEGACY


def test_layout_empty_runs_dir_is_legacy(tmp_path):
    # An Inputs/Runs/ with no subfolders does not trigger run-folder mode.
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    (proj / "Inputs" / "Runs").mkdir(parents=True)
    assert detect_layout(proj) == LEGACY


# ── run.yaml parsing (Q2) ───────────────────────────────────────────────────

def test_parse_run_yaml_missing(tmp_path):
    m = parse_run_yaml(tmp_path / "nope.yaml")
    assert m.error is None and m.pid_map is None and m.conditions == []
    assert m.manifest_meta == {} and m.unknown_keys == []


def test_parse_run_yaml_full(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(
        "participants: {0: S70, 1: S71}\n"
        "conditions: [collab, kitchenA]\n"
        "date: 2026-07-02\n"
        "session: dyad-07\n"
        "notes: camera bumped\n"
        "extra: {experimenter: KD}\n")
    m = parse_run_yaml(p)
    assert m.error is None
    assert m.pid_map == {0: "S70", 1: "S71"}
    assert m.conditions == ["collab", "kitchenA"]
    assert m.manifest_meta["date"] == "2026-07-02"
    assert m.manifest_meta["session"] == "dyad-07"
    assert m.manifest_meta["extra"] == {"experimenter": "KD"}
    assert m.unknown_keys == []


def test_parse_run_yaml_conditions_scalar(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("conditions: solo\n")
    assert parse_run_yaml(p).conditions == ["solo"]


def test_parse_run_yaml_unknown_keys_warn(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("participants: {0: S70}\ncondtions: [typo]\nfoo: bar\n")
    m = parse_run_yaml(p)
    assert m.error is None
    assert m.unknown_keys == ["condtions", "foo"]   # sorted


def test_parse_run_yaml_bad_participants_type(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("participants: [S70, S71]\n")
    m = parse_run_yaml(p)
    assert m.error and "participants" in m.error


def test_parse_run_yaml_bad_track_id(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("participants: {alice: S70}\n")
    m = parse_run_yaml(p)
    assert m.error and "integer track" in m.error


def test_parse_run_yaml_bad_conditions_type(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("conditions: {a: 1}\n")
    m = parse_run_yaml(p)
    assert m.error and "conditions" in m.error


def test_parse_run_yaml_non_mapping(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("- just\n- a\n- list\n")
    m = parse_run_yaml(p)
    assert m.error and "mapping" in m.error


# ── legacy producer -- byte-compatible ──────────────────────────────────────

def test_legacy_run_specs_run_ids_are_filenames(tmp_path):
    proj = _flat_project(tmp_path, videos=("b.mp4", "a.mp4"))
    specs = discover_run_specs(proj, None, layout=LEGACY)
    assert [s.run_id for s in specs] == ["a.mp4", "b.mp4"]      # sorted
    assert all(isinstance(s, RunSpec) for s in specs)
    # Output paths stay flat (T1): CSV Files / Videos.
    log = specs[0].output_paths["log"]
    assert log.endswith("Outputs/CSV Files/a_Events.csv")
    assert run_display_name(specs[0]) == "a"                    # source stem


def test_legacy_conditions_and_pid_from_project_yaml(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    cfg = ProjectConfig(conditions={"a.mp4": ["GroupA"]},
                        participants={"a.mp4": {0: "S70"}})
    specs = discover_run_specs(proj, cfg, layout=LEGACY,
                               pid_maps=cfg.participants)
    assert specs[0].conditions == "GroupA"
    assert specs[0].pid_map == {0: "S70"}


# ── run-folder producer (Q1/Q2) ─────────────────────────────────────────────

def test_run_folder_specs_basic(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "dyad07_collab", video="clip.mp4",
                run_yaml="participants: {0: S70, 1: S71}\n"
                         "conditions: [collab]\n"
                         "date: 2026-07-02\nnotes: ok\n")
    specs = discover_run_specs(proj, None, layout=RUN_FOLDER)
    assert len(specs) == 1
    s = specs[0]
    assert s.run_id == "dyad07_collab"
    assert s.source.name == "clip.mp4"
    assert s.pid_map == {0: "S70", 1: "S71"}
    assert s.conditions == "collab"
    assert s.meta == {"date": "2026-07-02", "notes": "ok"}
    # Mirrored output placement (Q3).
    assert s.output_paths["summary"].endswith(
        "Outputs/Runs/dyad07_collab/dyad07_collab_summary.csv")
    assert run_display_name(s) == "dyad07_collab"


def test_run_folder_bare_folder_just_works(tmp_path):
    # A folder with one video and no run.yaml (Q1: should just work).
    proj = tmp_path / "proj"
    _run_folder(proj, "run01")
    specs = discover_run_specs(proj, None, layout=RUN_FOLDER)
    assert specs[0].run_id == "run01"
    assert specs[0].pid_map is None and specs[0].conditions == ""
    assert specs[0].meta == {}


def test_run_folder_unicode_and_space_run_id(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "dyad 07 café")
    specs = discover_run_specs(proj, None, layout=RUN_FOLDER)
    assert specs[0].run_id == "dyad 07 café"
    assert "dyad 07 café" in specs[0].output_paths["log"]


def test_run_folder_two_videos_raises(tmp_path):
    proj = tmp_path / "proj"
    folder = _run_folder(proj, "run01", video="a.mp4")
    _touch(folder / "b.mp4")
    with pytest.raises(ValueError, match="exactly one"):
        discover_run_specs(proj, None, layout=RUN_FOLDER)


def test_run_folder_zero_videos_raises(tmp_path):
    proj = tmp_path / "proj"
    folder = proj / "Inputs" / "Runs" / "run01"
    folder.mkdir(parents=True)
    (folder / "run.yaml").write_text("conditions: [x]\n")
    with pytest.raises(ValueError, match="no video"):
        discover_run_specs(proj, None, layout=RUN_FOLDER)


def test_run_folder_bad_metadata_raises(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "run01", run_yaml="participants: [S70]\n")
    with pytest.raises(ValueError, match="participants"):
        discover_run_specs(proj, None, layout=RUN_FOLDER)


def test_ambiguous_layout_raises(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    _run_folder(proj, "run01")
    with pytest.raises(ValueError, match="ambiguous"):
        discover_run_specs(proj, None)


# ── metadata precedence (Q2): run.yaml > project.yaml > CSV ──────────────────

def test_precedence_run_yaml_wins(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "run01",
                run_yaml="participants: {0: FROM_RUN}\nconditions: [run_cond]\n")
    cfg = ProjectConfig(conditions={"run01": ["proj_cond"]},
                        participants={"run01": {0: "FROM_PROJ"}})
    specs = discover_run_specs(proj, cfg, layout=RUN_FOLDER,
                               pid_maps={"run01": {0: "FROM_CSV"}})
    assert specs[0].pid_map == {0: "FROM_RUN"}
    assert specs[0].conditions == "run_cond"


def test_precedence_project_yaml_then_csv(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "run01")            # no run.yaml
    cfg = ProjectConfig(conditions={"run01": ["proj_cond"]},
                        participants={"run01": {0: "FROM_PROJ"}})
    specs = discover_run_specs(proj, cfg, layout=RUN_FOLDER,
                               pid_maps={"run01": {0: "FROM_CSV"}})
    assert specs[0].pid_map == {0: "FROM_PROJ"}   # project.yaml over CSV
    assert specs[0].conditions == "proj_cond"


def test_precedence_csv_fallback(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "run01")
    specs = discover_run_specs(proj, None, layout=RUN_FOLDER,
                               pid_maps={"run01": {0: "FROM_CSV"}})
    assert specs[0].pid_map == {0: "FROM_CSV"}


# ── output path resolution honours project.yaml output.directory ────────────

def test_run_folder_output_root_override(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "run01")
    cfg = ProjectConfig(output=ProjectOutputConfig(directory=str(tmp_path / "OUT")))
    specs = discover_run_specs(proj, cfg, layout=RUN_FOLDER)
    assert str(tmp_path / "OUT" / "Runs" / "run01") in specs[0].output_paths["log"]


def test_run_folder_output_paths_helper():
    paths = run_folder_output_paths(Path("/o"), "R1")
    assert paths["save"] == "/o/Runs/R1/R1_Video_Output.mp4"
    assert paths["heatmap"] == "/o/Runs/R1/R1_Heatmap"


# ── ledger-key compatibility: a pre-SP3 ledger still resumes ────────────────

def test_legacy_run_id_matches_old_ledger_key(tmp_path):
    """A pre-SP3 ledger keyed by the video filename decides 'skip' for the
    matching legacy RunSpec (run_id == source.name)."""
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    specs = discover_run_specs(proj, None, layout=LEGACY)
    spec = specs[0]
    assert spec.run_id == "a.mp4"

    out_root = proj / "Outputs"
    ledger = Ledger.load(out_root)
    vhash = compute_video_hash(spec.source, pid_map=spec.pid_map,
                               conditions=spec.conditions,
                               aux_streams=spec.aux_streams)
    hashes = ("CFG", vhash)
    ledger.mark_started(spec.run_id, hashes, spec.output_paths)
    ledger.mark_done(spec.run_id, "manifest.json")
    # Fresh load off disk (simulating a later batch) -> skip.
    assert Ledger.load(out_root).decide(spec.run_id, hashes) == "skip"


def test_inspect_run_folders_non_raising(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "bad", run_yaml="participants: [oops]\n")
    infos = inspect_run_folders(proj)
    assert len(infos) == 1 and infos[0].meta.error   # inspection never raises
