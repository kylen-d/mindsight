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


# ══════════════════════════════════════════════════════════════════════════════
# SP3.1 Batch E Step 11: manual staging (Q7) -- stage_run / single_run_spec
# ══════════════════════════════════════════════════════════════════════════════

from mindsight.project.staging import (  # noqa: E402
    parse_run_mapping,
    single_run_spec,
    stage_run,
)


def _video(tmp_path, name="clip.mp4", data=b"\x00" * 64):
    return _touch(tmp_path / "incoming" / name, data)


def test_stage_run_copy_default(tmp_path):
    proj = tmp_path / "proj"
    vid = _video(tmp_path)
    meta = {"participants": {0: "S70"}, "conditions": ["collab"],
            "date": "2026-07-02", "notes": "hi"}
    spec = stage_run(proj, vid, meta)
    # Folder created from the video stem; video COPIED (original kept, Q7).
    dest = proj / "Inputs" / "Runs" / "clip" / "clip.mp4"
    assert dest.is_file() and vid.is_file()
    assert dest.read_bytes() == vid.read_bytes()
    # run.yaml written and re-parseable to the same values.
    reparsed = discover_run_specs(proj, None, layout=RUN_FOLDER)[0]
    assert reparsed.pid_map == {0: "S70"} and reparsed.conditions == "collab"
    assert reparsed.meta == {"date": "2026-07-02", "notes": "hi"}
    # Returned spec matches what discovery produces.
    assert spec.run_id == "clip" and spec.source == dest
    assert spec.pid_map == {0: "S70"} and spec.conditions == "collab"
    assert spec.output_paths == reparsed.output_paths
    assert spec.output_paths["summary"].endswith(
        "Outputs/Runs/clip/clip_summary.csv")


def test_stage_run_move_removes_original(tmp_path):
    proj = tmp_path / "proj"
    vid = _video(tmp_path)
    spec = stage_run(proj, vid, mode="move")
    assert not vid.exists()                          # moved, not copied
    assert Path(spec.source).is_file()


def test_stage_run_move_cross_device_fallback(tmp_path, monkeypatch):
    """When os.rename fails (EXDEV), shutil.move's copy fallback still stages."""
    import errno
    import os
    proj = tmp_path / "proj"
    vid = _video(tmp_path)

    real_rename = os.rename

    def _exdev(src, dst, *a, **kw):
        # Only sabotage the video move itself; everything else renames fine.
        if str(src) == str(vid):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr(os, "rename", _exdev)
    spec = stage_run(proj, vid, mode="move")
    assert Path(spec.source).is_file() and not vid.exists()


def test_stage_run_no_meta_writes_no_run_yaml(tmp_path):
    proj = tmp_path / "proj"
    spec = stage_run(proj, _video(tmp_path))
    run_dir = proj / "Inputs" / "Runs" / "clip"
    assert not (run_dir / "run.yaml").exists()       # bare folder just works
    assert spec.pid_map is None and spec.conditions == "" and spec.meta == {}


def test_stage_run_restage_same_video_is_collision_safe(tmp_path):
    proj = tmp_path / "proj"
    vid = _video(tmp_path)
    s1 = stage_run(proj, vid)
    s2 = stage_run(proj, vid)                        # re-stage the SAME video
    assert s1.run_id == "clip" and s2.run_id == "clip_2"
    assert (proj / "Inputs" / "Runs" / "clip_2" / "clip.mp4").is_file()
    s3 = stage_run(proj, vid)
    assert s3.run_id == "clip_3"


def test_stage_run_explicit_run_id_and_sanitization(tmp_path):
    proj = tmp_path / "proj"
    spec = stage_run(proj, _video(tmp_path), run_id='dyad/07:pilot?')
    assert spec.run_id == "dyad_07_pilot_"           # unsafe chars -> _
    assert (proj / "Inputs" / "Runs" / spec.run_id).is_dir()


def test_stage_run_into_flat_project_raises(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    with pytest.raises(ValueError, match="ambiguous"):
        stage_run(proj, _video(tmp_path))


def test_stage_run_bad_meta_raises(tmp_path):
    proj = tmp_path / "proj"
    with pytest.raises(ValueError, match="participants"):
        stage_run(proj, _video(tmp_path), {"participants": ["S70"]})
    with pytest.raises(ValueError, match="unknown run metadata"):
        stage_run(proj, _video(tmp_path), {"condtions": ["typo"]})


def test_stage_run_missing_video_raises(tmp_path):
    with pytest.raises(ValueError, match="video not found"):
        stage_run(tmp_path / "proj", tmp_path / "nope.mp4")


def test_stage_run_bad_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        stage_run(tmp_path / "proj", _video(tmp_path), mode="link")


def test_staged_project_runs_through_discovery(tmp_path):
    # End-to-end staging -> discovery: two manual stages become two RunSpecs.
    proj = tmp_path / "proj"
    stage_run(proj, _video(tmp_path, "one.mp4"), {"conditions": ["c1"]})
    stage_run(proj, _video(tmp_path, "two.mp4"), {"conditions": ["c2"]})
    specs = discover_run_specs(proj, None)
    assert [s.run_id for s in specs] == ["one", "two"]
    assert [s.conditions for s in specs] == ["c1", "c2"]
    assert detect_layout(proj) == RUN_FOLDER


# ── single_run_spec (run-now, Q7) ────────────────────────────────────────────

def test_single_run_spec_basic(tmp_path):
    vid = _video(tmp_path)
    spec = single_run_spec(vid, {"participants": {0: "S70"},
                                 "conditions": ["pilot"],
                                 "session": "s1"},
                           output_dir=tmp_path / "OUT")
    assert spec.run_id == "clip" and spec.source == vid
    assert spec.pid_map == {0: "S70"} and spec.conditions == "pilot"
    assert spec.meta == {"session": "s1"}
    # Outputs land FLAT in the chosen dir (no Runs/ nesting, no ledger).
    assert spec.output_paths["summary"] == str(
        tmp_path / "OUT" / "clip_summary.csv")
    assert spec.output_paths["save"] == str(
        tmp_path / "OUT" / "clip_Video_Output.mp4")


def test_single_run_spec_defaults_to_project_outputs(tmp_path):
    # B1 F1: an omitted output_dir defaults to the open project's Outputs root
    # (NOT a CWD-relative Outputs/), so files never vanish for the user.
    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    spec = single_run_spec(_video(tmp_path), project=proj)
    assert spec.output_paths["summary"] == str(
        proj / "Outputs" / "clip_summary.csv")
    assert spec.output_paths["save"] == str(
        proj / "Outputs" / "clip_Video_Output.mp4")


def test_single_run_spec_no_context_raises(tmp_path):
    # B1 F1: neither an output_dir nor a project -> no sane default -> plain
    # ValueError (surfaced by the GUI) rather than a silent CWD-relative dir.
    with pytest.raises(ValueError, match="choose an output directory"):
        single_run_spec(_video(tmp_path))


def test_single_run_spec_explicit_dir_wins_over_project(tmp_path):
    # An explicit output_dir always wins, even with a project available.
    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    spec = single_run_spec(_video(tmp_path), output_dir=tmp_path / "OUT",
                           project=proj)
    assert spec.output_paths["summary"] == str(
        tmp_path / "OUT" / "clip_summary.csv")


def test_single_run_spec_bad_meta_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown run metadata"):
        single_run_spec(_video(tmp_path), {"nope": 1})
    with pytest.raises(ValueError, match="video not found"):
        single_run_spec(tmp_path / "missing.mp4")


def test_parse_run_mapping_shared_with_yaml_path():
    m = parse_run_mapping({"participants": {0: "S70"}, "conditions": "solo"})
    assert m.pid_map == {0: "S70"} and m.conditions == ["solo"]
    assert m.error is None


# ══════════════════════════════════════════════════════════════════════════════
# update_run_metadata (G-DEFER-1): the single pre-run metadata WRITE path
# ══════════════════════════════════════════════════════════════════════════════

import yaml  # noqa: E402

from mindsight.project.runner import load_project_config  # noqa: E402
from mindsight.project.staging import update_run_metadata  # noqa: E402


def test_update_metadata_legacy_sets_project_yaml(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4", "b.mp4"))
    update_run_metadata(proj, "a.mp4", participants={0: "S70", 1: "S71"},
                        conditions=["collab", "kitchenA"])
    cfg = load_project_config(proj)
    assert cfg.participants["a.mp4"] == {0: "S70", 1: "S71"}
    assert cfg.conditions["a.mp4"] == ["collab", "kitchenA"]
    # discovery reflects the edit (pid maps resolved from project.yaml, as the
    # facade does)
    spec = next(s for s in discover_run_specs(proj, cfg, pid_maps=cfg.participants)
                if s.run_id == "a.mp4")
    assert spec.pid_map == {0: "S70", 1: "S71"}
    assert spec.conditions == "collab|kitchenA"


def test_update_metadata_legacy_string_condition_and_clear(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    update_run_metadata(proj, "a.mp4", conditions="solo")
    assert load_project_config(proj).conditions["a.mp4"] == ["solo"]
    # None clears the entry
    update_run_metadata(proj, "a.mp4", conditions=None)
    assert "a.mp4" not in (load_project_config(proj).conditions or {})


def test_update_metadata_legacy_preserves_other_fields(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4", "b.mp4"))
    update_run_metadata(proj, "a.mp4", conditions="x")
    update_run_metadata(proj, "b.mp4", participants={0: "P9"})
    cfg = load_project_config(proj)
    assert cfg.conditions["a.mp4"] == ["x"]
    assert cfg.participants["b.mp4"] == {0: "P9"}


def test_update_metadata_only_touches_named_field(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    update_run_metadata(proj, "a.mp4", participants={0: "P0"}, conditions="c1")
    # A later edit that names ONLY conditions leaves participants intact.
    update_run_metadata(proj, "a.mp4", conditions="c2")
    cfg = load_project_config(proj)
    assert cfg.participants["a.mp4"] == {0: "P0"}
    assert cfg.conditions["a.mp4"] == ["c2"]


def test_update_metadata_run_folder_writes_run_yaml(tmp_path):
    proj = tmp_path / "rf"
    _run_folder(proj, "dyad07", run_yaml="date: 2026-07-02\nnotes: hi\n")
    update_run_metadata(proj, "dyad07", participants={0: "S70"},
                        conditions=["collab"])
    raw = yaml.safe_load((proj / "Inputs" / "Runs" / "dyad07" / "run.yaml")
                         .read_text())
    assert raw["participants"] == {0: "S70"}
    assert raw["conditions"] == ["collab"]
    # manifest-only keys preserved
    assert raw["notes"] == "hi"
    assert str(raw["date"]).startswith("2026-07-02")


def test_update_metadata_run_folder_creates_run_yaml(tmp_path):
    proj = tmp_path / "rf"
    _run_folder(proj, "solo")            # bare folder, no run.yaml
    update_run_metadata(proj, "solo", conditions="kitchen")
    raw = yaml.safe_load((proj / "Inputs" / "Runs" / "solo" / "run.yaml")
                         .read_text())
    assert raw["conditions"] == ["kitchen"]


def test_update_metadata_run_folder_clear_removes_empty_yaml(tmp_path):
    proj = tmp_path / "rf"
    _run_folder(proj, "solo", run_yaml="conditions: [x]\n")
    update_run_metadata(proj, "solo", conditions=None)
    assert not (proj / "Inputs" / "Runs" / "solo" / "run.yaml").exists()


def test_update_metadata_bad_participants_raises(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    with pytest.raises(ValueError, match="track ids"):
        update_run_metadata(proj, "a.mp4", participants={"notanint": "S"})
    # nothing was written on the failed validate
    assert load_project_config(proj) is None


def test_update_metadata_missing_run_folder_raises(tmp_path):
    proj = tmp_path / "rf"
    _run_folder(proj, "exists")
    with pytest.raises(ValueError, match="run folder not found"):
        update_run_metadata(proj, "ghost", conditions="x")


def test_update_metadata_noop_when_nothing_named(tmp_path):
    proj = _flat_project(tmp_path, videos=("a.mp4",))
    update_run_metadata(proj, "a.mp4")   # neither field named
    assert load_project_config(proj) is None
