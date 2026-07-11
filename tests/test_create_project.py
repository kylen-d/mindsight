"""Coverage for runner.create_project -- the blank-project scaffolder.

The GUI New Project button reuses this; these pin the on-disk shape and the
guard rails (bad names, missing parent, non-empty target) without any Qt.
"""

import pytest

from mindsight.project.runner import create_project, load_project_config


def test_create_project_standard_layout(tmp_path):
    proj = create_project(tmp_path, "MyStudy")
    assert proj == tmp_path / "MyStudy"
    assert (proj / "project.yaml").is_file()
    assert (proj / "Inputs" / "Videos").is_dir()
    assert (proj / "Inputs" / "Prompts").is_dir()
    assert (proj / "Pipeline").is_dir()
    cfg = load_project_config(proj)
    assert cfg is not None
    assert cfg.pipeline_path == "Pipeline/pipeline.yaml"


def test_create_project_opens_cleanly(tmp_path):
    from mindsight.project.project import Project
    proj = create_project(tmp_path, "Fresh")
    # A freshly scaffolded project validates + preflights (advisories, no crash).
    project = Project.open(proj)
    report = project.preflight()
    assert report.checks
    # No hard failure on the structure/pipeline checks of an empty project.
    by_id = {c.id: c for c in report.checks}
    assert by_id["project_structure"].severity == "ok"
    assert by_id["pipeline_config"].severity in ("ok", "warn")


@pytest.mark.parametrize("bad", ["", "   ", ".", "..", "a/b", "a\\b"])
def test_create_project_rejects_bad_names(tmp_path, bad):
    with pytest.raises(ValueError):
        create_project(tmp_path, bad)


def test_create_project_missing_parent(tmp_path):
    with pytest.raises(ValueError):
        create_project(tmp_path / "nope", "X")


def test_create_project_rejects_nonempty_target(tmp_path):
    (tmp_path / "Dup").mkdir()
    (tmp_path / "Dup" / "keep.txt").write_text("x")
    with pytest.raises(ValueError):
        create_project(tmp_path, "Dup")


def test_create_project_allows_empty_existing_dir(tmp_path):
    (tmp_path / "Empty").mkdir()
    proj = create_project(tmp_path, "Empty")   # empty existing dir is fine
    assert (proj / "project.yaml").is_file()


def test_create_project_run_folder_layout_no_vestigial_videos(tmp_path):
    # Eyes-on C: wizard projects stage Inputs/Runs/ -- an empty Inputs/Videos/
    # beside it is a vestige (and a step toward the dual-layout error).
    proj = create_project(tmp_path, "Staged", layout="run_folder")
    assert (proj / "Inputs" / "Runs").is_dir()
    assert not (proj / "Inputs" / "Videos").exists()
    assert (proj / "Inputs" / "Prompts").is_dir()
