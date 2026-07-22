"""v1.3.1 item 3a: condition-tagged VP files (format v2) + effective-subset
resolution.  Pure payload logic -- no models, no Qt except the widgets
serializer round trip."""

import json

import pytest

from mindsight.ObjectDetection.object_detection import (
    build_vp_payload,
    ensure_vp_version,
    filter_vp_for_conditions,
    load_vp_data,
    vp_content_digest,
    vp_declared_conditions,
    vp_has_conditions,
)

CLASSES_PLAIN = [{"id": 0, "name": "bowl"}, {"id": 1, "name": "spoon"}]
REFS = [
    {"image": "/a/ref1.jpg", "annotations": [
        {"cls_id": 0, "bbox": [1, 2, 10, 12]},
        {"cls_id": 1, "bbox": [3, 4, 20, 22]}]},
    {"image": "/a/ref2.jpg", "annotations": [
        {"cls_id": 0, "bbox": [5, 6, 30, 32]}]},
]


# ── payload building ─────────────────────────────────────────────────────────

def test_v1_payload_byte_stable_without_conditions():
    payload = build_vp_payload(CLASSES_PLAIN, REFS)
    legacy = {"version": 1, "classes": CLASSES_PLAIN, "references": REFS}
    assert json.dumps(payload, indent=2) == json.dumps(legacy, indent=2)
    # An empty conditions key on a class is stripped, keeping v1.
    with_empty = [{"id": 0, "name": "bowl", "conditions": []},
                  {"id": 1, "name": "spoon"}]
    assert (json.dumps(build_vp_payload(with_empty, REFS), indent=2)
            == json.dumps(legacy, indent=2))


def test_v2_payload_when_tagged_and_vocab_derivation():
    tagged = [{"id": 0, "name": "bowl", "conditions": ["warm"]},
              {"id": 1, "name": "spoon"}]
    p = build_vp_payload(tagged, REFS, conditions=["cold"])
    assert p["version"] == 2
    assert p["conditions"] == ["cold", "warm"]   # declared first, then tags
    assert p["classes"][0]["conditions"] == ["warm"]
    assert "conditions" not in p["classes"][1]
    # Vocabulary alone (no tagged class) still upgrades to v2.
    p2 = build_vp_payload(CLASSES_PLAIN, REFS, conditions=["warm"])
    assert p2["version"] == 2 and p2["conditions"] == ["warm"]


# ── version guard ────────────────────────────────────────────────────────────

def test_version_guard(tmp_path):
    ok = tmp_path / "ok.vp.json"
    ok.write_text(json.dumps({"classes": [], "references": []}))
    assert load_vp_data(ok)["classes"] == []     # missing version = 1

    future = tmp_path / "future.vp.json"
    future.write_text(json.dumps({"version": 3, "classes": []}))
    with pytest.raises(ValueError, match="version 3"):
        load_vp_data(future)

    with pytest.raises(ValueError, match="unreadable"):
        ensure_vp_version({"version": "banana"})


# ── effective-subset resolution ──────────────────────────────────────────────

def _conditioned_vp():
    return {
        "version": 2,
        "conditions": ["warm", "cold"],
        "classes": [
            {"id": 0, "name": "table"},                          # always
            {"id": 1, "name": "soup", "conditions": ["warm"]},
            {"id": 2, "name": "icecream", "conditions": ["cold"]},
        ],
        "references": [
            {"image": "/r/both.jpg", "annotations": [
                {"cls_id": 0, "bbox": [0, 0, 5, 5]},
                {"cls_id": 1, "bbox": [1, 1, 6, 6]},
                {"cls_id": 2, "bbox": [2, 2, 7, 7]}]},
            {"image": "/r/cold_only.jpg", "annotations": [
                {"cls_id": 2, "bbox": [3, 3, 8, 8]}]},
        ],
    }


def test_filter_matches_tags_and_renumbers():
    out = filter_vp_for_conditions(_conditioned_vp(), ["warm"])
    assert [c["name"] for c in out["classes"]] == ["table", "soup"]
    assert [c["id"] for c in out["classes"]] == [0, 1]
    assert all("conditions" not in c for c in out["classes"])
    # The cold-only reference dropped; annotations remapped.
    assert len(out["references"]) == 1
    assert [a["cls_id"] for a in out["references"][0]["annotations"]] == [0, 1]


def test_filter_multi_tag_union_and_untagged_video():
    vp = _conditioned_vp()
    both = filter_vp_for_conditions(vp, ["warm", "cold"])
    assert [c["name"] for c in both["classes"]] == ["table", "soup", "icecream"]
    none = filter_vp_for_conditions(vp, [])
    assert [c["name"] for c in none["classes"]] == ["table"]   # always-on only
    unknown = filter_vp_for_conditions(vp, ["neutral"])
    assert [c["name"] for c in unknown["classes"]] == ["table"]


def test_filter_fast_path_returns_same_object():
    plain = {"version": 1, "classes": CLASSES_PLAIN, "references": REFS}
    assert filter_vp_for_conditions(plain, ["warm"]) is plain
    assert not vp_has_conditions(plain)
    assert vp_has_conditions(_conditioned_vp())


def test_declared_conditions_union():
    vp = _conditioned_vp()
    vp["classes"].append({"id": 3, "name": "x", "conditions": ["extra"]})
    assert vp_declared_conditions(vp) == ["warm", "cold", "extra"]


def test_content_digest(tmp_path):
    import hashlib
    f = tmp_path / "a.vp.json"
    f.write_text("{}")
    assert vp_content_digest(f) == hashlib.sha256(b"{}").hexdigest()


# ── serializer + archive round trips (Qt import for widgets) ─────────────────

def test_save_vp_file_round_trips_conditions(tmp_path):
    pytest.importorskip("PyQt6")
    from mindsight.GUI.widgets import load_vp_file, save_vp_file
    tagged = [{"id": 0, "name": "bowl", "conditions": ["warm"]},
              {"id": 1, "name": "spoon"}]
    p = tmp_path / "cond.vp.json"
    save_vp_file(str(p), tagged, REFS, conditions=["warm", "cold"])
    data = load_vp_file(str(p))
    assert data["version"] == 2
    assert data["conditions"] == ["warm", "cold"]
    assert data["classes"][0]["conditions"] == ["warm"]
    # Plain save stays byte-identical to the pre-1.3.1 writer.
    plain = tmp_path / "plain.vp.json"
    save_vp_file(str(plain), CLASSES_PLAIN, REFS)
    legacy = json.dumps({"version": 1, "classes": CLASSES_PLAIN,
                         "references": REFS}, indent=2)
    assert plain.read_text() == legacy


def test_archive_round_trips_conditions(tmp_path):
    pytest.importorskip("PyQt6")
    import cv2
    import numpy as np

    from mindsight.GUI.vp_archive import export_vp_archive, import_vp_archive
    img = tmp_path / "ref.jpg"
    cv2.imwrite(str(img), np.full((16, 16, 3), 90, dtype=np.uint8))
    tagged = [{"id": 0, "name": "bowl", "conditions": ["warm"]}]
    refs = [{"image": str(img),
             "annotations": [{"cls_id": 0, "bbox": [1, 1, 8, 8]}]}]
    zp = export_vp_archive(tmp_path / "c.vp.zip", tagged, refs,
                           conditions=["warm", "cold"])
    vp_json = import_vp_archive(zp)
    data = json.loads(vp_json.read_text())
    assert data["version"] == 2
    assert data["conditions"] == ["warm", "cold"]
    assert data["classes"][0]["conditions"] == ["warm"]


def test_detector_rejects_future_format_before_model_load(tmp_path):
    from mindsight.ObjectDetection.object_detection import YOLOEVPDetector
    f = tmp_path / "future.vp.json"
    f.write_text(json.dumps({"version": 9, "classes": [], "references": []}))
    # Model path is bogus on purpose: the format guard must fire FIRST.
    with pytest.raises(ValueError, match="version 9"):
        YOLOEVPDetector("no-such-model.pt", str(f))


# ── 3b: flags, factory guards, identity, preflight, runner ──────────────────

def test_vp_condition_no_match_fails_before_model(tmp_path):
    from mindsight.ObjectDetection.model_factory import create_yolo_detector
    vp = tmp_path / "c.vp.json"
    vp.write_text(json.dumps({
        "version": 2,
        "classes": [{"id": 0, "name": "soup", "conditions": ["warm"]}],
        "references": [{"image": "/r/x.jpg",
                        "annotations": [{"cls_id": 0, "bbox": [1, 1, 5, 5]}]}],
    }))
    with pytest.raises(ValueError, match="matches no classes"):
        create_yolo_detector(vp_file=str(vp), vp_condition="nonexistent")


def test_factory_vp_condition_guards(tmp_path):
    from mindsight.cli_flags import parse_cli
    from mindsight.factory import build_from_namespace
    with pytest.raises(ValueError, match="needs --vp-file"):
        build_from_namespace(parse_cli(["--vp-condition", "warm"]))
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_from_namespace(parse_cli(
            ["--vp-condition", "warm", "--vp-ignore-conditions",
             "--vp-file", "x.vp.json"]))


def test_run_identity_condition_sensitivity(tmp_path):
    from mindsight.cli_flags import parse_cli
    from mindsight.config import PipelineConfig
    from mindsight.outputs import provenance

    def ident(args):
        ns = parse_cli(args)
        return provenance.run_identity(
            ns, config=PipelineConfig.from_namespace(ns), weights={})

    base = ident([])
    # Without a vp_file the new dests stay out of the wiring entirely:
    # non-VP identities (and their ledgers) are byte-unchanged.
    assert ident(["--vp-condition", "warm"]) == base
    assert ident(["--vp-ignore-conditions"]) == base

    vp = tmp_path / "x.vp.json"
    vp.write_text("{}")
    with_vp = ident(["--vp-file", str(vp)])
    assert with_vp != base
    assert ident(["--vp-file", str(vp), "--vp-condition", "warm"]) != with_vp
    assert ident(["--vp-file", str(vp), "--vp-ignore-conditions"]) != with_vp


def test_preflight_vp_condition_checks(tmp_path):
    from mindsight.cli_flags import parse_cli
    from mindsight.project.preflight import _check_vp_file
    proj = tmp_path / "p"
    (proj / "Inputs" / "Prompts").mkdir(parents=True)
    vp_path = proj / "Inputs" / "Prompts" / "s.vp.json"
    vp_path.write_text(json.dumps(_conditioned_vp()))

    runs = [{"run_id": "a.mp4", "tags": ["warm"]},
            {"run_id": "b.mp4", "tags": []}]
    res = _check_vp_file(proj, parse_cli([]), runs)
    assert res.severity == "warn"
    assert "untagged" in res.message and "unused" in res.message  # b + cold

    res_ok = _check_vp_file(
        proj, parse_cli([]),
        [{"run_id": "a.mp4", "tags": ["warm"]},
         {"run_id": "c.mp4", "tags": ["cold"]}])
    assert res_ok.severity == "ok" and "condition tags" in res_ok.message

    res_ig = _check_vp_file(proj, parse_cli(["--vp-ignore-conditions"]), runs)
    assert res_ig.severity == "ok" and "IGNORED" in res_ig.message

    # A fully-conditioned VP + an untagged video = empty subset = FAIL.
    vp_path.write_text(json.dumps({
        "version": 2,
        "classes": [{"id": 0, "name": "soup", "conditions": ["warm"]}],
        "references": [{"image": "/r/x.jpg",
                        "annotations": [{"cls_id": 0, "bbox": [1, 1, 5, 5]}]}],
    }))
    res_fail = _check_vp_file(proj, parse_cli([]),
                              [{"run_id": "b.mp4", "tags": []}])
    assert res_fail.severity == "fail" and "NO" in res_fail.message

    vp_path.write_text(json.dumps({"version": 9, "classes": []}))
    assert _check_vp_file(proj, parse_cli([]), runs).severity == "fail"


# ── 3b: project-runner per-video detector swaps (stubbed models) ─────────────

def _make_clip(path, frames=6, w=48, h=32):
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         10.0, (w, h))
    for i in range(frames):
        vw.write(np.full((h, w, 3), (i * 9) % 255, np.uint8))
    vw.release()


@pytest.fixture
def stub_pipeline(monkeypatch):
    from pathlib import Path

    import mindsight.pipeline as _pipeline_mod

    def _build(_ns):
        return tuple([None] * 14)

    class _StubPipeline:
        def __init__(self, *, output_cfg, **_kw):
            self._out = output_cfg

        def run(self, source, *, options=None, cancel=None):
            summ = Path(self._out.summary_path)
            summ.parent.mkdir(parents=True, exist_ok=True)
            summ.write_text("video_name,conditions,phenomenon,participant,"
                            "partner,object,metric,value\n")
            Path(self._out.log_path).write_text("frame,t_seconds\n")
            return iter(())

    monkeypatch.setattr(_pipeline_mod, "build_from_namespace", _build)
    monkeypatch.setattr(_pipeline_mod, "Pipeline", _StubPipeline)


@pytest.fixture
def rebuilds(monkeypatch):
    import mindsight.ObjectDetection.model_factory as mf
    calls: list = []

    def fake_create(**kw):
        calls.append([c["name"] for c in kw["vp_data"]["classes"]])
        return object(), None, set()

    monkeypatch.setattr(mf, "create_yolo_detector", fake_create)
    return calls


def _cond_project(tmp_path, vp_payload=None, project_yaml=None):
    proj = tmp_path / "proj"
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        _make_clip(proj / "Inputs" / "Videos" / name)
    (proj / "Inputs" / "Prompts").mkdir(parents=True)
    (proj / "Inputs" / "Prompts" / "study.vp.json").write_text(
        json.dumps(vp_payload or _conditioned_vp()))
    if project_yaml is None:
        project_yaml = ("version: 1\nconditions:\n  a.mp4: [warm]\n"
                        "  b.mp4: [warm]\n  c.mp4: [cold]\n")
    (proj / "project.yaml").write_text(project_yaml)
    return proj


def test_runner_swaps_detector_per_condition_group(tmp_path, stub_pipeline,
                                                   rebuilds):
    from mindsight.cli_flags import parse_cli
    from mindsight.project.runner import iter_project_runs
    proj = _cond_project(tmp_path)
    list(iter_project_runs(proj, parse_cli([])))
    # a(warm) differs from the batch-built FULL prompt -> swap; b(warm)
    # reuses it; c(cold) -> swap again.  Videos keep discovery order.
    assert rebuilds == [["table", "soup"], ["table", "icecream"]]


def test_runner_ignore_conditions_never_swaps(tmp_path, stub_pipeline,
                                              rebuilds):
    from mindsight.cli_flags import parse_cli
    from mindsight.project.runner import iter_project_runs
    proj = _cond_project(tmp_path)
    list(iter_project_runs(proj, parse_cli(["--vp-ignore-conditions"])))
    assert rebuilds == []


def test_runner_empty_subset_fails_fast(tmp_path, stub_pipeline, rebuilds):
    from mindsight.cli_flags import parse_cli
    from mindsight.project.runner import iter_project_runs
    proj = _cond_project(
        tmp_path,
        vp_payload={
            "version": 2,
            "classes": [{"id": 0, "name": "soup", "conditions": ["warm"]}],
            "references": [{"image": "/r/x.jpg", "annotations": [
                {"cls_id": 0, "bbox": [1, 1, 5, 5]}]}],
        },
        project_yaml="version: 1\n")          # every video untagged
    with pytest.raises(ValueError, match="matches no visual-prompt classes"):
        list(iter_project_runs(proj, parse_cli([])))
