"""UP2 Batch A -- RunSettings store + Analyze Footage decoupling.

Offscreen, model-free.  The ``_isolate_mindsight_settings`` autouse fixture
(tests/conftest.py) repoints ``SettingsManager.SETTINGS_DIR`` at a per-test temp
dir, and the store keys ``run_settings.json`` off that same dir, so every test
here runs against a fresh fake HOME with no touch of the real ~/.mindsight.
"""

import json
import os
import threading
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _weights_root():
    from mindsight import constants
    return constants.PROJECT_ROOT / "Weights"


def _make_project(tmp_path, *, pipeline_yaml=None):
    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    (proj / "Inputs" / "Videos" / "a.mp4").write_bytes(b"\x00" * 32)
    if pipeline_yaml is not None:
        (proj / "Pipeline").mkdir()
        (proj / "Pipeline" / "pipeline.yaml").write_text(pipeline_yaml)
    return proj


# ── A4.1 Seeding ─────────────────────────────────────────────────────────────

def test_seed_from_shipped_preset(qapp):
    from mindsight.GUI.run_settings import RunSettingsStore
    store = RunSettingsStore()
    ns = store.ns()
    # KG_Standard sentinels (configs/pipeline_known_good.yaml).
    assert ns.merge_overlaps is True
    assert ns.min_call_gap == 25
    assert ns.ray_length == 1.3
    assert ns.smooth_snap == "all"
    assert store.source_label() == "KG_Standard"
    assert store.is_modified() is False


def test_persisted_run_settings_wins_over_preset(qapp):
    from mindsight.GUI.run_settings import RunSettingsStore
    RunSettingsStore._path().parent.mkdir(parents=True, exist_ok=True)
    RunSettingsStore._path().write_text(json.dumps(
        {"ray_length": 3.3, "_source_label": "custom", "_version": 1}))
    store = RunSettingsStore()
    assert store.ns().ray_length == 3.3         # persisted state wins
    assert store.source_label() == "custom"


def test_corrupt_run_settings_falls_back_to_preset(qapp, capsys):
    from mindsight.GUI.run_settings import RunSettingsStore
    RunSettingsStore._path().parent.mkdir(parents=True, exist_ok=True)
    RunSettingsStore._path().write_text("{not json")
    store = RunSettingsStore()                  # must not raise
    assert store.ns().ray_length == 1.3         # preset fallback
    assert store.source_label() == "KG_Standard"
    assert "[WARN]" in capsys.readouterr().out


# ── A4.2 Persistence round-trip + portable weight reduction ──────────────────

def test_weight_dest_basename_reduction_on_persist(qapp):
    from mindsight.GUI.run_settings import RunSettingsStore
    store = RunSettingsStore()
    ns = store.working_copy()
    ns.mgaze_model = str(_weights_root() / "MGaze" / "custom.onnx")  # under root
    ns.model = "/foreign/abs/yolo.pt"                                # foreign
    store.commit(ns)

    saved = json.loads(RunSettingsStore._path().read_text())
    assert saved["mgaze_model"] == "custom.onnx"       # reduced to bare name
    assert saved["model"] == "/foreign/abs/yolo.pt"    # foreign path preserved

    # A fresh store seeds from the persisted (portable) state.
    store2 = RunSettingsStore()
    assert store2.ns().mgaze_model == "custom.onnx"
    assert store2.ns().model == "/foreign/abs/yolo.pt"


def test_absolute_default_mgaze_reduced(qapp):
    """The absolute-default trap: mgaze_model's parser default is an absolute
    path baked at parse time; persistence must basename-reduce it."""
    from mindsight.GUI.run_settings import RunSettingsStore
    from mindsight.cli_flags import parse_cli
    default_mgaze = parse_cli([]).mgaze_model
    assert os.path.isabs(default_mgaze)                # the trap exists
    store = RunSettingsStore()
    ns = store.working_copy()
    ns.mgaze_model = default_mgaze
    store.commit(ns)
    saved = json.loads(RunSettingsStore._path().read_text())
    assert saved["mgaze_model"] == os.path.basename(default_mgaze)
    assert not os.path.isabs(saved["mgaze_model"])


# ── A4.3 Modified tracking ───────────────────────────────────────────────────

def test_is_modified_lifecycle(qapp):
    from mindsight.GUI.run_settings import RunSettingsStore
    store = RunSettingsStore()
    assert store.is_modified() is False
    ns = store.working_copy()
    ns.ray_length = 9.0
    store.commit(ns)
    assert store.is_modified() is True
    store.reset_to_preset()
    assert store.is_modified() is False
    assert store.ns().ray_length == 1.3
    assert store.source_label() == "KG_Standard"


def test_is_modified_catches_non_schema_dest(qapp):
    """canonical_hash omits model/plugin dests; the dict signature catches them."""
    from mindsight.GUI.run_settings import RunSettingsStore
    store = RunSettingsStore()
    ns = store.working_copy()
    ns.mgaze_model = "some_other_family"      # a model-wiring dest (out of schema)
    store.commit(ns)
    assert store.is_modified() is True


# ── A4.4 Decoupling regression (the key test) ────────────────────────────────

def test_gaze_tuning_does_not_feed_run_config(qapp):
    from mindsight.GUI.gaze_tab import GazeTab
    from mindsight.GUI.run_settings import RunSettingsStore
    from mindsight.GUI.run_study_tab import RunStudyTab

    store = RunSettingsStore()
    gaze = GazeTab()
    tab = RunStudyTab(gaze_tab=gaze, settings=store)
    try:
        baseline = tab._current_ns().ray_length         # store preset (1.3)

        gns = gaze._build_namespace()
        gns.ray_length = 9.99
        gaze.apply_namespace(gns)                        # edit Gaze Tuning widget
        assert tab._current_ns().ray_length == baseline  # NOT reflected

        sns = store.working_copy()
        sns.ray_length = 7.77
        store.commit(sns)                                # edit the store
        assert tab._current_ns().ray_length == 7.77      # IS reflected
    finally:
        gaze.deleteLater()
        tab.deleteLater()


# ── A4.5 Project open loads pipeline into the store ──────────────────────────

def test_project_open_loads_pipeline_into_store(qapp, tmp_path):
    from mindsight.GUI.gaze_tab import GazeTab
    from mindsight.GUI.run_settings import RunSettingsStore
    from mindsight.GUI.run_study_tab import RunStudyTab

    proj = _make_project(tmp_path, pipeline_yaml="gaze:\n  ray_length: 4.2\n")
    store = RunSettingsStore()
    gaze = GazeTab()
    gaze_before = gaze._build_namespace().ray_length
    tab = RunStudyTab(gaze_tab=gaze, settings=store)
    try:
        tab._open_project(str(proj))
        assert tab._current_ns().ray_length == 4.2       # store carries it
        assert store.source_label() == "project pipeline"
        # Gaze Tuning widgets untouched by project open (decoupling).
        assert gaze._build_namespace().ray_length == gaze_before
    finally:
        gaze.deleteLater()
        tab.deleteLater()


# ── A4.6 Output-toggle mapping (one-off launch layer) ────────────────────────

class _FakeGazeWorker(threading.Thread):
    instances: list = []

    def __init__(self, ns, frame_q, log_q, dashboard_q=None):
        super().__init__(daemon=True)
        self.ns = ns
        self._alive = False
        _FakeGazeWorker.instances.append(self)

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive

    def stop(self):
        self._alive = False


def _one_off_dlg(project):
    return SimpleNamespace(
        video=str(project / "Inputs" / "Videos" / "a.mp4"),
        meta={}, output_dir=None, move=False)


def _launch_and_capture(qapp, tmp_path, monkeypatch, store=None):
    import mindsight.GUI.workers as workers_mod
    from mindsight.GUI.run_settings import RunSettingsStore
    from mindsight.GUI.run_study_tab import RunStudyTab

    monkeypatch.setattr(workers_mod, "GazeWorker", _FakeGazeWorker)
    _FakeGazeWorker.instances = []
    proj = _make_project(tmp_path)
    tab = RunStudyTab(settings=store or RunSettingsStore())
    tab._open_project(str(proj))
    tab._run_single_run(_one_off_dlg(proj))
    tab._poll_timer.stop()
    assert _FakeGazeWorker.instances
    return _FakeGazeWorker.instances[-1].ns


def test_one_off_default_sets_all_four_paths(qapp, tmp_path, monkeypatch):
    ns = _launch_and_capture(qapp, tmp_path, monkeypatch)
    # Events + summary + video + heatmap all set (today's behavior, all-on store).
    assert ns.log and ns.summary and ns.save and ns.heatmap
    # Charts have no per-run output path -> never written to disk here.
    assert ns.charts is None


def test_one_off_save_off_drops_video_only(qapp, tmp_path, monkeypatch):
    from mindsight.GUI.run_settings import RunSettingsStore
    store = RunSettingsStore()
    sns = store.working_copy()
    sns.save = False
    store.commit(sns)
    ns = _launch_and_capture(qapp, tmp_path, monkeypatch, store=store)
    assert ns.save is None                       # video suppressed
    assert ns.log and ns.summary and ns.heatmap  # everything else still set


def test_want_artifact_semantics(qapp):
    from argparse import Namespace

    from mindsight.GUI.run_settings import want_artifact
    assert want_artifact(Namespace(save=True), "save") is True
    assert want_artifact(Namespace(save=False), "save") is False
    assert want_artifact(Namespace(save=None), "save") is True   # absent -> produce
    assert want_artifact(Namespace(), "save") is True            # missing -> produce
