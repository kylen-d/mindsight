"""UP2 Batch B -- the Inference Settings dialog + its layout contract.

The census test (B3.1) is headless.  The dialog tests (B3.2-B3.6) run offscreen
Qt against the isolated fake HOME (conftest ``_isolate_mindsight_settings`` +
the store keying ``run_settings.json`` off ``SettingsManager.SETTINGS_DIR``), so
nothing here touches the real ~/.mindsight.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── B3.1 Census (headless -- no Qt) ──────────────────────────────────────────

def test_every_spec_dest_exists_in_parser_census():
    from mindsight.GUI.inference_settings.spec import (
        all_dests,
        parser_census_dests,
    )
    spec_dests = all_dests()
    census = parser_census_dests()
    orphans = sorted(spec_dests - census)
    assert not orphans, f"SETTINGS_SPEC dests absent from the parser: {orphans}"


def test_dropped_dests_never_appear():
    from mindsight.GUI.inference_settings.spec import DROPPED_DESTS, all_dests
    leaking = sorted(all_dests() & DROPPED_DESTS)
    assert not leaking, f"dropped dests leaked into the dialog: {leaking}"


def test_spec_shape_is_stable():
    """Seven tabs; the per-tab field/toggle dest counts are pinned (a field
    silently vanishing changes runs)."""
    from mindsight.GUI.inference_settings.spec import (
        SETTINGS_SPEC,
        all_dests,
        tab_field_dests,
    )
    assert len(SETTINGS_SPEC) == 7
    counts = {t.key: len(tab_field_dests(t)) for t in SETTINGS_SPEC}
    assert counts == {
        "models": 10, "gaze": 41, "detection": 14, "phenomena": 42,
        "output": 7, "performance": 5, "experimental": 22,
    }
    # rf_gazelle_model is the one dest carried on two tabs (Models value +
    # Gaze blend enable), so unique dests = sum(counts) - 1.
    assert len(all_dests()) == sum(counts.values()) - 1 == 140


def test_every_field_has_label_and_description_or_tooltip():
    from mindsight.GUI.inference_settings.spec import field_meta, iter_fields
    for f in iter_fields():
        meta = field_meta(f.dest)
        assert f.label, f.dest
        assert f.description or meta.tooltip, f.dest


# ── Qt fixtures ──────────────────────────────────────────────────────────────

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _store():
    from mindsight.GUI.run_settings import RunSettingsStore
    return RunSettingsStore()


def _dialog(store=None, **kw):
    from mindsight.GUI.inference_settings import InferenceSettingsDialog
    return InferenceSettingsDialog(store or _store(), **kw)


# ── B3.2 Offscreen build ─────────────────────────────────────────────────────

def test_dialog_builds_all_tabs_and_registers_every_dest(qapp):
    from mindsight.GUI.inference_settings.spec import all_dests
    dlg = _dialog()
    try:
        assert dlg._stack.count() == 7
        registered = set(dlg._controls) | set(dlg._toggles)
        assert registered == all_dests()
    finally:
        dlg.deleteLater()


def test_dialog_widgets_carry_a_tooltip(qapp):
    dlg = _dialog()
    try:
        for c in dlg._controls.values():
            tip = c["field"].description or c["meta"].tooltip
            assert tip, c["field"].dest
    finally:
        dlg.deleteLater()


# ── B2 entry points ──────────────────────────────────────────────────────────

def test_tools_menu_has_inference_settings(qapp):
    from mindsight.GUI.main_window import MainWindow
    win = MainWindow()
    try:
        tools = None
        for act in win.menuBar().actions():
            if act.text() == "&Tools":
                tools = act.menu()
        assert tools is not None
        assert "Inference Settings..." in [a.text() for a in tools.actions()]
    finally:
        win.close()


def test_tab_opens_inference_dialog_without_project(qapp, monkeypatch):
    from mindsight.GUI import inference_settings as is_mod
    from mindsight.GUI.run_study_tab import RunStudyTab

    opened = {}

    class _FakeDlg:
        def __init__(self, store, parent=None, *, gaze_tab=None,
                     project_pipeline_path=None):
            opened["store"] = store
            opened["pipe"] = project_pipeline_path

        def exec(self):
            opened["exec"] = True

    monkeypatch.setattr(is_mod, "InferenceSettingsDialog", _FakeDlg)
    tab = RunStudyTab()
    try:
        tab._open_inference_settings()          # no project open
        assert opened.get("exec") is True
        assert opened["pipe"] is None           # quick mode -> no project path
    finally:
        tab.deleteLater()


# ── B3.3 Round-trip / cancel / reset ─────────────────────────────────────────

def test_apply_round_trips_three_tabs(qapp):
    store = _store()
    dlg = _dialog(store)
    try:
        dlg._controls["ray_length"]["w"].setValue(2.5)       # gaze tab
        dlg._controls["conf"]["w"].setValue(0.6)             # detection tab
        dlg._controls["skip_frames"]["w"].setValue(4)        # performance tab
        dlg._on_apply()
        ns = store.ns()
        assert ns.ray_length == 2.5
        assert ns.conf == 0.6
        assert ns.skip_frames == 4
    finally:
        dlg.deleteLater()


def test_cancel_discards_edits(qapp):
    store = _store()
    before = store.ns().ray_length
    dlg = _dialog(store)
    try:
        dlg._controls["ray_length"]["w"].setValue(4.9)
        dlg.reject()                       # Cancel -- never commits
        assert store.ns().ray_length == before
    finally:
        dlg.deleteLater()


def test_reset_restores_kg_sentinels(qapp):
    store = _store()
    dlg = _dialog(store)
    try:
        dlg._controls["ray_length"]["w"].setValue(4.9)
        dlg._on_apply()
        assert store.ns().ray_length == 4.9
        dlg._on_reset()
        ns = store.ns()
        assert ns.ray_length == 1.3          # KG_Standard sentinels restored
        assert ns.smooth_snap == "all"
        assert store.source_label() == "KG_Standard"
    finally:
        dlg.deleteLater()


# ── B3.4 SliderValue over-range (Q14) ────────────────────────────────────────

def test_slidervalue_preserves_over_range_and_flags(qapp):
    from mindsight.GUI.inference_settings.widgets import SliderValue
    sv = SliderValue(is_int=False, minimum=0.2, maximum=5.0, step=0.1,
                     decimals=1)
    sv.setValue(9.9)                         # beyond the recommended max
    assert sv.value() == 9.9                 # NOT clamped
    assert sv.is_over_range() is True
    assert sv._slider.isEnabled() is False   # slider greys...
    assert sv._slider.value() == sv._steps   # ...and PINS to the max end (B2)
    sv.setValue(0.05)                        # below the recommended min
    assert sv._slider.value() == 0           # pins to the min end
    sv.setValue(2.0)                         # back in range
    assert sv.is_over_range() is False
    assert sv._slider.isEnabled() is True
    assert sv._slider.value() == sv._val_to_tick(2.0)


# ── B3.5 Save-to-project ─────────────────────────────────────────────────────

def test_save_to_project_writes_roundtrippable_yaml(qapp, tmp_path, monkeypatch):
    from argparse import Namespace

    import mindsight.GUI.inference_settings.dialog as dlg_mod
    from mindsight.config_compat import load_pipeline

    # The success path pops a modal QMessageBox -- neutralize it so the
    # offscreen test does not block.
    monkeypatch.setattr(dlg_mod.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    pipe = tmp_path / "proj" / "Pipeline" / "pipeline.yaml"
    store = _store()
    dlg = _dialog(store, project_pipeline_path=pipe)
    try:
        assert dlg._save_proj_btn.isEnabled()
        dlg._controls["ray_length"]["w"].setValue(3.1)
        dlg._on_save_project()
        assert pipe.is_file()
        loaded = load_pipeline(str(pipe), Namespace())
        assert loaded.ray_length == 3.1
        assert store.source_label() == "project pipeline"
    finally:
        dlg.deleteLater()


def test_save_to_project_disabled_without_project(qapp):
    dlg = _dialog()
    try:
        assert dlg._save_proj_btn.isEnabled() is False
    finally:
        dlg.deleteLater()


# ── B3.6 Blend owner semantics ───────────────────────────────────────────────

def test_blend_enable_with_model_present(qapp):
    store = _store()
    dlg = _dialog(store)
    try:
        dlg._controls["rf_gazelle_model"]["w"].setText("my_gazelle.pt")  # Tab 1
        dlg._toggles["rf_gazelle_model"]["box"].setChecked(True)         # Tab 2
        dlg._on_apply()
        assert store.ns().rf_gazelle_model == "my_gazelle.pt"
    finally:
        dlg.deleteLater()


def test_blend_disable_clears_model(qapp):
    store = _store()
    dlg = _dialog(store)
    try:
        dlg._controls["rf_gazelle_model"]["w"].setText("my_gazelle.pt")
        dlg._toggles["rf_gazelle_model"]["box"].setChecked(False)
        dlg._on_apply()
        assert store.ns().rf_gazelle_model is None
    finally:
        dlg.deleteLater()


def test_blend_enable_without_model_shows_hint(qapp, monkeypatch):
    store = _store()
    dlg = _dialog(store)
    try:
        # No model on Tab 1 and no resolvable default checkpoint -> amber hint,
        # empty (not None) value; preflight remains the hard gate.
        monkeypatch.setattr(dlg, "_resolve_default_gazelle", lambda _ns: None)
        dlg._controls["rf_gazelle_model"]["w"].setText("")
        dlg._toggles["rf_gazelle_model"]["box"].setChecked(True)
        dlg._update_blend_hint()
        # isHidden() reflects the widget's own shown state (the Gaze tab sits on
        # a non-current QStackedWidget page, so isVisible/isVisibleTo are False).
        assert dlg._blend_hint.isHidden() is False
        dlg._on_apply()
        assert store.ns().rf_gazelle_model == ""
    finally:
        dlg.deleteLater()
