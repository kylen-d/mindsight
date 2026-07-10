"""Offscreen coverage for UP1 D4 (File/Help menu entries) and the folded-in
flag fixes (#3 VP weights dir, #6 preflight weights-path message).

No models, no video -- these pin the menu wiring + two path-resolution fixes.
The autouse conftest fixture isolates ~/.mindsight; MainWindow construction is
the same lightweight offscreen build the namespace-census tests use.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _submenu(win, title):
    for act in win.menuBar().actions():
        if act.text() == title:
            return act.menu()
    raise AssertionError(f"menu {title!r} not found")


def _action(menu, text):
    for act in menu.actions():
        if act.text() == text:
            return act
    raise AssertionError(f"action {text!r} not found")


# ── D4: File menu project entries + Help menu ────────────────────────────────

def test_file_menu_has_project_entries_above_presets(qapp):
    from mindsight.GUI.main_window import MainWindow
    win = MainWindow()
    try:
        file_menu = _submenu(win, "&File")
        texts = [a.text() for a in file_menu.actions()]
        assert "&New Project..." in texts
        assert "&Open Project..." in texts
        # Project entries sit ABOVE the preset entries.
        assert texts.index("&New Project...") < texts.index("&Load Preset...")
        assert texts.index("&Open Project...") < texts.index("&Load Preset...")
    finally:
        win.close()


def test_file_menu_project_actions_delegate_and_switch_tab(qapp, monkeypatch):
    from mindsight.GUI.main_window import MainWindow
    win = MainWindow()
    try:
        called = []
        monkeypatch.setattr(win._run_study_tab, "new_project",
                            lambda: called.append("new"))
        monkeypatch.setattr(win._run_study_tab, "open_project_browse",
                            lambda: called.append("open"))
        file_menu = _submenu(win, "&File")
        win._tabs.setCurrentIndex(2)          # move off Analyze first
        _action(file_menu, "&New Project...").trigger()
        assert win._tabs.currentIndex() == 0  # switched to Analyze Footage
        win._tabs.setCurrentIndex(2)
        _action(file_menu, "&Open Project...").trigger()
        assert win._tabs.currentIndex() == 0
        assert called == ["new", "open"]
    finally:
        win.close()


def test_help_menu_has_docs_and_about(qapp):
    from mindsight.GUI.main_window import MainWindow
    win = MainWindow()
    try:
        help_menu = _submenu(win, "&Help")
        texts = [a.text() for a in help_menu.actions()]
        assert "Documentation" in texts
        assert "About MindSight" in texts
    finally:
        win.close()


def test_documentation_action_opens_docs_url(qapp, monkeypatch):
    from PyQt6.QtGui import QDesktopServices

    from mindsight.GUI.main_window import MainWindow
    win = MainWindow()
    try:
        opened = []
        monkeypatch.setattr(QDesktopServices, "openUrl",
                            lambda url: opened.append(url.toString()))
        _action(_submenu(win, "&Help"), "Documentation").trigger()
        assert opened == [MainWindow._DOCS_URL]
    finally:
        win.close()


def test_about_carries_version(qapp, monkeypatch):
    import mindsight.GUI.main_window as mw
    from mindsight import __version__
    from mindsight.GUI.main_window import MainWindow
    win = MainWindow()
    try:
        shown = []
        # Assert the About text carries __version__ WITHOUT exec'ing a modal.
        monkeypatch.setattr(mw.QMessageBox, "about",
                            lambda parent, title, text: shown.append(text))
        _action(_submenu(win, "&Help"), "About MindSight").trigger()
        assert shown and __version__ in shown[0]
    finally:
        win.close()


# ── Flag #3: VP builder model dropdown lists from WEIGHTS_ROOT ────────────────

def test_vp_builder_lists_models_from_weights_root(qapp, tmp_path, monkeypatch):
    import mindsight.weights as weights_mod
    fake_root = tmp_path / "WeightsHome"
    (fake_root / "YOLO").mkdir(parents=True)
    (fake_root / "YOLO" / "yoloe-99z-seg.pt").write_bytes(b"\x00")
    monkeypatch.setattr(weights_mod, "WEIGHTS_ROOT", fake_root)

    from mindsight.GUI.vp_builder_tab import VisualPromptBuilderTab
    tab = VisualPromptBuilderTab()
    items = [tab._test_model.itemText(i)
             for i in range(tab._test_model.count())]
    assert "yoloe-99z-seg.pt" in items


# ── Flag #6: preflight missing-weight message carries the absolute path ───────

def test_preflight_missing_weight_message_has_absolute_path(tmp_path,
                                                            monkeypatch):
    from mindsight.project import preflight

    resolved = tmp_path / "Weights" / "YOLO" / "yolov8n.pt"
    monkeypatch.setattr(
        preflight, "collect_weights",
        lambda ns: {"model": {"resolved": str(resolved), "sha256": "missing",
                              "backend": "YOLO"}},
        raising=False)
    # collect_weights is imported inside _check_weights from provenance; patch
    # there too.
    from mindsight.outputs import provenance
    monkeypatch.setattr(
        provenance, "collect_weights",
        lambda ns: {"model": {"resolved": str(resolved), "sha256": "missing",
                              "backend": "YOLO"}})

    result = preflight._check_weights(object())
    assert result.severity == "fail"
    assert str(resolved.resolve()) in result.message
    assert "shared Weights folder" in result.fix_hint
