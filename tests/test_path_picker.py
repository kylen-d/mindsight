"""W3Y item 5: known-location file pickers (path_picker module)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_known_candidates_enumerates_backend_dir(monkeypatch, tmp_path):
    import mindsight.weights as weights
    from mindsight.GUI import path_picker
    root = tmp_path / "Weights"
    (root / "MGaze").mkdir(parents=True)
    (root / "MGaze" / "b_gaze.onnx").write_bytes(b"x")
    (root / "MGaze" / "a.pt").write_bytes(b"x")
    (root / "MGaze" / "notes.txt").write_bytes(b"x")      # not a weight
    (root / "MGaze" / ".hidden.pt").write_bytes(b"x")     # hidden
    monkeypatch.setattr(weights, "WEIGHTS_ROOT", root)
    assert path_picker.known_candidates("mgaze_model") == ["a.pt", "b_gaze.onnx"]
    assert path_picker.default_browse_dir("mgaze_model") == str(root / "MGaze")


def test_unknown_dest_yields_no_candidates(monkeypatch, tmp_path):
    import mindsight.weights as weights
    from mindsight.GUI import path_picker
    monkeypatch.setattr(weights, "WEIGHTS_ROOT", tmp_path)
    assert path_picker.known_candidates("source") == []
    assert path_picker.default_browse_dir("source") == ""


def test_known_path_combo_is_lineedit_compatible(qapp):
    from mindsight.GUI.path_picker import KnownPathCombo
    w = KnownPathCombo(["a.pt", "b.pt"])
    assert w.text() == ""                       # starts empty, not first item
    seen = []
    w.textChanged.connect(seen.append)
    w.setText("custom/path.onnx")
    assert w.text() == "custom/path.onnx"
    assert "custom/path.onnx" in seen
    w.setText(None)
    assert w.text() == ""
    w.setPlaceholderText("pick one")            # must not raise
    assert [w.itemText(i) for i in range(w.count())] == ["a.pt", "b.pt"]


def test_vp_dir_memory_roundtrip(monkeypatch, tmp_path):
    from mindsight.GUI import path_picker
    from mindsight.GUI.settings_manager import SettingsManager
    monkeypatch.setattr(SettingsManager, "SETTINGS_DIR", tmp_path)
    vp = tmp_path / "prompts" / "toys.vp.json"
    vp.parent.mkdir()
    vp.write_text("{}")
    assert path_picker.vp_default_dir() == ""
    path_picker.remember_vp_dir(str(vp))
    assert path_picker.vp_default_dir() == str(vp.parent.resolve())
    assert path_picker.default_browse_dir("vp_file") == str(vp.parent.resolve())
