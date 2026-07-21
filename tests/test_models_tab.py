"""Fast offscreen coverage for the Models tab manager (SP4.1 Batch F, Step 10).

No network, no real weights: a tmp manifest + tmp weights dir drive the rows,
and ``mindsight.weights.download`` is monkeypatched so the Install flow never
touches the network (executor lesson 6).  These pin the manifest-driven manager
contract -- one row per entry, install of a missing weight, verify surfacing a
mismatch, and the unmanaged-custom-weight row.
"""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from mindsight import weights  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _sha(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _write_manifest(tmp_path, entries) -> str:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "weights": entries}))
    return str(path)


def _entry(backend, filename, data, *, required=False, label=None,
           source=weights.SOURCE_GITHUB, url="https://example.invalid/w"):
    return {
        "backend": backend, "filename": filename,
        "label": label or f"{backend} {filename}",
        "url": url if source == weights.SOURCE_GITHUB else None,
        "sha256": _sha(data) if source == weights.SOURCE_GITHUB else None,
        "size": len(data), "license": "MIT", "required": required,
        "source": source, "note": None,
    }


def _join_and_drain(tab):
    for t in list(tab._threads):
        t.join(timeout=5)
    tab._drain()


def _state_text(tab, row):
    from mindsight.GUI.models_tab import _COL_STATE
    return tab._table.item(row, _COL_STATE).text()


# ── Rendering ────────────────────────────────────────────────────────────────

def test_rows_render_from_tmp_manifest(qapp, tmp_path):
    from mindsight.GUI.models_tab import ModelsTab
    payload = b"gaze-weight-bytes"
    manifest = _write_manifest(tmp_path, [
        _entry("MGaze", "present.onnx", payload, required=True),
        _entry("MGaze", "absent.onnx", b"other"),
        _entry("MobileClip", "mobileclip_blt.ts", b"",
               source=weights.SOURCE_ULTRALYTICS_AUTO),
    ])
    wroot = tmp_path / "Weights"
    (wroot / "MGaze").mkdir(parents=True)
    (wroot / "MGaze" / "present.onnx").write_bytes(payload)

    tab = ModelsTab(manifest_path=manifest, weights_root=wroot)
    assert tab._table.rowCount() == 3
    # present file -> "present (unverified)"; absent -> MISSING; auto -> auto-fetch
    states = {tab._table.item(r, 0).text(): _state_text(tab, r)
              for r in range(3)}
    assert "present" in states["MGaze present.onnx"].lower()
    assert states["MGaze absent.onnx"] == "MISSING"
    assert "auto" in states["MobileClip mobileclip_blt.ts"].lower()


def test_verify_surfaces_mismatch(qapp, tmp_path):
    from mindsight.GUI.models_tab import ModelsTab
    manifest = _write_manifest(tmp_path, [
        _entry("MGaze", "w.onnx", b"the-published-bytes"),
    ])
    wroot = tmp_path / "Weights"
    (wroot / "MGaze").mkdir(parents=True)
    # On-disk bytes differ from the manifest sha -> mismatch.
    (wroot / "MGaze" / "w.onnx").write_bytes(b"tampered-local-copy")

    tab = ModelsTab(manifest_path=manifest, weights_root=wroot)
    tab._verify_all()
    _join_and_drain(tab)
    assert _state_text(tab, 0) == "mismatch"


def test_verify_ok(qapp, tmp_path):
    from mindsight.GUI.models_tab import ModelsTab
    data = b"good-bytes"
    manifest = _write_manifest(tmp_path, [_entry("MGaze", "w.onnx", data)])
    wroot = tmp_path / "Weights"
    (wroot / "MGaze").mkdir(parents=True)
    (wroot / "MGaze" / "w.onnx").write_bytes(data)

    tab = ModelsTab(manifest_path=manifest, weights_root=wroot)
    tab._verify_all()
    _join_and_drain(tab)
    assert _state_text(tab, 0) == "OK"


def test_install_flow_monkeypatched_download(qapp, tmp_path, monkeypatch):
    from mindsight.GUI.models_tab import ModelsTab
    data = b"freshly-downloaded-weight"
    manifest = _write_manifest(tmp_path, [_entry("MGaze", "new.onnx", data)])
    wroot = tmp_path / "Weights"

    def fake_download(entry, *, dest=None, progress=print, retries=2):
        progress(f"  Downloading {dest.name} ...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)          # correct bytes -> verify OK
        return dest

    monkeypatch.setattr(weights, "download", fake_download)

    tab = ModelsTab(manifest_path=manifest, weights_root=wroot)
    assert _state_text(tab, 0) == "MISSING"
    tab._start_download(0)
    _join_and_drain(tab)
    assert (wroot / "MGaze" / "new.onnx").exists()
    assert _state_text(tab, 0) == "OK"


def test_install_flow_offline_error_surfaces(qapp, tmp_path, monkeypatch):
    from mindsight.GUI.models_tab import ModelsTab
    manifest = _write_manifest(tmp_path, [_entry("MGaze", "new.onnx", b"x")])
    wroot = tmp_path / "Weights"

    def boom(entry, *, dest=None, progress=print, retries=2):
        raise weights.WeightsError("could not download new.onnx -- offline")

    monkeypatch.setattr(weights, "download", boom)

    tab = ModelsTab(manifest_path=manifest, weights_root=wroot)
    tab._start_download(0)
    _join_and_drain(tab)
    # Readable error on the status line, row falls back to MISSING (no crash).
    assert "offline" in tab._status.text().lower()
    assert _state_text(tab, 0) == "MISSING"


def test_unmanaged_custom_weight_row(qapp, tmp_path):
    from argparse import Namespace
    from mindsight.GUI.models_tab import ModelsTab

    manifest = _write_manifest(tmp_path, [_entry("MGaze", "w.onnx", b"a")])

    class FakeGaze:
        def _build_namespace(self):
            return Namespace(model="my_custom_yolo.pt")

    # collect_weights on that namespace resolves a custom file not in the
    # manifest -> it must appear as an unmanaged row.
    tab = ModelsTab(gaze_tab=FakeGaze(), manifest_path=manifest,
                    weights_root=tmp_path / "Weights")
    labels = [tab._table.item(r, 0).text() for r in range(tab._table.rowCount())]
    assert "my_custom_yolo.pt" in labels
    row = labels.index("my_custom_yolo.pt")
    assert "unmanaged" in _state_text(tab, row).lower()


def test_optimal_for_device_tag(qapp, tmp_path):
    """Manifest entries whose 'optimal' list matches this machine's device
    class show 'optimal for this device' in the tag column."""
    from mindsight.GUI.models_tab import ModelsTab
    e_match = _entry("MGaze", "match_gaze.onnx", b"a", required=True)
    e_match["optimal"] = ["mps", "cpu"]
    e_other = _entry("MGaze", "other.pt", b"b")
    e_other["optimal"] = ["cuda"]
    manifest = _write_manifest(tmp_path, [e_match, e_other])
    wroot = tmp_path / "Weights"
    (wroot / "MGaze").mkdir(parents=True)

    ModelsTab._device_class_cache = "cpu"
    try:
        tab = ModelsTab(manifest_path=manifest, weights_root=wroot)
        from mindsight.GUI.models_tab import _COL_REQUIRED
        tags = {tab._table.item(r, 0).text():
                tab._table.item(r, _COL_REQUIRED).text()
                for r in range(tab._table.rowCount())}
        assert tags["MGaze match_gaze.onnx"] == "required, optimal for this device"
        assert tags["MGaze other.pt"] == ""
    finally:
        ModelsTab._device_class_cache = None


def test_license_column_last_with_summary_and_details_popup(qapp, tmp_path,
                                                            monkeypatch):
    """W4C rework (user request): License is the LAST column; the cell shows
    a short summary (SPDX id, '(research-only)' tag when the note says so)
    and a 'details' link that pops the full license_note -- the
    paragraph-length notes no longer render inline."""
    from mindsight.GUI.models_tab import _COL_LICENSE, _COLS, ModelsTab
    assert _COLS[-1] == "License" and _COL_LICENSE == len(_COLS) - 1

    plain = _entry("MGaze", "plain.onnx", b"a")
    noted = _entry("MGaze", "noted.onnx", b"b")
    noted["license_note"] = "weights: research use only - trained on X"
    manifest = _write_manifest(tmp_path, [plain, noted])
    wroot = tmp_path / "Weights"
    (wroot / "MGaze").mkdir(parents=True)

    tab = ModelsTab(manifest_path=manifest, weights_root=wroot)
    rows = {tab._table.item(r, 0).text(): r
            for r in range(tab._table.rowCount())}

    # Plain id: ordinary short cell, no link widget.
    r = rows["MGaze plain.onnx"]
    assert tab._table.item(r, _COL_LICENSE).text() == "MIT"
    assert tab._table.cellWidget(r, _COL_LICENSE) is None

    # Noted: summary + details link; full note lives in tooltip + popup.
    r = rows["MGaze noted.onnx"]
    label = tab._table.cellWidget(r, _COL_LICENSE)
    assert label is not None
    assert "MIT (research-only)" in label.text() and "details" in label.text()
    assert "research use only" in label.toolTip()

    popped = {}
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QMessageBox.information",
        lambda _parent, title, text: popped.update(title=title, text=text))
    label.linkActivated.emit("#")
    assert "noted.onnx" in popped["title"]
    assert "research use only - trained on X" in popped["text"]
