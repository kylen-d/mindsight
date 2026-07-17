"""Tests for the checksummed weights manifest + verified downloads (SP4 Batch B)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from mindsight import weights

# The required set (Q6, +yolo11n.pt for the v1.1 3.8 default flip) --
# exactly these five filenames.
REQUIRED_FILENAMES = {
    "yolov8n.pt",
    "yolo11n.pt",
    "gazelle_dinov2_vitb14.pt",
    "resnet50_gaze.onnx",
    "mobileone_s0_gaze.onnx",
}
_ALLOWED_SOURCES = {weights.SOURCE_GITHUB, weights.SOURCE_ULTRALYTICS_AUTO}


# ══════════════════════════════════════════════════════════════════════════════
# Committed manifest shape (D3)
# ══════════════════════════════════════════════════════════════════════════════

def test_committed_manifest_loads():
    data = weights.load_manifest()
    assert data["schema_version"] == 1
    assert isinstance(data["weights"], list) and data["weights"]


def test_committed_manifest_required_set_is_exactly_five():
    required = {e["filename"] for e in weights.manifest_entries() if e["required"]}
    assert required == REQUIRED_FILENAMES


def test_committed_manifest_entry_shape():
    for e in weights.manifest_entries():
        for field in ("backend", "filename", "label", "url", "sha256",
                      "size", "license", "required", "source", "note"):
            assert field in e, f"{e.get('filename')} missing {field}"
        assert isinstance(e["backend"], str) and e["backend"]
        assert isinstance(e["filename"], str) and e["filename"]
        assert isinstance(e["label"], str) and e["label"]
        assert isinstance(e["required"], bool)
        assert e["source"] in _ALLOWED_SOURCES
        if e["source"] == weights.SOURCE_ULTRALYTICS_AUTO:
            assert e["url"] is None and e["sha256"] is None
            assert e["note"]  # a note is mandatory for auto-fetch entries
        else:
            assert e["url"].startswith("https://")
            assert isinstance(e["sha256"], str) and len(e["sha256"]) == 64
            assert isinstance(e["size"], int) and e["size"] > 0


def test_committed_manifest_labels_use_paper_terms():
    # T8: user-facing labels say MobileGaze / Gaze-LLE, never MGaze/Gazelle.
    for e in weights.manifest_entries():
        assert "MGaze" not in e["label"]
        assert "Gazelle" not in e["label"]


# ══════════════════════════════════════════════════════════════════════════════
# Lookup
# ══════════════════════════════════════════════════════════════════════════════

def test_find_entry_by_filename_and_backend():
    e = weights.find_entry("resnet50_gaze.onnx")
    assert e is not None and e["backend"] == "MGaze"
    assert weights.find_entry("resnet50_gaze.onnx", backend="YOLO") is None
    # A directory-prefixed path resolves by basename.
    assert weights.find_entry("Weights/MGaze/resnet50_gaze.onnx") is not None
    assert weights.find_entry("nope_custom.onnx") is None


# ══════════════════════════════════════════════════════════════════════════════
# Hashing single-source
# ══════════════════════════════════════════════════════════════════════════════

def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "blob.bin"
    p.write_bytes(b"mindsight-weights")
    assert weights.sha256_file(p) == hashlib.sha256(b"mindsight-weights").hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# verify()
# ══════════════════════════════════════════════════════════════════════════════

def _entry(tmp_path, data=b"payload"):
    return {"backend": "MGaze", "filename": "x.onnx", "label": "X",
            "url": "https://example.invalid/x.onnx",
            "sha256": weights.sha256_file(_write(tmp_path / "ref", data)),
            "size": len(data), "license": "MIT", "required": False,
            "source": weights.SOURCE_GITHUB, "note": None}


def _write(p, data):
    p.write_bytes(data)
    return p


def test_verify_missing(tmp_path):
    e = _entry(tmp_path)
    assert weights.verify(tmp_path / "absent.onnx", e) == weights.MISSING


def test_verify_ok(tmp_path):
    e = _entry(tmp_path, b"payload")
    good = _write(tmp_path / "good.onnx", b"payload")
    assert weights.verify(good, e) == weights.OK


def test_verify_mismatch(tmp_path):
    e = _entry(tmp_path, b"payload")
    bad = _write(tmp_path / "bad.onnx", b"tampered")
    assert weights.verify(bad, e) == weights.MISMATCH


def test_verify_no_sha_present_is_ok(tmp_path):
    e = {"sha256": None}
    present = _write(tmp_path / "auto.ts", b"anything")
    assert weights.verify(present, e) == weights.OK
    assert weights.verify(tmp_path / "gone.ts", e) == weights.MISSING


# ══════════════════════════════════════════════════════════════════════════════
# download() -- monkeypatched urlretrieve, NO network
# ══════════════════════════════════════════════════════════════════════════════

def _fake_urlretrieve(data):
    def _f(url, filename):
        with open(filename, "wb") as fh:
            fh.write(data)
        return filename, {}
    return _f


def test_download_success(tmp_path, monkeypatch):
    data = b"the-real-weight-bytes"
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve(data))
    e = {"backend": "MGaze", "filename": "w.onnx",
         "url": "https://example.invalid/w.onnx",
         "sha256": weights.sha256_file(_write(tmp_path / "ref", data)),
         "source": weights.SOURCE_GITHUB}
    dest = tmp_path / "out" / "w.onnx"
    got = weights.download(e, dest=dest, progress=lambda *a: None)
    assert got == dest and dest.read_bytes() == data
    assert not (tmp_path / "out" / "w.onnx.part").exists()


def test_download_sha_mismatch_deletes_and_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve(b"corrupt"))
    e = {"backend": "MGaze", "filename": "w.onnx",
         "url": "https://example.invalid/w.onnx",
         "sha256": "0" * 64, "source": weights.SOURCE_GITHUB}
    dest = tmp_path / "w.onnx"
    with pytest.raises(weights.WeightsError) as exc:
        weights.download(e, dest=dest, progress=lambda *a: None)
    assert "checksum" in str(exc.value).lower()
    assert not dest.exists()
    assert not dest.with_name("w.onnx.part").exists()


def test_download_offline_readable_error(tmp_path, monkeypatch):
    def _boom(url, filename):
        raise urllib.error.URLError("Name or service not known")
    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
    e = {"backend": "MGaze", "filename": "w.onnx",
         "url": "https://example.invalid/w.onnx",
         "sha256": "0" * 64, "source": weights.SOURCE_GITHUB}
    dest = tmp_path / "w.onnx"
    with pytest.raises(weights.WeightsError) as exc:
        weights.download(e, dest=dest, progress=lambda *a: None, retries=1)
    msg = str(exc.value)
    assert "could not download" in msg.lower()
    assert "internet" in msg.lower()
    assert "Traceback" not in msg
    assert not dest.exists()
    assert not dest.with_name("w.onnx.part").exists()


def test_download_partial_cleanup_between_retries(tmp_path, monkeypatch):
    calls = {"n": 0}
    data = b"good-bytes"

    def _flaky(url, filename):
        calls["n"] += 1
        with open(filename, "wb") as fh:
            fh.write(b"partial")   # a partial file is written
        if calls["n"] == 1:
            raise urllib.error.URLError("dropped")
        with open(filename, "wb") as fh:
            fh.write(data)         # second attempt succeeds
        return filename, {}

    monkeypatch.setattr(urllib.request, "urlretrieve", _flaky)
    e = {"backend": "MGaze", "filename": "w.onnx",
         "url": "https://example.invalid/w.onnx",
         "sha256": weights.sha256_file(_write(tmp_path / "ref", data)),
         "source": weights.SOURCE_GITHUB}
    dest = tmp_path / "w.onnx"
    got = weights.download(e, dest=dest, progress=lambda *a: None, retries=2)
    assert got.read_bytes() == data and calls["n"] == 2
    assert not dest.with_name("w.onnx.part").exists()


def test_download_ultralytics_auto_refuses(tmp_path):
    e = {"backend": "MobileClip", "filename": "mobileclip_blt.ts",
         "url": None, "sha256": None, "source": weights.SOURCE_ULTRALYTICS_AUTO}
    with pytest.raises(weights.WeightsError) as exc:
        weights.download(e, dest=tmp_path / "mc.ts", progress=lambda *a: None)
    assert "automatically" in str(exc.value).lower()


# ══════════════════════════════════════════════════════════════════════════════
# downloadable_missing (preflight one-click fetch -- consume, don't compute)
# ══════════════════════════════════════════════════════════════════════════════

def test_downloadable_missing_filters(tmp_path):
    data = b"present-bytes"
    present = _entry(tmp_path, data)      # sha of tmp_path/ref
    present["filename"] = "present.onnx"
    absent = dict(present, filename="absent.onnx")
    auto = {"backend": "MobileClip", "filename": "mobileclip_blt.ts",
            "label": "MobileCLIP", "url": None, "sha256": None, "size": None,
            "license": "x", "required": False,
            "source": weights.SOURCE_ULTRALYTICS_AUTO, "note": "auto"}
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(
        {"schema_version": 1, "weights": [present, absent, auto]}))

    # present.onnx exists on disk (its entry_dest is Weights/MGaze/... in the
    # repo, so it will not be present there) -- to test presence deterministically
    # we point verify at what find_entry resolves.  Both present.onnx and
    # absent.onnx resolve to the same repo Weights dir; neither is on disk there,
    # so both are missing+downloadable, while the auto entry is excluded.
    got = weights.downloadable_missing(
        ["present.onnx", "absent.onnx", "mobileclip_blt.ts", "custom.onnx"],
        path=manifest)
    names = {e["filename"] for e in got}
    assert "mobileclip_blt.ts" not in names   # auto-fetch excluded
    assert "custom.onnx" not in names         # not in the manifest
    assert names <= {"present.onnx", "absent.onnx"}


def test_downloadable_missing_skips_present(tmp_path, monkeypatch):
    e = _entry(tmp_path, b"x")
    e["filename"] = "w.onnx"
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"schema_version": 1, "weights": [e]}))
    # Force verify to report the file present -> it must be filtered out.
    monkeypatch.setattr(weights, "verify", lambda p, entry: weights.OK)
    assert weights.downloadable_missing(["w.onnx"], path=manifest) == []


# ══════════════════════════════════════════════════════════════════════════════
# load_manifest error paths
# ══════════════════════════════════════════════════════════════════════════════

def test_load_manifest_missing(tmp_path):
    with pytest.raises(weights.WeightsError):
        weights.load_manifest(tmp_path / "nope.json")


def test_load_manifest_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(weights.WeightsError):
        weights.load_manifest(p)


def test_load_manifest_no_weights_key(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(weights.WeightsError):
        weights.load_manifest(p)


# ══════════════════════════════════════════════════════════════════════════════
# CLI (mindsight-weights) against a tmp manifest, NO network
# ══════════════════════════════════════════════════════════════════════════════

def _tmp_manifest(tmp_path, data=b"payload"):
    wpath = tmp_path / "w.onnx"
    entry = {"backend": "MGaze", "filename": "w.onnx", "label": "MobileGaze test",
             "url": "https://example.invalid/w.onnx",
             "sha256": weights.sha256_file(_write(wpath, data)),
             "size": len(data), "license": "MIT", "required": True,
             "source": weights.SOURCE_GITHUB, "note": None}
    mpath = tmp_path / "weights_manifest.json"
    mpath.write_text(json.dumps({"schema_version": 1, "weights": [entry]}))
    return mpath, entry


def test_cli_verify_only_reports_missing(tmp_path, capsys, monkeypatch):
    mpath, _ = _tmp_manifest(tmp_path)
    # resolve_weight would look under the repo Weights/ dir -> the file is absent.
    rc = weights.main(["--verify-only", "--manifest", str(mpath)])
    out = capsys.readouterr().out
    assert rc == 1 and "MISSING" in out


def test_cli_dry_run(tmp_path, capsys):
    mpath, _ = _tmp_manifest(tmp_path)
    rc = weights.main(["--all", "--dry-run", "--manifest", str(mpath)])
    out = capsys.readouterr().out
    assert rc == 0 and "dry-run" in out


def test_cli_missing_manifest_returns_1(tmp_path, capsys):
    rc = weights.main(["--required", "--manifest", str(tmp_path / "nope.json")])
    out = capsys.readouterr().out
    assert rc == 1 and "ERROR" in out


# ── Device-switching MobileGaze family names (user ruling 2026-07-09) ────────

def _fake_device(monkeypatch, dev_type):
    import torch

    monkeypatch.setattr("mindsight.utils.device.resolve_device",
                        lambda device="auto": torch.device(dev_type))


def test_mgaze_family_picks_pt_on_cuda(monkeypatch):
    _fake_device(monkeypatch, "cpu")  # ensure the patch target is importable
    monkeypatch.setattr("mindsight.utils.device.resolve_device",
                        lambda device="auto": __import__("torch").device("cuda"))
    assert weights.resolve_mgaze_family("resnet50") == "resnet50.pt"
    assert weights.resolve_mgaze_family("mobileone_s0") == "mobileone_s0.pt"


def test_mgaze_family_picks_onnx_elsewhere(monkeypatch):
    for dev in ("mps", "cpu"):
        _fake_device(monkeypatch, dev)
        assert weights.resolve_mgaze_family("resnet50") == "resnet50_gaze.onnx"


def test_mgaze_family_explicit_extension_wins(monkeypatch):
    _fake_device(monkeypatch, "cpu")
    assert weights.resolve_mgaze_family("resnet50.pt") == "resnet50.pt"
    assert (weights.resolve_mgaze_family("resnet34_gaze.onnx")
            == "resnet34_gaze.onnx")


def test_committed_manifest_optimal_tags_cover_mgaze():
    """Every MGaze entry carries an optimal-device tag: .pt -> cuda,
    .onnx -> mps/cpu (drives the Models tab 'optimal for this device')."""
    entries = [e for e in weights.manifest_entries() if e["backend"] == "MGaze"]
    assert entries
    for e in entries:
        if e["filename"].endswith(".pt"):
            assert e.get("optimal") == ["cuda"], e["filename"]
        else:
            assert e.get("optimal") == ["mps", "cpu"], e["filename"]


def test_mgaze_from_args_resolves_family_on_this_machine():
    """End to end: an extensionless mgaze_model builds the right engine for
    this machine (onnx on non-CUDA boxes; arch auto-derived for .pt)."""
    from argparse import Namespace
    from pathlib import Path

    onnx = (Path(__file__).resolve().parents[1]
            / "Weights" / "MGaze" / "resnet50_gaze.onnx")
    if not onnx.exists():
        pytest.skip("resnet50_gaze.onnx not present")
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA box: family resolves to .pt (heavier load)")

    import importlib
    mg = importlib.import_module(
        "mindsight.GazeTracking.Backends.MGaze.MGaze_Tracking")
    plugin = mg.MGazePlugin if hasattr(mg, "MGazePlugin") else None
    cls = plugin or next(
        v for v in vars(mg).values()
        if isinstance(v, type) and hasattr(v, "from_args")
        and v.__module__ == mg.__name__)
    engine = cls.from_args(Namespace(mgaze_model="resnet50", device="auto"))
    assert engine is not None
