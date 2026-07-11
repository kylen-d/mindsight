"""Tests for outputs/provenance.py -- run-identity, weight cache, manifest."""

import json

from mindsight.cli import _args
from mindsight.config import PipelineConfig
from mindsight.outputs import provenance


def _ns(*extra):
    return _args(["--source", "test_data/trimmed.mp4", *extra])


def _rid(ns, weights=None):
    cfg = PipelineConfig.from_namespace(ns)
    return provenance.run_identity(ns, config=cfg, weights=weights or {})


# ── run_identity (D6) ─────────────────────────────────────────────────────────

class TestRunIdentity:

    def test_stable_same_inputs(self):
        assert _rid(_ns()) == _rid(_ns())

    def test_output_section_change_same_hash(self, tmp_path):
        # Changing output paths must NOT change the run identity (D6).
        base = _rid(_ns())
        changed = _rid(_ns("--summary", str(tmp_path / "x_summary.csv"),
                           "--save", str(tmp_path / "o.mp4"),
                           "--log", str(tmp_path / "e.csv")))
        assert base == changed

    def test_conf_change_differs(self):
        assert _rid(_ns("--conf", "0.35")) != _rid(_ns("--conf", "0.30"))

    def test_device_change_differs(self):
        # Q3: --device is part of the identity (backend numerics differ).
        assert _rid(_ns("--device", "auto")) != _rid(_ns("--device", "cpu"))

    def test_plugin_flag_change_differs(self):
        assert _rid(_ns()) != _rid(_ns("--gaze-follow"))

    def test_weight_sha_change_differs(self):
        cfg = PipelineConfig.from_namespace(_ns())
        ns = _ns()
        a = provenance.run_identity(ns, config=cfg, weights={
            "model": {"sha256": "aaa"}})
        b = provenance.run_identity(ns, config=cfg, weights={
            "model": {"sha256": "bbb"}})
        assert a != b


# ── weight enumeration + sha cache (D7) ───────────────────────────────────────

class TestWeights:

    def test_cache_hashes_once(self):
        ns = _ns("--model", "Weights/YOLO/yolov8n.pt")
        provenance._SHA_CACHE.clear()
        w1 = provenance.collect_weights(ns)
        count_after_first = provenance._sha_compute_count
        w2 = provenance.collect_weights(ns)
        count_after_second = provenance._sha_compute_count
        assert w1["model"]["sha256"] == w2["model"]["sha256"]
        assert len(w1["model"]["sha256"]) == 64
        # Second identical call is served entirely from cache.
        assert count_after_second == count_after_first

    def test_missing_weight_recorded(self):
        ns = _ns("--rf-gazelle-model", "definitely_not_here_xyz.pt")
        w = provenance.collect_weights(ns)
        assert w["rf_gazelle_model"]["sha256"] == "missing"

    def test_vp_model_skipped_without_vp_file(self):
        ns = _ns()  # no --vp-file
        w = provenance.collect_weights(ns)
        assert "vp_model" not in w

    def test_mgaze_family_resolved_per_device(self):
        # Extensionless family names must resolve to the device build the run
        # actually loads (eyes-on A4: preflight flagged bare "resnet50" as a
        # missing file while the run itself worked).
        ns = _ns("--mgaze-model", "resnet50", "--device", "cpu")
        w = provenance.collect_weights(ns)
        assert w["mgaze_model"]["requested"] == "resnet50"
        assert w["mgaze_model"]["resolved"].endswith("resnet50_gaze.onnx")

    def test_mgaze_explicit_extension_unchanged(self):
        ns = _ns("--mgaze-model", "resnet50.pt", "--device", "cpu")
        w = provenance.collect_weights(ns)
        assert w["mgaze_model"]["resolved"].endswith("resnet50.pt")


# ── environment ───────────────────────────────────────────────────────────────

def test_environment_keys_present():
    env = provenance.collect_environment()
    assert isinstance(env["mindsight"], str)
    assert "python" in env and "platform" in env
    deps = env["dependencies"]
    for mod in ["torch", "ultralytics", "onnxruntime", "cv2", "numpy",
                "mediapipe"]:
        assert mod in deps


# ── manifest write (atomic) ───────────────────────────────────────────────────

class TestManifest:

    def _write(self, tmp_path, name, started, finished, **over):
        ns = over.pop("ns", None) or _ns()
        status = over.pop("status", "completed")
        error = over.pop("error", None)
        cfg = PipelineConfig.from_namespace(ns)
        p = tmp_path / name
        provenance.write_run_manifest(
            str(p), ns=ns, config=cfg, source="test_data/trimmed.mp4",
            output_paths={"summary": str(tmp_path / "s.csv")},
            started=started, finished=finished, status=status, error=error)
        return p

    def test_atomic_write_valid_json_no_tmp(self, tmp_path):
        p = self._write(tmp_path / "sub", "m.json", "T0", "T1")
        data = json.loads(p.read_text())
        assert data["schema_version"] == 1
        assert data["status"] == "completed"
        assert data["source"]["exists"] is True
        assert len(data["source"]["sha256"]) == 64
        assert "environment" in data and "weights" in data
        assert len(data["run_identity"]) == 64
        assert not list(p.parent.glob("*.tmp.*"))

    def test_manifest_stable_except_timestamps(self, tmp_path):
        p1 = self._write(tmp_path, "m1.json", "A", "B")
        p2 = self._write(tmp_path, "m2.json", "C", "D")
        d1, d2 = json.loads(p1.read_text()), json.loads(p2.read_text())
        for d in (d1, d2):
            d.pop("started")
            d.pop("finished")
        assert d1 == d2

    def test_error_status_recorded(self, tmp_path):
        p = self._write(tmp_path, "e.json", "A", "B",
                        status="error", error="boom")
        data = json.loads(p.read_text())
        assert data["status"] == "error"
        assert data["error"] == "boom"


# ── manifest location / no-output rule (Q4 / D8) ──────────────────────────────

class TestManifestLocation:

    def test_no_manifest_when_no_file_output(self):
        ns = _ns()
        outputs = provenance.resolve_single_source_outputs(
            ns, "test_data/trimmed.mp4")
        assert provenance.manifest_path_for(outputs) is None

    def test_anchor_prefers_summary(self, tmp_path):
        ns = _ns("--summary", str(tmp_path / "v_summary.csv"),
                 "--log", str(tmp_path / "v_events.csv"))
        outputs = provenance.resolve_single_source_outputs(
            ns, "test_data/trimmed.mp4")
        assert provenance.manifest_path_for(outputs) == str(
            tmp_path / "v_summary_manifest.json")

    def test_anchor_falls_back_to_log(self, tmp_path):
        ns = _ns("--log", str(tmp_path / "v_events.csv"))
        outputs = provenance.resolve_single_source_outputs(
            ns, "test_data/trimmed.mp4")
        assert provenance.manifest_path_for(outputs) == str(
            tmp_path / "v_events_manifest.json")


# ── static plugin-dest list welded to the live registries (D6) ────────────────

def test_plugin_dests_match_registries():
    from tests.test_config_schema import get_plugin_dests
    assert set(provenance._PLUGIN_DESTS) == get_plugin_dests()
