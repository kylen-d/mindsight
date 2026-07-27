"""apptainer/bin/mindsight-seed-home -- symlink-seeding contract.

The script runs inside the Apptainer image, but it is plain bash: these tests
exercise it directly against a fake baked-home layout via MINDSIGHT_BAKED_HOME.
"""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "apptainer" / "bin" / "mindsight-seed-home"


@pytest.fixture()
def baked(tmp_path):
    b = tmp_path / "baked"
    (b / "Weights" / "YOLO").mkdir(parents=True)
    (b / "Weights" / "YOLO" / "yolo11n.pt").write_bytes(b"baked-yolo")
    (b / "Weights" / "MGaze").mkdir()
    (b / "Weights" / "MGaze" / "resnet50_gaze.onnx").write_bytes(b"baked-mgaze")
    (b / "weights_manifest.json").write_text("{}")
    return b


def run(baked, home, *args):
    return subprocess.run(
        [str(SCRIPT), str(home), *args], capture_output=True, text=True,
        env={**os.environ, "MINDSIGHT_BAKED_HOME": str(baked)})


def test_seeds_symlinks_manifest_and_outputs(baked, tmp_path):
    home = tmp_path / "home"
    r = run(baked, home)
    assert r.returncode == 0, r.stderr
    assert (home / "Outputs").is_dir()
    link = home / "Weights" / "YOLO" / "yolo11n.pt"
    assert link.is_symlink() and link.read_bytes() == b"baked-yolo"
    assert (home / "Weights" / "MGaze" / "resnet50_gaze.onnx").is_symlink()
    assert (home / "weights_manifest.json").is_symlink()


def test_real_files_never_replaced(baked, tmp_path):
    home = tmp_path / "home"
    (home / "Weights" / "YOLO").mkdir(parents=True)
    (home / "Weights" / "YOLO" / "yolo11n.pt").write_bytes(b"user-downloaded")
    r = run(baked, home)
    assert r.returncode == 0, r.stderr
    f = home / "Weights" / "YOLO" / "yolo11n.pt"
    assert not f.is_symlink() and f.read_bytes() == b"user-downloaded"


def test_shared_weights_merged(baked, tmp_path):
    shared = tmp_path / "shared"
    (shared / "Gazelle").mkdir(parents=True)
    (shared / "Gazelle" / "big.onnx").write_bytes(b"shared")
    home = tmp_path / "home"
    r = run(baked, home, "--shared-weights", str(shared))
    assert r.returncode == 0, r.stderr
    assert (home / "Weights" / "Gazelle" / "big.onnx").is_symlink()
    assert (home / "Weights" / "YOLO" / "yolo11n.pt").is_symlink()


def test_reseed_is_idempotent(baked, tmp_path):
    home = tmp_path / "home"
    assert run(baked, home).returncode == 0
    r2 = run(baked, home)
    assert r2.returncode == 0, r2.stderr


def test_usage_errors_exit_2(baked, tmp_path):
    assert run(baked, tmp_path / "h", "--bogus").returncode == 2
    bare = subprocess.run([str(SCRIPT)], capture_output=True, text=True)
    assert bare.returncode == 2
