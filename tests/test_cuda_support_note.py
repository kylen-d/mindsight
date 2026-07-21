"""W3Y item 4: NVIDIA-GPU-with-CPU-torch detection (cuda_support_note).

Headless reproduction of the Windows lab-machine decision path: PyPI's
Windows torch wheels are CPU-only, so torch.cuda.is_available() is False
on a dual-Quadro machine and the Models tab marked ONNX weights
"optimal".  The chooser (resolve_mgaze_family) is CORRECT given a
CUDA-less torch -- the fix is detecting the install problem and saying
so loudly.  All probes are injected; nothing here touches real
nvidia-smi or needs a GPU.
"""
from types import SimpleNamespace

import pytest

from mindsight.utils.device import cuda_support_note


def _fake_torch(*, cuda_available: bool, cuda_version):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        version=SimpleNamespace(cuda=cuda_version),
    )


def _smi_ok(cmd, **kw):
    return SimpleNamespace(returncode=0,
                           stdout="GPU 0: Quadro P4000\nGPU 1: Quadro P4000\n")


def _smi_fail(cmd, **kw):
    return SimpleNamespace(returncode=1, stdout="")


@pytest.fixture(autouse=True)
def _pretend_not_macos(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Windows")


def test_cpu_wheel_with_nvidia_gpu_is_flagged():
    note = cuda_support_note(
        _torch=_fake_torch(cuda_available=False, cuda_version=None),
        _which=lambda name: "C:/Windows/System32/nvidia-smi.exe",
        _run=_smi_ok)
    assert note is not None
    assert "no CUDA support" in note or "CPU-only" in note
    assert "--index-url" in note          # remedy included


def test_working_cuda_says_nothing():
    note = cuda_support_note(
        _torch=_fake_torch(cuda_available=True, cuda_version="12.6"),
        _which=lambda name: "x", _run=_smi_ok)
    assert note is None


def test_cuda_build_without_gpu_is_a_driver_story_not_flagged():
    # CUDA wheel installed but no usable GPU: not the install bug.
    note = cuda_support_note(
        _torch=_fake_torch(cuda_available=False, cuda_version="12.6"),
        _which=lambda name: "x", _run=_smi_ok)
    assert note is None


def test_genuinely_gpuless_machine_not_flagged():
    note = cuda_support_note(
        _torch=_fake_torch(cuda_available=False, cuda_version=None),
        _which=lambda name: None, _run=_smi_ok)
    assert note is None


def test_smi_failure_not_flagged():
    note = cuda_support_note(
        _torch=_fake_torch(cuda_available=False, cuda_version=None),
        _which=lambda name: "x", _run=_smi_fail)
    assert note is None


def test_macos_short_circuits(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    note = cuda_support_note(
        _torch=_fake_torch(cuda_available=False, cuda_version=None),
        _which=lambda name: "x", _run=_smi_ok)
    assert note is None


def test_chooser_is_correct_given_cuda_probe():
    """The 'optimal' logic itself: .pt on CUDA, ONNX elsewhere -- the lab
    bug was the CPU-only wheel feeding it False, not this decision."""
    from mindsight.weights import resolve_mgaze_family
    assert resolve_mgaze_family("resnet50", device="cuda") == "resnet50.pt"
    assert resolve_mgaze_family("resnet50", device="cpu") == "resnet50_gaze.onnx"
    assert resolve_mgaze_family("resnet50.pt", device="cpu") == "resnet50.pt"


def test_preflight_surfaces_the_note(monkeypatch):
    """The device preflight check turns the note into a WARN line."""
    import mindsight.utils.device as device_mod
    from mindsight.project.preflight import _check_device
    monkeypatch.setattr(device_mod, "cuda_support_note",
                        lambda **kw: "torch has no CUDA support installed")
    res = _check_device(SimpleNamespace(device="auto"),
                        lambda req: (True, "cpu"))
    assert res.severity == "warn"
    assert "CUDA" in res.message
