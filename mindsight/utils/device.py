"""
utils/device.py — Shared device detection for all backends.
"""
import torch


def cuda_support_note(*, _torch=None, _which=None, _run=None) -> str | None:
    """A loud note when an NVIDIA GPU is visible but torch cannot use it.

    The common trap (W3Y item 4, seen on the dual-Quadro Windows lab
    machine): PyPI's Windows ``torch`` wheels are CPU-only, so
    ``torch.cuda.is_available()`` is False and every "optimal for this
    device" decision silently degrades to the CPU/ONNX story.  This probe
    distinguishes that INSTALL problem (CPU-only build + ``nvidia-smi``
    reports a GPU) from a genuinely GPU-less machine, and returns a
    remedy string for the Models tab and project preflight -- or None
    when there is nothing to say.  Never raises; the underscore kwargs
    are test seams.
    """
    import platform
    import shutil
    import subprocess

    t = _torch if _torch is not None else torch
    try:
        if platform.system() == "Darwin":
            return None                      # no NVIDIA path on macOS
        if t.cuda.is_available():
            return None                      # CUDA works; nothing to say
        if getattr(t.version, "cuda", None):
            # CUDA build without a usable GPU: a driver/hardware story,
            # not the CPU-wheel install bug this note is for.
            return None
        which = _which if _which is not None else shutil.which
        if which("nvidia-smi") is None:
            return None                      # no NVIDIA driver installed
        run = _run if _run is not None else subprocess.run
        r = run(["nvidia-smi", "-L"], capture_output=True, text=True,
                timeout=5)
        if r.returncode != 0 or not (r.stdout or "").strip():
            return None
    except Exception:
        return None
    return ("An NVIDIA GPU is present but the installed torch build has no "
            "CUDA support (CPU-only wheel) -- GPU inference and the CUDA "
            "'optimal' weights are unavailable. Reinstall torch from the "
            "CUDA index, e.g.: pip install torch==2.10.0 "
            "torchvision==0.25.0 --index-url "
            "https://download.pytorch.org/whl/cu126")


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string to a ``torch.device``.

    When *device* is ``"auto"`` the preference order is CUDA > MPS > CPU.
    Any other string (``"cpu"``, ``"cuda"``, ``"mps"``) is passed through
    directly to ``torch.device``.
    """
    if device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(device)
