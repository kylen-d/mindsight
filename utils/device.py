"""
utils/device.py — Shared device detection for all backends.
"""
import torch


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
