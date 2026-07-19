"""
mindsight.validation — In-app validation & tuning suite (v1.1 W4B, Layout B).

Ground-truth annotation sets plus (in later phases) the in-app runner
and report machinery.  The on-disk set format is a strict SUPERSET of
the eval-harness label format (``eval_data/{stem}_labels.json``), so
``scripts/eval_gaze.py score`` works on a validation set unchanged.
"""
from .store import (
    LABEL_STATES,
    ValidationSet,
    ValidationSetError,
    ValidationStore,
    validation_root,
)

__all__ = [
    "LABEL_STATES", "ValidationSet", "ValidationSetError",
    "ValidationStore", "validation_root",
]
