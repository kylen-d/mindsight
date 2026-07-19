"""
mindsight.validation — In-app validation & tuning suite (v1.1 W4B, Layout B).

Ground-truth annotation sets plus (in later phases) the in-app runner
and report machinery.  The on-disk set format is a strict SUPERSET of
the eval-harness label format (``eval_data/{stem}_labels.json``), so
``scripts/eval_gaze.py score`` works on a validation set unchanged.
"""
from .runner import (
    allocate_run_dir,
    latest_score,
    list_run_dirs,
    prepare_validation_namespace,
    run_history,
    score_and_persist,
    settings_diff,
)
from .scoring import score_run
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
    "score_run", "allocate_run_dir", "list_run_dirs", "latest_score",
    "prepare_validation_namespace", "score_and_persist",
    "run_history", "settings_diff",
]
