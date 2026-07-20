"""
mindsight.validation — In-app validation & tuning suite (v1.1 W4B, Layout B).

Ground-truth annotation sets plus (in later phases) the in-app runner
and report machinery.  The on-disk set format is a strict SUPERSET of
the eval-harness label format (``eval_data/{stem}_labels.json``), so
``scripts/eval_gaze.py score`` works on a validation set unchanged.
"""
from .runner import (
    allocate_run_dir,
    embed_validation_summary,
    latest_score,
    list_run_dirs,
    prepare_clip_namespace,
    prepare_validation_namespace,
    run_history,
    score_and_persist,
    settings_diff,
    validation_summary_block,
)
from .scoring import score_run
from .sweep import (
    COMBO_CAP,
    CURATED_KNOBS,
    allocate_sweep_path,
    estimate_seconds,
    expand_combos,
    latest_sweep,
    new_sweep_manifest,
    pick_winner,
    prepare_sweep_clip_namespace,
    prepare_sweep_namespace,
    save_sweep,
)
from .store import (
    LABEL_STATES,
    ValidationClip,
    ValidationSet,
    ValidationSetError,
    ValidationStore,
    clips_from_project,
    validation_root,
)

__all__ = [
    "LABEL_STATES", "ValidationClip", "ValidationSet", "ValidationSetError",
    "ValidationStore", "validation_root", "clips_from_project",
    "score_run", "allocate_run_dir", "list_run_dirs", "latest_score",
    "prepare_validation_namespace", "prepare_clip_namespace",
    "prepare_sweep_clip_namespace", "score_and_persist",
    "run_history", "settings_diff",
    "validation_summary_block", "embed_validation_summary",
    "COMBO_CAP", "CURATED_KNOBS", "expand_combos", "estimate_seconds",
    "prepare_sweep_namespace", "new_sweep_manifest", "allocate_sweep_path",
    "save_sweep", "latest_sweep", "pick_winner",
]
