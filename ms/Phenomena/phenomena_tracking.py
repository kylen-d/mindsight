"""
Phenomena/phenomena_tracking.py — Backward-compatible re-export shim.

All tracker classes have been moved to individual files under
``Phenomena/Default/``.  Helper functions have been moved to
``Phenomena/helpers.py``.  This file re-exports everything so existing
imports continue to work.

Phenomena CLI flags are now generated from the pydantic schema via
``ms/cli_flags.py`` (SP1.3); the old ``add_arguments`` registration was removed.
"""

# ── Re-exports from Default pack ─────────────────────────────────────────────
from ms.Phenomena.Default import (  # noqa: F401
    AttentionSpanTracker,
    GazeAversionTracker,
    GazeFollowingTracker,
    GazeLeadershipTracker,
    JointAttentionTracker,
    MutualGazeTracker,
    ScanpathTracker,
    SocialReferenceTracker,
)

# ── Re-exports from helpers ──────────────────────────────────────────────────
from ms.Phenomena.helpers import (  # noqa: F401
    gaze_convergence,
    joint_attention,
)
