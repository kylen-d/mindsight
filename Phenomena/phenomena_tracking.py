"""
Phenomena/phenomena_tracking.py — Backward-compatible re-export shim.

All tracker classes have been moved to individual files under
``Phenomena/Default/``.  Helper functions have been moved to
``Phenomena/helpers.py``.  This file re-exports everything so existing
imports continue to work.

CLI argument registration (``add_arguments``) also lives here since
MindSight.py imports it from this module.
"""

# ── Re-exports from Default pack ─────────────────────────────────────────────
from Phenomena.Default import (  # noqa: F401
    AttentionSpanTracker,
    GazeAversionTracker,
    GazeFollowingTracker,
    GazeLeadershipTracker,
    JointAttentionTemporalTracker,
    MutualGazeTracker,
    ScanpathTracker,
    SocialReferenceTracker,
)

# ── Re-exports from helpers ──────────────────────────────────────────────────
from Phenomena.helpers import (  # noqa: F401
    gaze_convergence,
    joint_attention,
)

# ══════════════════════════════════════════════════════════════════════════════
# CLI argument registration
# ══════════════════════════════════════════════════════════════════════════════

def add_arguments(parser) -> None:
    """Register all phenomena-tracking CLI flags onto *parser*."""
    # Joint attention
    parser.add_argument("--joint-attention", action="store_true",
                        help="Enable joint-attention tracking.")
    parser.add_argument("--ja-window", type=int, default=0, metavar="N",
                        help="Temporal consistency window (frames). 0 = disabled (default).")
    parser.add_argument("--ja-window-thresh", type=float, default=0.70, metavar="F")
    parser.add_argument("--ja-quorum", type=float, default=1.0, metavar="F",
                        help="Fraction of faces required for joint attention (default 1.0).")
    # Gaze phenomena flags
    parser.add_argument("--mutual-gaze",   action="store_true")
    parser.add_argument("--social-ref",    action="store_true")
    parser.add_argument("--social-ref-window", type=int, default=60, metavar="N")
    parser.add_argument("--gaze-follow",   action="store_true")
    parser.add_argument("--gaze-follow-lag", type=int, default=30, metavar="N")
    parser.add_argument("--gaze-aversion", action="store_true")
    parser.add_argument("--aversion-window", type=int, default=60, metavar="N")
    parser.add_argument("--aversion-conf", type=float, default=0.5, metavar="F")
    parser.add_argument("--scanpath",      action="store_true")
    parser.add_argument("--scanpath-dwell", type=int, default=8, metavar="N")
    parser.add_argument("--gaze-leader",   action="store_true")
    parser.add_argument("--gaze-leader-tips", action="store_true",
                        help="Also detect leadership via gaze-tip convergence "
                             "(requires --gaze-tips).")
    parser.add_argument("--gaze-leader-tip-lag", type=int, default=15, metavar="N",
                        help="Lookback frames for tip-arrival priority (default: 15).")
    parser.add_argument("--attn-span",     action="store_true",
                        help="Track per-participant per-object average attention span "
                             "(mean completed-glance duration). Most salient object shown in HUD.")
    parser.add_argument("--all-phenomena", action="store_true",
                        help="Enable all gaze-phenomena trackers at once.")
