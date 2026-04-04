"""
Phenomena — Psychological gaze-phenomena tracking pipeline stage.

Houses the built-in phenomena trackers (mutual gaze, social referencing,
gaze following, gaze aversion, scanpath analysis, gaze leadership,
attention span, and joint attention) as well as configuration and helpers
shared across trackers.
"""

from .helpers import gaze_convergence, joint_attention
from .phenomena_config import PhenomenaConfig
from .phenomena_pipeline import (
    init_phenomena_trackers,
    post_run_summary,
    update_phenomena_step,
)

__all__ = [
    "init_phenomena_trackers",
    "update_phenomena_step",
    "post_run_summary",
    "joint_attention",
    "gaze_convergence",
    "PhenomenaConfig",
]
