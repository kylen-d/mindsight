"""
Phenomena — Psychological gaze-phenomena tracking pipeline stage.

Houses the built-in phenomena trackers (mutual gaze, social referencing,
gaze following, gaze aversion, scanpath analysis, gaze leadership,
attention span, and joint attention) as well as configuration and helpers
shared across trackers.
"""

from .phenomena_pipeline import (
    init_phenomena_trackers,
    update_phenomena_step,
    post_run_summary,
)
from .helpers import joint_attention, gaze_convergence
from .phenomena_config import PhenomenaConfig

__all__ = [
    "init_phenomena_trackers",
    "update_phenomena_step",
    "post_run_summary",
    "joint_attention",
    "gaze_convergence",
    "PhenomenaConfig",
]
