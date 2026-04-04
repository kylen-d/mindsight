"""
Phenomena/phenomena_config.py — Opaque configuration for phenomena trackers.

The PhenomenaConfig dataclass holds all flags and thresholds that control
which phenomena trackers are enabled and how they behave.  MindSight.py
passes it as a single opaque object — it never needs to know which
individual phenomena exist.
"""
from dataclasses import dataclass


@dataclass
class PhenomenaConfig:
    """All phenomena-related configuration in one object."""

    # Joint attention
    joint_attention: bool = False
    ja_window: int = 0
    ja_window_thresh: float = 0.70
    ja_quorum: float = 1.0

    # Gaze phenomena toggles + parameters
    mutual_gaze: bool = False
    social_ref: bool = False
    social_ref_window: int = 60
    gaze_follow: bool = False
    gaze_follow_lag: int = 30
    gaze_aversion: bool = False
    aversion_window: int = 60
    aversion_conf: float = 0.5
    scanpath: bool = False
    scanpath_dwell: int = 8
    gaze_leader: bool = False
    gaze_leader_tips: bool = False
    gaze_leader_tip_lag: int = 15
    attn_span: bool = False

    @classmethod
    def from_namespace(cls, ns) -> "PhenomenaConfig":
        """Construct from an ``argparse.Namespace``, honouring ``--all-phenomena``."""
        all_on = getattr(ns, "all_phenomena", False)
        return cls(
            joint_attention=ns.joint_attention or all_on,
            ja_window=ns.ja_window,
            ja_window_thresh=ns.ja_window_thresh,
            ja_quorum=ns.ja_quorum,
            mutual_gaze=ns.mutual_gaze or all_on,
            social_ref=ns.social_ref or all_on,
            social_ref_window=ns.social_ref_window,
            gaze_follow=ns.gaze_follow or all_on,
            gaze_follow_lag=ns.gaze_follow_lag,
            gaze_aversion=ns.gaze_aversion or all_on,
            aversion_window=ns.aversion_window,
            aversion_conf=ns.aversion_conf,
            scanpath=ns.scanpath or all_on,
            scanpath_dwell=ns.scanpath_dwell,
            gaze_leader=ns.gaze_leader or all_on,
            gaze_leader_tips=getattr(ns, "gaze_leader_tips", False),
            gaze_leader_tip_lag=getattr(ns, "gaze_leader_tip_lag", 15),
            attn_span=ns.attn_span or all_on,
        )
