"""
GUI/phenomena_panel.py — Phenomena tracking configuration panel.

Provides toggle controls for all built-in phenomena trackers and Joint
Attention accuracy parameters.  Embedded in the Gaze Tracker tab.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PhenomenaPanel(QWidget):
    """Configuration panel for phenomena tracker toggles and JA accuracy."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # ── Master toggle ─────────────────────────────────────────────────
        self._all_cb = QCheckBox("Enable all phenomena trackers")
        self._all_cb.setToolTip("Toggle all phenomena trackers on/off at once")
        self._all_cb.toggled.connect(self._toggle_all)
        lay.addWidget(self._all_cb)

        # ── Individual trackers ───────────────────────────────────────────
        # Each tracker: checkbox + optional parameter spinboxes

        # Mutual Gaze
        self._mutual_gaze = QCheckBox("Mutual Gaze")
        self._mutual_gaze.setToolTip("Track when two participants look at each other")
        lay.addWidget(self._mutual_gaze)

        # Social Referencing
        sr_row = QWidget()
        sr_lay = QHBoxLayout(sr_row)
        sr_lay.setContentsMargins(0, 0, 0, 0)
        self._social_ref = QCheckBox("Social Referencing")
        self._social_ref.setToolTip("Track when a participant looks at another person then at an object")
        sr_lay.addWidget(self._social_ref)
        sr_lay.addWidget(QLabel("window:"))
        self._social_ref_window = QSpinBox()
        self._social_ref_window.setRange(1, 300)
        self._social_ref_window.setValue(60)
        self._social_ref_window.setToolTip("Frame window for social referencing detection")
        self._social_ref_window.setFixedWidth(60)
        sr_lay.addWidget(self._social_ref_window)
        sr_lay.addStretch()
        lay.addWidget(sr_row)

        # Gaze Following
        gf_row = QWidget()
        gf_lay = QHBoxLayout(gf_row)
        gf_lay.setContentsMargins(0, 0, 0, 0)
        self._gaze_follow = QCheckBox("Gaze Following")
        self._gaze_follow.setToolTip("Track when one participant follows another's gaze direction")
        gf_lay.addWidget(self._gaze_follow)
        gf_lay.addWidget(QLabel("lag:"))
        self._gaze_follow_lag = QSpinBox()
        self._gaze_follow_lag.setRange(1, 120)
        self._gaze_follow_lag.setValue(30)
        self._gaze_follow_lag.setToolTip("Maximum frame lag for gaze-following detection")
        self._gaze_follow_lag.setFixedWidth(60)
        gf_lay.addWidget(self._gaze_follow_lag)
        gf_lay.addStretch()
        lay.addWidget(gf_row)

        # Gaze Aversion
        ga_grp = QGroupBox("Gaze Aversion")
        ga_grp.setCheckable(True)
        ga_grp.setChecked(False)
        ga_lay = QFormLayout(ga_grp)
        self._gaze_aversion = ga_grp  # the group itself is the toggle
        self._aversion_window = QSpinBox()
        self._aversion_window.setRange(1, 300)
        self._aversion_window.setValue(60)
        self._aversion_window.setToolTip("Frame window for aversion detection")
        ga_lay.addRow("Window:", self._aversion_window)
        self._aversion_conf = QDoubleSpinBox()
        self._aversion_conf.setRange(0.0, 1.0)
        self._aversion_conf.setSingleStep(0.05)
        self._aversion_conf.setValue(0.5)
        self._aversion_conf.setDecimals(2)
        self._aversion_conf.setToolTip("Confidence threshold for aversion detection")
        ga_lay.addRow("Confidence:", self._aversion_conf)
        lay.addWidget(ga_grp)

        # Scanpath
        sp_row = QWidget()
        sp_lay = QHBoxLayout(sp_row)
        sp_lay.setContentsMargins(0, 0, 0, 0)
        self._scanpath = QCheckBox("Scanpath")
        self._scanpath.setToolTip("Track sequential gaze transitions between objects")
        sp_lay.addWidget(self._scanpath)
        sp_lay.addWidget(QLabel("dwell:"))
        self._scanpath_dwell = QSpinBox()
        self._scanpath_dwell.setRange(1, 60)
        self._scanpath_dwell.setValue(8)
        self._scanpath_dwell.setToolTip("Frames of fixation to register a scanpath stop")
        self._scanpath_dwell.setFixedWidth(60)
        sp_lay.addWidget(self._scanpath_dwell)
        sp_lay.addStretch()
        lay.addWidget(sp_row)

        # Gaze Leadership
        gl_row = QWidget()
        gl_lay = QHBoxLayout(gl_row)
        gl_lay.setContentsMargins(0, 0, 0, 0)
        self._gaze_leader = QCheckBox("Gaze Leadership")
        self._gaze_leader.setToolTip("Track who initiates gaze shifts first")
        gl_lay.addWidget(self._gaze_leader)
        self._gaze_leader_tips = QCheckBox("+ Tips")
        self._gaze_leader_tips.setToolTip(
            "Also detect leadership via gaze-tip convergence (requires Gaze Tips)")
        gl_lay.addWidget(self._gaze_leader_tips)
        gl_lay.addWidget(QLabel("lag:"))
        self._gaze_leader_tip_lag = QSpinBox()
        self._gaze_leader_tip_lag.setRange(1, 120)
        self._gaze_leader_tip_lag.setValue(15)
        self._gaze_leader_tip_lag.setToolTip("Lookback frames for tip-arrival priority")
        self._gaze_leader_tip_lag.setFixedWidth(60)
        gl_lay.addWidget(self._gaze_leader_tip_lag)
        gl_lay.addStretch()
        lay.addWidget(gl_row)

        # Attention Span
        self._attn_span = QCheckBox("Attention Span")
        self._attn_span.setToolTip("Track average glance duration per participant per object")
        lay.addWidget(self._attn_span)

        # ── Joint Attention (toggleable tracker like the rest) ────────────
        ja_grp = QGroupBox("Joint Attention")
        ja_grp.setCheckable(True)
        ja_grp.setChecked(False)
        ja_grp.setToolTip("Track when multiple participants look at the same object simultaneously")
        self._ja_grp = ja_grp
        ja_lay = QFormLayout(ja_grp)

        self._ja_quorum = QDoubleSpinBox()
        self._ja_quorum.setRange(0.0, 1.0)
        self._ja_quorum.setSingleStep(0.05)
        self._ja_quorum.setValue(1.0)
        self._ja_quorum.setDecimals(2)
        self._ja_quorum.setToolTip("Fraction of faces required for joint attention (1.0 = all)")
        ja_lay.addRow("Quorum:", self._ja_quorum)

        self._ja_window = QSpinBox()
        self._ja_window.setRange(0, 300)
        self._ja_window.setValue(30)
        self._ja_window.setToolTip(
            "Temporal consistency window in frames (0 = no filtering, "
            "raw JA only)")
        ja_lay.addRow("Temporal window:", self._ja_window)

        self._ja_window_thresh = QDoubleSpinBox()
        self._ja_window_thresh.setRange(0.0, 1.0)
        self._ja_window_thresh.setSingleStep(0.05)
        self._ja_window_thresh.setValue(0.70)
        self._ja_window_thresh.setDecimals(2)
        self._ja_window_thresh.setToolTip("Fraction of window frames required for JA confirmation")
        ja_lay.addRow("Window threshold:", self._ja_window_thresh)

        lay.addWidget(ja_grp)

    def _toggle_all(self, checked: bool):
        """Toggle all individual phenomena checkboxes, including JA accuracy."""
        self._mutual_gaze.setChecked(checked)
        self._social_ref.setChecked(checked)
        self._gaze_follow.setChecked(checked)
        self._gaze_aversion.setChecked(checked)
        self._scanpath.setChecked(checked)
        self._gaze_leader.setChecked(checked)
        self._attn_span.setChecked(checked)
        self._ja_grp.setChecked(checked)

    def get_values(self) -> dict:
        """Return a dict of namespace attribute names to values."""
        return {
            "all_phenomena": self._all_cb.isChecked(),
            "joint_attention": self._ja_grp.isChecked(),
            "ja_window": self._ja_window.value(),
            "ja_window_thresh": self._ja_window_thresh.value(),
            "ja_quorum": self._ja_quorum.value(),
            "mutual_gaze": self._mutual_gaze.isChecked(),
            "social_ref": self._social_ref.isChecked(),
            "social_ref_window": self._social_ref_window.value(),
            "gaze_follow": self._gaze_follow.isChecked(),
            "gaze_follow_lag": self._gaze_follow_lag.value(),
            "gaze_aversion": self._gaze_aversion.isChecked(),
            "aversion_window": self._aversion_window.value(),
            "aversion_conf": self._aversion_conf.value(),
            "scanpath": self._scanpath.isChecked(),
            "scanpath_dwell": self._scanpath_dwell.value(),
            "gaze_leader": self._gaze_leader.isChecked(),
            "gaze_leader_tips": self._gaze_leader_tips.isChecked(),
            "gaze_leader_tip_lag": self._gaze_leader_tip_lag.value(),
            "attn_span": self._attn_span.isChecked(),
        }

    def apply_values(self, d: dict):
        """Set widget values from a dict (e.g. from a namespace)."""
        self._all_cb.setChecked(d.get("all_phenomena", False))
        self._ja_grp.setChecked(d.get("joint_attention", False))
        self._ja_window.setValue(d.get("ja_window", 30))
        self._ja_window_thresh.setValue(d.get("ja_window_thresh", 0.70))
        self._ja_quorum.setValue(d.get("ja_quorum", 1.0))
        self._mutual_gaze.setChecked(d.get("mutual_gaze", False))
        self._social_ref.setChecked(d.get("social_ref", False))
        self._social_ref_window.setValue(d.get("social_ref_window", 60))
        self._gaze_follow.setChecked(d.get("gaze_follow", False))
        self._gaze_follow_lag.setValue(d.get("gaze_follow_lag", 30))
        self._gaze_aversion.setChecked(d.get("gaze_aversion", False))
        self._aversion_window.setValue(d.get("aversion_window", 60))
        self._aversion_conf.setValue(d.get("aversion_conf", 0.5))
        self._scanpath.setChecked(d.get("scanpath", False))
        self._scanpath_dwell.setValue(d.get("scanpath_dwell", 8))
        self._gaze_leader.setChecked(d.get("gaze_leader", False))
        self._gaze_leader_tips.setChecked(d.get("gaze_leader_tips", False))
        self._gaze_leader_tip_lag.setValue(d.get("gaze_leader_tip_lag", 15))
        self._attn_span.setChecked(d.get("attn_span", False))
