"""Gaze ray geometry, adaptive snap, fixation lock-on, and hit detection."""

from __future__ import annotations

from argparse import Namespace

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class RaySection(QWidget):
    """Ray geometry + adaptive snap + fixation lock-on + hit detection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._build_ray_geometry(lay)
        self._build_adaptive_snap(lay)
        self._build_fixation_lockon(lay)
        self._build_hit_detection(lay)

    # -- Ray Geometry ---------------------------------------------------------

    def _build_ray_geometry(self, lay):
        g = QGroupBox("Gaze Ray Geometry")
        f = QFormLayout(g)

        self._ray_length = QDoubleSpinBox()
        self._ray_length.setRange(0.2, 5.0)
        self._ray_length.setSingleStep(0.1)
        self._ray_length.setValue(1.0)
        self._ray_length.setDecimals(1)
        f.addRow("Ray length:", self._ray_length)

        self._cb_conf_ray = QCheckBox("Scale ray length by confidence")
        f.addRow(self._cb_conf_ray)

        self._gaze_cone = QDoubleSpinBox()
        self._gaze_cone.setRange(0.0, 45.0)
        self._gaze_cone.setSingleStep(1.0)
        self._gaze_cone.setValue(0.0)
        self._gaze_cone.setDecimals(1)
        self._gaze_cone.setToolTip(
            "Vision cone angle in degrees (0 = standard ray)")
        f.addRow("Gaze cone:", self._gaze_cone)

        self._forward_gaze_thresh = QDoubleSpinBox()
        self._forward_gaze_thresh.setRange(0.0, 30.0)
        self._forward_gaze_thresh.setSingleStep(0.5)
        self._forward_gaze_thresh.setValue(5.0)
        self._forward_gaze_thresh.setDecimals(1)
        self._forward_gaze_thresh.setToolTip(
            "Pitch/yaw angles below this threshold (degrees) are treated as\n"
            "looking forward (at camera). Set to 0 to disable.")
        f.addRow("Forward threshold (\u00b0):", self._forward_gaze_thresh)

        # Gaze tips (nested checkable sub-group)
        self._gaze_tips_group = QGroupBox("Gaze tips (virtual objects)")
        self._gaze_tips_group.setCheckable(True)
        self._gaze_tips_group.setChecked(False)
        ft = QFormLayout(self._gaze_tips_group)
        self._tip_radius = QSpinBox()
        self._tip_radius.setRange(20, 300)
        self._tip_radius.setValue(80)
        ft.addRow("Tip radius:", self._tip_radius)
        f.addRow(self._gaze_tips_group)

        lay.addWidget(g)

    # -- Adaptive Snap --------------------------------------------------------

    def _build_adaptive_snap(self, lay):
        self._adaptive_snap_group = QGroupBox("Adaptive Snap")
        self._adaptive_snap_group.setCheckable(True)
        self._adaptive_snap_group.setChecked(False)
        f = QFormLayout(self._adaptive_snap_group)

        self._adaptive_mode_combo = QComboBox()
        self._adaptive_mode_combo.addItems(["Extend", "Snap"])
        self._adaptive_mode_combo.setToolTip(
            "Extend: ray freely extends toward the nearest object.\n"
            "Snap: ray locks to the object centre.")
        f.addRow("Mode:", self._adaptive_mode_combo)

        self._snap_dist = QSpinBox()
        self._snap_dist.setRange(20, 500)
        self._snap_dist.setValue(150)
        f.addRow("Snap dist (px):", self._snap_dist)

        self._snap_bbox_scale = QDoubleSpinBox()
        self._snap_bbox_scale.setRange(0.0, 2.0)
        self._snap_bbox_scale.setSingleStep(0.1)
        self._snap_bbox_scale.setValue(0.0)
        self._snap_bbox_scale.setDecimals(2)
        self._snap_bbox_scale.setToolTip(
            "Fraction of bbox half-diagonal added to effective snap radius")
        f.addRow("Bbox scale:", self._snap_bbox_scale)

        # Scoring Weights sub-heading
        lbl_scoring = QLabel("Scoring Weights")
        lbl_scoring.setStyleSheet(
            "color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_scoring)

        self._snap_w_dist = QDoubleSpinBox()
        self._snap_w_dist.setRange(0.0, 3.0)
        self._snap_w_dist.setSingleStep(0.1)
        self._snap_w_dist.setValue(1.0)
        self._snap_w_dist.setDecimals(2)
        self._snap_w_dist.setToolTip(
            "Scoring weight for normalized distance penalty")
        f.addRow("W distance:", self._snap_w_dist)

        self._snap_w_size = QDoubleSpinBox()
        self._snap_w_size.setRange(0.0, 3.0)
        self._snap_w_size.setSingleStep(0.1)
        self._snap_w_size.setValue(0.0)
        self._snap_w_size.setDecimals(2)
        self._snap_w_size.setToolTip("Scoring weight for angular size reward")
        f.addRow("W size:", self._snap_w_size)

        self._snap_w_intersect = QDoubleSpinBox()
        self._snap_w_intersect.setRange(0.0, 3.0)
        self._snap_w_intersect.setSingleStep(0.1)
        self._snap_w_intersect.setValue(0.5)
        self._snap_w_intersect.setDecimals(2)
        self._snap_w_intersect.setToolTip(
            "Scoring bonus for ray-bbox intersection")
        f.addRow("W intersect:", self._snap_w_intersect)

        # Stabilization sub-heading
        lbl_stab = QLabel("Stabilization")
        lbl_stab.setStyleSheet(
            "color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_stab)

        self._snap_switch = QSpinBox()
        self._snap_switch.setRange(1, 30)
        self._snap_switch.setValue(8)
        self._snap_switch.setToolTip("Frames before snap switches target")
        f.addRow("Snap switch frames:", self._snap_switch)

        lay.addWidget(self._adaptive_snap_group)

    # -- Fixation Lock-On -----------------------------------------------------

    def _build_fixation_lockon(self, lay):
        self._fixation_group = QGroupBox("Fixation Lock-On")
        self._fixation_group.setCheckable(True)
        self._fixation_group.setChecked(False)
        f = QFormLayout(self._fixation_group)

        self._dwell_frames = QSpinBox()
        self._dwell_frames.setRange(1, 120)
        self._dwell_frames.setValue(15)
        self._dwell_frames.setToolTip(
            "Consecutive frames the gaze ray must stay near an object "
            "to trigger lock-on")
        f.addRow("Dwell frames:", self._dwell_frames)

        self._lock_dist = QSpinBox()
        self._lock_dist.setRange(20, 400)
        self._lock_dist.setValue(100)
        self._lock_dist.setToolTip(
            "Distance (px) from gaze ray to object centre for dwell counting")
        f.addRow("Lock dist (px):", self._lock_dist)

        lay.addWidget(self._fixation_group)

    # -- Hit Detection --------------------------------------------------------

    def _build_hit_detection(self, lay):
        g = QGroupBox("Hit Detection")
        f = QFormLayout(g)

        self._hit_conf_gate = QDoubleSpinBox()
        self._hit_conf_gate.setRange(0.0, 1.0)
        self._hit_conf_gate.setSingleStep(0.05)
        self._hit_conf_gate.setValue(0.0)
        self._hit_conf_gate.setDecimals(2)
        self._hit_conf_gate.setToolTip(
            "Minimum per-face gaze confidence for ray-object hit detection.\n"
            "0 = disabled (all faces participate).")
        f.addRow("Hit conf gate:", self._hit_conf_gate)

        self._detect_extend = QSpinBox()
        self._detect_extend.setRange(0, 20000)
        self._detect_extend.setSingleStep(50)
        self._detect_extend.setValue(0)
        self._detect_extend.setToolTip(
            "Extend gaze-object detection N pixels past the visual\n"
            "ray/cone endpoint. 0 = detection matches visual exactly.")
        f.addRow("Extend detection (px):", self._detect_extend)

        self._detect_extend_scope = QComboBox()
        self._detect_extend_scope.addItems(["Objects", "Phenomena", "Both"])
        self._detect_extend_scope.setToolTip(
            "Scope for detection extension:\n"
            "Objects: extends ray-object hit detection only.\n"
            "Phenomena: extends phenomena tracking (mutual gaze, social ref) "
            "only.\n"
            "Both: extends both.")
        f.addRow("Extend scope:", self._detect_extend_scope)

        lay.addWidget(g)

    # -- Namespace interface --------------------------------------------------

    def namespace_values(self) -> dict:
        return dict(
            ray_length=self._ray_length.value(),
            conf_ray=self._cb_conf_ray.isChecked(),
            gaze_cone=self._gaze_cone.value(),
            forward_gaze_threshold=self._forward_gaze_thresh.value(),
            gaze_tips=self._gaze_tips_group.isChecked(),
            tip_radius=self._tip_radius.value(),
            adaptive_ray=(self._adaptive_mode_combo.currentText().lower()
                          if self._adaptive_snap_group.isChecked()
                          else "off"),
            snap_dist=float(self._snap_dist.value()),
            snap_bbox_scale=self._snap_bbox_scale.value(),
            snap_w_dist=self._snap_w_dist.value(),
            snap_w_size=self._snap_w_size.value(),
            snap_w_intersect=self._snap_w_intersect.value(),
            snap_switch_frames=self._snap_switch.value(),
            gaze_lock=self._fixation_group.isChecked(),
            dwell_frames=self._dwell_frames.value(),
            lock_dist=self._lock_dist.value(),
            hit_conf_gate=self._hit_conf_gate.value(),
            detect_extend=float(self._detect_extend.value()),
            detect_extend_scope=(
                self._detect_extend_scope.currentText().lower()),
        )

    def apply_namespace(self, ns: Namespace):
        self._ray_length.setValue(getattr(ns, 'ray_length', 1.0))
        self._cb_conf_ray.setChecked(bool(getattr(ns, 'conf_ray', False)))
        self._gaze_cone.setValue(getattr(ns, 'gaze_cone', 0.0))
        self._forward_gaze_thresh.setValue(
            getattr(ns, 'forward_gaze_threshold', 5.0))
        self._gaze_tips_group.setChecked(
            bool(getattr(ns, 'gaze_tips', False)))
        self._tip_radius.setValue(getattr(ns, 'tip_radius', 80))

        ar = getattr(ns, 'adaptive_ray', 'off')
        if isinstance(ar, bool):
            ar = 'extend' if ar else 'off'
        ar = str(ar).lower()
        if ar == 'off':
            self._adaptive_snap_group.setChecked(False)
        else:
            self._adaptive_snap_group.setChecked(True)
            mode_idx = {"extend": 0, "snap": 1}.get(ar, 0)
            self._adaptive_mode_combo.setCurrentIndex(mode_idx)
        self._snap_dist.setValue(int(getattr(ns, 'snap_dist', 150)))
        self._snap_bbox_scale.setValue(
            float(getattr(ns, 'snap_bbox_scale', 0.5)))
        self._snap_w_dist.setValue(float(getattr(ns, 'snap_w_dist', 1.0)))
        self._snap_w_size.setValue(float(getattr(ns, 'snap_w_size', 0.3)))
        self._snap_w_intersect.setValue(
            float(getattr(ns, 'snap_w_intersect', 0.5)))
        self._snap_switch.setValue(
            int(getattr(ns, 'snap_switch_frames', 8)))

        self._fixation_group.setChecked(
            bool(getattr(ns, 'gaze_lock', False)))
        self._dwell_frames.setValue(getattr(ns, 'dwell_frames', 15))
        self._lock_dist.setValue(getattr(ns, 'lock_dist', 100))

        self._hit_conf_gate.setValue(
            float(getattr(ns, 'hit_conf_gate', 0.0)))
        self._detect_extend.setValue(int(getattr(ns, 'detect_extend', 0)))
        scope = str(getattr(ns, 'detect_extend_scope', 'objects')).lower()
        scope_idx = {"objects": 0, "phenomena": 1, "both": 2}.get(scope, 0)
        self._detect_extend_scope.setCurrentIndex(scope_idx)

    def reset_defaults(self):
        # Ray geometry
        self._ray_length.setValue(1.0)
        self._cb_conf_ray.setChecked(False)
        self._gaze_cone.setValue(0.0)
        self._forward_gaze_thresh.setValue(5.0)
        self._gaze_tips_group.setChecked(False)
        self._tip_radius.setValue(80)
        # Adaptive snap
        self._adaptive_snap_group.setChecked(False)
        self._adaptive_mode_combo.setCurrentIndex(0)
        self._snap_dist.setValue(150)
        self._snap_bbox_scale.setValue(0.0)
        self._snap_w_dist.setValue(1.0)
        self._snap_w_size.setValue(0.0)
        self._snap_w_intersect.setValue(0.5)
        self._snap_switch.setValue(8)
        # Fixation lock-on
        self._fixation_group.setChecked(False)
        self._dwell_frames.setValue(15)
        self._lock_dist.setValue(100)
        # Hit detection
        self._hit_conf_gate.setValue(0.0)
        self._detect_extend.setValue(0)
        self._detect_extend_scope.setCurrentIndex(0)
