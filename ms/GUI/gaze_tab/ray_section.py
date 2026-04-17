"""Gaze ray geometry, adaptive snap, fixation lock-on, and hit detection."""

from __future__ import annotations

from argparse import Namespace

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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
        self._build_gazelle_blend(lay)
        self._build_adaptive_snap(lay)
        self._build_fixation_lockon(lay)
        self._build_hit_detection(lay)
        self._build_depth_estimation(lay)

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

    # -- Gazelle Blend -------------------------------------------------------

    def _build_gazelle_blend(self, lay):
        self._gazelle_group = QGroupBox("Gazelle Blend (Ray Forming)")
        self._gazelle_group.setCheckable(True)
        self._gazelle_group.setChecked(False)
        self._gazelle_group.setToolTip(
            "Enable Gaze-LLE heatmap fusion with pitch/yaw rays.\n"
            "Requires a Gaze-LLE checkpoint (.pt).")
        f = QFormLayout(self._gazelle_group)

        # -- Model selection --
        lbl_model = QLabel("Gaze-LLE Model")
        lbl_model.setStyleSheet(
            "color:#888; font-size:11px; margin-top:2px;")
        f.addRow(lbl_model)

        self._gazelle_model_path = QLineEdit()
        self._gazelle_model_path.setPlaceholderText("Path to .pt checkpoint")
        self._gazelle_model_path.setToolTip(
            "Path to a Gaze-LLE checkpoint file (.pt).")
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_gazelle_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self._gazelle_model_path, 1)
        model_row.addWidget(browse_btn)
        f.addRow("Model:", model_row)

        self._gazelle_name_combo = QComboBox()
        self._gazelle_name_combo.addItems([
            "gazelle_dinov2_vitb14",
            "gazelle_dinov2_vitb14_inout",
            "gazelle_dinov2_vitl14",
            "gazelle_dinov2_vitl14_inout",
        ])
        self._gazelle_name_combo.setToolTip(
            "Gaze-LLE model variant.\n"
            "vitb14 = ViT-B/14 (faster), vitl14 = ViT-L/14 (slightly better).\n"
            "_inout variants predict whether gaze target is in-frame.")
        f.addRow("Variant:", self._gazelle_name_combo)

        self._gazelle_interval = QSpinBox()
        self._gazelle_interval.setRange(1, 120)
        self._gazelle_interval.setValue(30)
        self._gazelle_interval.setToolTip(
            "Run Gaze-LLE inference every N frames.\n"
            "Lower = more accurate but slower. 30 is a good default.")
        f.addRow("Inference interval:", self._gazelle_interval)

        # -- Blend parameters --
        lbl_blend = QLabel("Blend Parameters")
        lbl_blend.setStyleSheet(
            "color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_blend)

        self._direction_blend = QDoubleSpinBox()
        self._direction_blend.setRange(0.0, 1.0)
        self._direction_blend.setSingleStep(0.05)
        self._direction_blend.setValue(1.0)
        self._direction_blend.setDecimals(2)
        self._direction_blend.setToolTip(
            "Direction blend strength.\n"
            "0.0 = pure pitch/yaw direction, 1.0 = full Gazelle correction.")
        f.addRow("Direction blend:", self._direction_blend)

        self._length_blend = QDoubleSpinBox()
        self._length_blend.setRange(0.0, 1.0)
        self._length_blend.setSingleStep(0.05)
        self._length_blend.setValue(1.0)
        self._length_blend.setDecimals(2)
        self._length_blend.setToolTip(
            "Length/reach blend strength.\n"
            "0.0 = pitch/yaw-derived length only,\n"
            "1.0 = full Gazelle ray extension.")
        f.addRow("Length blend:", self._length_blend)

        self._cb_length_only = QCheckBox("Length only (direction from pitch/yaw)")
        self._cb_length_only.setToolTip(
            "When checked, Gazelle only influences ray reach/length,\n"
            "not direction. Useful for preserving pitch/yaw temporal\n"
            "gaze movement while using Gazelle for ray extension.")
        f.addRow(self._cb_length_only)

        # -- Belief map tuning --
        lbl_belief = QLabel("Belief Map Tuning")
        lbl_belief.setStyleSheet(
            "color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_belief)

        self._direction_decay = QDoubleSpinBox()
        self._direction_decay.setRange(0.0, 1.0)
        self._direction_decay.setSingleStep(0.05)
        self._direction_decay.setValue(0.30)
        self._direction_decay.setDecimals(2)
        self._direction_decay.setToolTip(
            "How quickly the ray direction responds to changes.\n"
            "Higher = direction follows belief centroid faster.\n"
            "Lower = direction changes more smoothly/slowly.")
        f.addRow("Direction response:", self._direction_decay)

        self._length_decay = QDoubleSpinBox()
        self._length_decay.setRange(0.0, 1.0)
        self._length_decay.setSingleStep(0.05)
        self._length_decay.setValue(0.15)
        self._length_decay.setDecimals(2)
        self._length_decay.setToolTip(
            "How quickly the ray length/reach responds to changes.\n"
            "Lower = ray reach persists longer between Gazelle updates.\n"
            "Higher = ray length follows belief more responsively.\n"
            "Typically set lower than direction response so ray\n"
            "reach holds while direction may shift.")
        f.addRow("Length response:", self._length_decay)

        self._diffusion_sigma = QDoubleSpinBox()
        self._diffusion_sigma.setRange(0.0, 3.0)
        self._diffusion_sigma.setSingleStep(0.05)
        self._diffusion_sigma.setValue(0.40)
        self._diffusion_sigma.setDecimals(2)
        self._diffusion_sigma.setToolTip(
            "Per-frame Gaussian blur sigma on the belief map.\n"
            "Controls how fast the belief spreads (forgets)\n"
            "between Gazelle updates. Higher = faster decay\n"
            "of Gazelle correction confidence. 0 = no decay.")
        f.addRow("Diffusion sigma:", self._diffusion_sigma)

        self._blend_conf_scale = QDoubleSpinBox()
        self._blend_conf_scale.setRange(0.0, 1.0)
        self._blend_conf_scale.setSingleStep(0.05)
        self._blend_conf_scale.setValue(0.70)
        self._blend_conf_scale.setDecimals(2)
        self._blend_conf_scale.setToolTip(
            "How much gaze confidence tightens the PY prior.\n"
            "Higher = confident gaze estimates produce narrower\n"
            "priors that steer the belief more strongly.")
        f.addRow("Conf scale:", self._blend_conf_scale)

        self._inout_threshold = QDoubleSpinBox()
        self._inout_threshold.setRange(0.0, 1.0)
        self._inout_threshold.setSingleStep(0.05)
        self._inout_threshold.setValue(0.5)
        self._inout_threshold.setDecimals(2)
        self._inout_threshold.setToolTip(
            "Suppress Gaze-LLE heatmap when in/out score is below this.\n"
            "Relevant for _inout model variants only.")
        f.addRow("In/out threshold:", self._inout_threshold)

        lay.addWidget(self._gazelle_group)

    def _browse_gazelle_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Gaze-LLE checkpoint", "",
            "PyTorch checkpoints (*.pt *.pth);;All files (*)")
        if path:
            self._gazelle_model_path.setText(path)

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

        # -- Scoring Weights sub-heading --
        lbl_scoring = QLabel("Scoring Weights")
        lbl_scoring.setStyleSheet(
            "color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_scoring)

        self._snap_w_dist = QDoubleSpinBox()
        self._snap_w_dist.setRange(0.0, 3.0)
        self._snap_w_dist.setSingleStep(0.1)
        self._snap_w_dist.setValue(1.0)
        self._snap_w_dist.setDecimals(2)
        self._snap_w_dist.setToolTip("Penalty for distance from ray to object")
        f.addRow("W distance:", self._snap_w_dist)

        self._snap_w_angle = QDoubleSpinBox()
        self._snap_w_angle.setRange(0.0, 3.0)
        self._snap_w_angle.setSingleStep(0.1)
        self._snap_w_angle.setValue(0.8)
        self._snap_w_angle.setDecimals(2)
        self._snap_w_angle.setToolTip(
            "Penalty for angular deviation from blended gaze+head direction")
        f.addRow("W angle:", self._snap_w_angle)

        self._snap_w_size = QDoubleSpinBox()
        self._snap_w_size.setRange(0.0, 3.0)
        self._snap_w_size.setSingleStep(0.1)
        self._snap_w_size.setValue(0.0)
        self._snap_w_size.setDecimals(2)
        self._snap_w_size.setToolTip("Reward for larger objects (off by default)")
        f.addRow("W size:", self._snap_w_size)

        self._snap_w_intersect = QDoubleSpinBox()
        self._snap_w_intersect.setRange(0.0, 3.0)
        self._snap_w_intersect.setSingleStep(0.1)
        self._snap_w_intersect.setValue(0.5)
        self._snap_w_intersect.setDecimals(2)
        self._snap_w_intersect.setToolTip(
            "Bonus when ray passes through object bounding box")
        f.addRow("W intersect:", self._snap_w_intersect)

        self._snap_w_temporal = QDoubleSpinBox()
        self._snap_w_temporal.setRange(0.0, 3.0)
        self._snap_w_temporal.setSingleStep(0.1)
        self._snap_w_temporal.setValue(0.3)
        self._snap_w_temporal.setDecimals(2)
        self._snap_w_temporal.setToolTip(
            "Stickiness bonus for the previous frame's snap target")
        f.addRow("W temporal:", self._snap_w_temporal)

        # -- Angular Plausibility sub-heading --
        lbl_angular = QLabel("Angular Plausibility")
        lbl_angular.setStyleSheet(
            "color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_angular)

        self._snap_gate_angle = QDoubleSpinBox()
        self._snap_gate_angle.setRange(10.0, 180.0)
        self._snap_gate_angle.setSingleStep(5.0)
        self._snap_gate_angle.setValue(60.0)
        self._snap_gate_angle.setDecimals(1)
        self._snap_gate_angle.setToolTip(
            "Hard angular cutoff: objects beyond this angle from the\n"
            "blended gaze+head direction are never snap candidates")
        f.addRow("Gate angle (\u00b0):", self._snap_gate_angle)

        self._snap_head_blend = QDoubleSpinBox()
        self._snap_head_blend.setRange(0.0, 1.0)
        self._snap_head_blend.setSingleStep(0.05)
        self._snap_head_blend.setValue(0.3)
        self._snap_head_blend.setDecimals(2)
        self._snap_head_blend.setToolTip(
            "Blend factor: 0 = pure gaze direction, 1 = pure head orientation")
        f.addRow("Head blend:", self._snap_head_blend)

        self._snap_quality_thresh = QDoubleSpinBox()
        self._snap_quality_thresh.setRange(0.1, 3.0)
        self._snap_quality_thresh.setSingleStep(0.1)
        self._snap_quality_thresh.setValue(0.8)
        self._snap_quality_thresh.setDecimals(2)
        self._snap_quality_thresh.setToolTip(
            "Maximum score to accept a snap match.\n"
            "Higher = more permissive. Lower = stricter.")
        f.addRow("Quality threshold:", self._snap_quality_thresh)

        # -- Stabilization sub-heading --
        lbl_stab = QLabel("Stabilization")
        lbl_stab.setStyleSheet(
            "color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_stab)

        self._snap_release = QSpinBox()
        self._snap_release.setRange(1, 30)
        self._snap_release.setValue(5)
        self._snap_release.setToolTip(
            "Frames of no-match before releasing the held snap target")
        f.addRow("Release frames:", self._snap_release)

        self._snap_engage = QSpinBox()
        self._snap_engage.setRange(0, 30)
        self._snap_engage.setValue(0)
        self._snap_engage.setToolTip(
            "Frames of consistent match required before engaging snap.\n"
            "0 = instant engage.")
        f.addRow("Engage frames:", self._snap_engage)

        # -- Tip Snap Overrides sub-heading --
        lbl_tip = QLabel("Tip Snap Overrides")
        lbl_tip.setStyleSheet(
            "color:#888; font-size:11px; margin-top:4px;")
        f.addRow(lbl_tip)

        self._snap_tip_dist = QDoubleSpinBox()
        self._snap_tip_dist.setRange(-1.0, 500.0)
        self._snap_tip_dist.setSingleStep(10.0)
        self._snap_tip_dist.setValue(-1.0)
        self._snap_tip_dist.setDecimals(1)
        self._snap_tip_dist.setToolTip(
            "Independent distance threshold for tip snapping.\n"
            "-1 = use main snap dist.")
        f.addRow("Tip dist (px):", self._snap_tip_dist)

        self._snap_tip_quality = QDoubleSpinBox()
        self._snap_tip_quality.setRange(-1.0, 3.0)
        self._snap_tip_quality.setSingleStep(0.1)
        self._snap_tip_quality.setValue(-1.0)
        self._snap_tip_quality.setDecimals(2)
        self._snap_tip_quality.setToolTip(
            "Independent quality threshold for tip snapping.\n"
            "-1 = use main quality threshold.")
        f.addRow("Tip quality:", self._snap_tip_quality)

        lay.addWidget(self._adaptive_snap_group)

        # Ray Forming Smoothing (independent of adaptive snap)
        self._smooth_group = QGroupBox("Ray Forming Smoothing")
        self._smooth_group.setCheckable(True)
        self._smooth_group.setChecked(False)
        self._smooth_group.setToolTip(
            "Smoothly interpolate ray endpoints over multiple frames\n"
            "instead of jumping instantly. Works independently of\n"
            "adaptive snap -- can smooth Gazelle blend, object snaps,\n"
            "or gaze tip snaps.")
        sf = QFormLayout(self._smooth_group)

        self._smooth_snap_combo = QComboBox()
        self._smooth_snap_combo.addItems(["Objects", "Gaze Tips", "All"])
        self._smooth_snap_combo.setCurrentIndex(2)
        self._smooth_snap_combo.setToolTip(
            "Which ray endpoints to smooth:\n"
            "Objects: smooth object/face snaps only.\n"
            "Gaze Tips: smooth gaze-tip snaps only.\n"
            "All: smooth all ray endpoint transitions.")
        sf.addRow("Smooth targets:", self._smooth_snap_combo)

        self._smooth_snap_alpha = QDoubleSpinBox()
        self._smooth_snap_alpha.setRange(0.01, 1.0)
        self._smooth_snap_alpha.setSingleStep(0.05)
        self._smooth_snap_alpha.setValue(0.20)
        self._smooth_snap_alpha.setDecimals(2)
        self._smooth_snap_alpha.setToolTip(
            "EMA rate: lower = smoother/slower,\n"
            "higher = faster/more responsive")
        sf.addRow("Smooth alpha:", self._smooth_snap_alpha)

        lay.addWidget(self._smooth_group)

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

    # -- Depth Estimation ----------------------------------------------------

    def _build_depth_estimation(self, lay):
        self._depth_group = QGroupBox("Depth Estimation")
        self._depth_group.setCheckable(True)
        self._depth_group.setChecked(False)
        self._depth_group.setToolTip(
            "Enable monocular depth estimation for depth-aware\n"
            "gaze-object scoring and hit-event enrichment.")
        f = QFormLayout(self._depth_group)

        self._depth_backend = QComboBox()
        self._depth_backend.addItems(["midas_small"])
        self._depth_backend.setToolTip("Depth model backend.")
        f.addRow("Backend:", self._depth_backend)

        self._depth_input_size = QSpinBox()
        self._depth_input_size.setRange(256, 512)
        self._depth_input_size.setSingleStep(64)
        self._depth_input_size.setValue(384)
        self._depth_input_size.setToolTip(
            "Model input resolution (smaller = faster).")
        f.addRow("Input size (px):", self._depth_input_size)

        self._depth_skip_frames = QSpinBox()
        self._depth_skip_frames.setRange(1, 10)
        self._depth_skip_frames.setValue(1)
        self._depth_skip_frames.setToolTip(
            "Run depth every N detection cycles.")
        f.addRow("Skip frames:", self._depth_skip_frames)

        self._depth_aware_scoring = QCheckBox("Enable depth-weighted snap scoring")
        self._depth_aware_scoring.setToolTip(
            "When enabled, snap scoring uses depth to prefer the\n"
            "object whose depth best matches the gaze termination\n"
            "point. Off by default -- hit events always get depth\n"
            "metadata regardless of this setting.")
        f.addRow(self._depth_aware_scoring)

        self._depth_w_depth = QDoubleSpinBox()
        self._depth_w_depth.setRange(0.0, 2.0)
        self._depth_w_depth.setSingleStep(0.05)
        self._depth_w_depth.setValue(0.4)
        self._depth_w_depth.setDecimals(2)
        self._depth_w_depth.setToolTip(
            "Depth match weight in snap scoring.")
        f.addRow("Depth weight:", self._depth_w_depth)

        lay.addWidget(self._depth_group)

    # -- Namespace interface --------------------------------------------------

    def namespace_values(self) -> dict:
        return dict(
            ray_length=self._ray_length.value(),
            conf_ray=self._cb_conf_ray.isChecked(),
            gaze_cone=self._gaze_cone.value(),
            forward_gaze_threshold=self._forward_gaze_thresh.value(),
            gaze_tips=self._gaze_tips_group.isChecked(),
            tip_radius=self._tip_radius.value(),
            # Gazelle blend
            rf_gazelle_model=(self._gazelle_model_path.text().strip() or None
                              if self._gazelle_group.isChecked() else None),
            rf_gazelle_name=self._gazelle_name_combo.currentText(),
            rf_gazelle_interval=self._gazelle_interval.value(),
            direction_blend=self._direction_blend.value(),
            length_blend=self._length_blend.value(),
            length_only=self._cb_length_only.isChecked(),
            direction_decay=self._direction_decay.value(),
            length_decay=self._length_decay.value(),
            diffusion_sigma=self._diffusion_sigma.value(),
            blend_conf_scale=self._blend_conf_scale.value(),
            inout_threshold=self._inout_threshold.value(),
            adaptive_ray=(self._adaptive_mode_combo.currentText().lower()
                          if self._adaptive_snap_group.isChecked()
                          else "off"),
            snap_dist=float(self._snap_dist.value()),
            snap_bbox_scale=self._snap_bbox_scale.value(),
            snap_w_dist=self._snap_w_dist.value(),
            snap_w_angle=self._snap_w_angle.value(),
            snap_w_size=self._snap_w_size.value(),
            snap_w_intersect=self._snap_w_intersect.value(),
            snap_w_temporal=self._snap_w_temporal.value(),
            snap_gate_angle=self._snap_gate_angle.value(),
            snap_head_blend=self._snap_head_blend.value(),
            snap_quality_thresh=self._snap_quality_thresh.value(),
            snap_release_frames=self._snap_release.value(),
            snap_engage_frames=self._snap_engage.value(),
            snap_tip_dist=self._snap_tip_dist.value(),
            snap_tip_quality=self._snap_tip_quality.value(),
            smooth_snap=({"Objects": "objects", "Gaze Tips": "gaze_tips",
                          "All": "all"}[self._smooth_snap_combo.currentText()]
                         if self._smooth_group.isChecked() else "off"),
            smooth_snap_alpha=self._smooth_snap_alpha.value(),
            gaze_lock=self._fixation_group.isChecked(),
            dwell_frames=self._dwell_frames.value(),
            lock_dist=self._lock_dist.value(),
            hit_conf_gate=self._hit_conf_gate.value(),
            detect_extend=float(self._detect_extend.value()),
            detect_extend_scope=(
                self._detect_extend_scope.currentText().lower()),
            depth=self._depth_group.isChecked(),
            depth_backend=self._depth_backend.currentText(),
            depth_input_size=self._depth_input_size.value(),
            depth_skip_frames=self._depth_skip_frames.value(),
            depth_aware_scoring=self._depth_aware_scoring.isChecked(),
            depth_w_depth=self._depth_w_depth.value(),
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

        # Gazelle blend
        gz_model = (getattr(ns, 'rf_gazelle_model', None)
                    or getattr(ns, 'gs_gazelle_model', None) or '')
        self._gazelle_group.setChecked(bool(gz_model))
        self._gazelle_model_path.setText(gz_model)
        gz_name = getattr(ns, 'rf_gazelle_name',
                   getattr(ns, 'gs_gazelle_name', 'gazelle_dinov2_vitb14'))
        gz_name_idx = self._gazelle_name_combo.findText(gz_name)
        if gz_name_idx >= 0:
            self._gazelle_name_combo.setCurrentIndex(gz_name_idx)
        self._gazelle_interval.setValue(
            int(getattr(ns, 'rf_gazelle_interval',
                getattr(ns, 'gs_snap_interval', 30))))
        self._direction_blend.setValue(
            float(getattr(ns, 'direction_blend',
                   getattr(ns, 'blend_strength', 1.0))))
        self._length_blend.setValue(
            float(getattr(ns, 'length_blend',
                   getattr(ns, 'blend_strength', 1.0))))
        self._cb_length_only.setChecked(
            bool(getattr(ns, 'length_only', False)))
        self._direction_decay.setValue(
            float(getattr(ns, 'direction_decay', 0.30)))
        self._length_decay.setValue(
            float(getattr(ns, 'length_decay', 0.15)))
        self._diffusion_sigma.setValue(
            float(getattr(ns, 'diffusion_sigma', 0.40)))
        self._blend_conf_scale.setValue(
            float(getattr(ns, 'blend_conf_scale', 0.7)))
        self._inout_threshold.setValue(
            float(getattr(ns, 'inout_threshold', 0.5)))

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
            float(getattr(ns, 'snap_bbox_scale', 0.0)))
        self._snap_w_dist.setValue(float(getattr(ns, 'snap_w_dist', 1.0)))
        self._snap_w_angle.setValue(float(getattr(ns, 'snap_w_angle', 0.8)))
        self._snap_w_size.setValue(float(getattr(ns, 'snap_w_size', 0.0)))
        self._snap_w_intersect.setValue(
            float(getattr(ns, 'snap_w_intersect', 0.5)))
        self._snap_w_temporal.setValue(
            float(getattr(ns, 'snap_w_temporal', 0.3)))
        self._snap_gate_angle.setValue(
            float(getattr(ns, 'snap_gate_angle', 60.0)))
        self._snap_head_blend.setValue(
            float(getattr(ns, 'snap_head_blend', 0.3)))
        self._snap_quality_thresh.setValue(
            float(getattr(ns, 'snap_quality_thresh', 0.8)))
        self._snap_release.setValue(
            int(getattr(ns, 'snap_release_frames', 5)))
        self._snap_engage.setValue(
            int(getattr(ns, 'snap_engage_frames', 0)))
        self._snap_tip_dist.setValue(
            float(getattr(ns, 'snap_tip_dist', -1.0)))
        self._snap_tip_quality.setValue(
            float(getattr(ns, 'snap_tip_quality', -1.0)))
        ss = str(getattr(ns, 'smooth_snap', 'off')).lower()
        self._smooth_group.setChecked(ss != 'off')
        ss_idx = {"objects": 0, "gaze_tips": 1, "all": 2}.get(ss, 2)
        self._smooth_snap_combo.setCurrentIndex(ss_idx)
        self._smooth_snap_alpha.setValue(
            float(getattr(ns, 'smooth_snap_alpha', 0.20)))

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
        # Gazelle blend
        self._gazelle_group.setChecked(False)
        self._gazelle_model_path.clear()
        self._gazelle_name_combo.setCurrentIndex(0)
        self._gazelle_interval.setValue(30)
        self._direction_blend.setValue(1.0)
        self._length_blend.setValue(1.0)
        self._cb_length_only.setChecked(False)
        self._direction_decay.setValue(0.30)
        self._length_decay.setValue(0.15)
        self._diffusion_sigma.setValue(0.40)
        self._blend_conf_scale.setValue(0.7)
        self._inout_threshold.setValue(0.5)
        # Adaptive snap
        self._adaptive_snap_group.setChecked(False)
        self._adaptive_mode_combo.setCurrentIndex(0)
        self._snap_dist.setValue(150)
        self._snap_bbox_scale.setValue(0.0)
        self._snap_w_dist.setValue(1.0)
        self._snap_w_angle.setValue(0.8)
        self._snap_w_size.setValue(0.0)
        self._snap_w_intersect.setValue(0.5)
        self._snap_w_temporal.setValue(0.3)
        self._snap_gate_angle.setValue(60.0)
        self._snap_head_blend.setValue(0.3)
        self._snap_quality_thresh.setValue(0.8)
        self._snap_release.setValue(5)
        self._snap_engage.setValue(0)
        self._snap_tip_dist.setValue(-1.0)
        self._snap_tip_quality.setValue(-1.0)
        self._smooth_group.setChecked(False)
        self._smooth_snap_combo.setCurrentIndex(2)
        self._smooth_snap_alpha.setValue(0.20)
        # Fixation lock-on
        self._fixation_group.setChecked(False)
        self._dwell_frames.setValue(15)
        self._lock_dist.setValue(100)
        # Hit detection
        self._hit_conf_gate.setValue(0.0)
        self._detect_extend.setValue(0)
        self._detect_extend_scope.setCurrentIndex(0)
