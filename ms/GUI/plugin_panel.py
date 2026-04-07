"""
GUI/plugin_panel.py — Dynamic plugin argument panel.

Discovers installed plugins via the MindSight plugin registries, introspects
their CLI arguments, and renders appropriate Qt widgets for each argument.
New plugins get UI controls automatically without any GUI code changes.

Plugins are organized by type (Gaze, Detection, Phenomena, Data Collection)
with human-readable names, activation toggles, logical subsections, and
mode-dependent parameter visibility.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .arg_introspector import ArgSpec, introspect_plugin

# Argument dest names already handled by hardcoded widgets in gaze_tab.py
# and phenomena_panel.py — these are excluded from the dynamic plugin panel.
_HANDLED_DESTS = {
    # Global
    "device",
    # Detection
    "model", "conf", "classes", "blacklist", "detect_scale", "vp_file",
    "vp_model", "skip_frames", "obj_persistence",
    # Gaze
    "mgaze_model", "mgaze_arch", "mgaze_dataset", "ray_length", "adaptive_ray",
    "snap_dist", "snap_bbox_scale", "snap_w_dist", "snap_w_size", "snap_w_intersect",
    "conf_ray", "gaze_tips", "tip_radius",
    "gaze_cone", "gaze_lock", "dwell_frames", "lock_dist", "gaze_debug",
    "snap_switch_frames", "reid_grace_seconds",
    # L2CS-Net
    "l2cs_model", "l2cs_arch", "l2cs_dataset",
    # UniGaze
    "unigaze_model",
    # Gazelle
    "gazelle_model", "gazelle_name", "gazelle_inout_threshold",
    "gazelle_device", "gazelle_skip_frames", "gazelle_fp16", "gazelle_compile",
    # Output
    "source", "save", "log", "summary", "heatmap", "pipeline", "project",
    # Phenomena (handled by phenomena_panel)
    "mutual_gaze", "social_ref", "social_ref_window", "gaze_follow",
    "gaze_follow_lag", "gaze_aversion", "aversion_window", "aversion_conf",
    "scanpath", "scanpath_dwell", "gaze_leader", "attn_span", "all_phenomena",
    "ja_window", "ja_window_thresh", "ja_quorum", "hit_conf_gate",
    "detect_extend", "detect_extend_scope",
}

# Common abbreviation expansions for parameter labels
_ABBREV = {
    "Thresh": "Threshold",
    "Conf": "Confidence",
    "Dist": "Distance",
    "Pct": "Percent",
    "Src": "Source",
    "Dir": "Direction",
    "Num": "Number",
    "Max": "Maximum",
    "Min": "Minimum",
    "Freq": "Frequency",
    "Vel": "Velocity",
    "Meas": "Measurement",
}


# ══════════════════════════════════════════════════════════════════════════════
# Plugin layout declarations — subsections and mode-dependent visibility
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Section:
    """A named subsection within a plugin's settings panel."""
    label: str
    dests: list[str]
    # If set, this section is only visible when mode_dest has mode_value
    mode_dest: str | None = None
    mode_value: str | None = None


@dataclass
class _PluginLayout:
    """Declarative layout for a plugin's GUI panel."""
    sections: list[_Section] = field(default_factory=list)


# Layouts keyed by plugin name (pname from registry).
# Parameters not listed in any section fall into an auto-generated group.
_PLUGIN_LAYOUTS: dict[str, _PluginLayout] = {
    "pupillometry": _PluginLayout(sections=[
        _Section("Measurement Mode", ["pupil_mode"]),
        _Section("Calibration", ["pupil_baseline"]),
        _Section("RGB Settings", ["pupil_upscale"],
                 mode_dest="pupil_mode", mode_value="rgb"),
        _Section("IR Settings", ["pupil_ir_thresh"],
                 mode_dest="pupil_mode", mode_value="ir"),
        _Section("Smoothing Filter", ["pupil_filter"]),
        _Section("EMA Smoothing", ["pupil_ema_alpha"],
                 mode_dest="pupil_filter", mode_value="ema"),
        _Section("Kalman Smoothing",
                 ["pupil_kalman_process_noise", "pupil_kalman_meas_noise"],
                 mode_dest="pupil_filter", mode_value="kalman"),
        _Section("Blink Detection", ["pupil_ear_thresh", "pupil_blink_frames"]),
        _Section("Outlier Rejection", ["pupil_outlier_window"]),
        _Section("Output", ["pupil_per_eye"]),
    ]),
    "eye_movement": _PluginLayout(sections=[
        _Section("Velocity Source", ["em_source"]),
        _Section("Classification Thresholds",
                 ["em_saccade_thresh", "em_fixation_thresh", "em_min_fixation"]),
        _Section("Smoothing", ["em_velocity_window"]),
    ]),
    "novel_salience": _PluginLayout(sections=[
        _Section("Detection", ["ns_speed_thresh", "ns_cooldown"]),
        _Section("Smoothing", ["ns_history"]),
        _Section("Display", ["ns_flash"]),
    ]),
    "gaze_boost": _PluginLayout(sections=[
        _Section("Boost", ["gaze_boost_factor", "gaze_boost_radius"]),
        _Section("Confidence Bounds",
                 ["gaze_boost_min_conf", "gaze_boost_max_conf"]),
        _Section("Class Filter", ["gaze_boost_classes"]),
    ]),
    "iris_refined": _PluginLayout(sections=[
        _Section("Refinement", ["iris_refine_weight", "iris_refine_upscale"]),
    ]),
    "gazelle_snap": _PluginLayout(sections=[
        _Section("Model", ["gs_gazelle_model", "gs_gazelle_name"]),
        _Section("Heatmap Inference",
                 ["gs_snap_interval", "gs_heatmap_threshold",
                  "gs_heatmap_weight", "gs_heatmap_decay"]),
        _Section("Object Snapping", ["gs_obj_snap"]),
    ]),
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _derive_flag_prefix(specs: list[ArgSpec]) -> str:
    """Derive the common CLI flag prefix for a plugin's arguments."""
    flags = [s.flag.lstrip("-") for s in specs
             if s.action not in ("store_true", "store_false")]
    if not flags:
        return ""
    parts_list = [f.split("-") for f in flags]
    if not parts_list:
        return ""
    prefix_parts = []
    for i, segment in enumerate(parts_list[0]):
        if all(len(p) > i and p[i] == segment for p in parts_list):
            prefix_parts.append(segment)
        else:
            break
    return "-".join(prefix_parts)


def _make_label(spec: ArgSpec, prefix: str) -> str:
    """Create a human-readable label from an ArgSpec, stripping the plugin prefix."""
    raw = spec.flag.lstrip("-")
    if prefix and raw.startswith(prefix + "-"):
        raw = raw[len(prefix) + 1:]
    label = raw.replace("-", " ").title()
    for short, full in _ABBREV.items():
        label = re.sub(rf'\b{short}\b', full, label)
    return label


def _clean_group_title(group_title: str, pname: str) -> str:
    """Derive a clean display name from the argparse group title or plugin name."""
    if group_title:
        title = group_title.strip()
        for suffix in (" plugin", " Plugin", " backend", " Backend"):
            if title.endswith(suffix):
                title = title[:-len(suffix)].strip()
        return title
    return pname.replace("_", " ").title()


def _find_activation_spec(specs: list[ArgSpec], pname: str) -> ArgSpec | None:
    """Find the store_true spec that serves as the plugin's activation toggle."""
    bool_specs = [s for s in specs if s.action == "store_true"]
    if not bool_specs:
        return None
    for s in bool_specs:
        if s.dest == pname:
            return s
    for s in bool_specs:
        if pname.startswith(s.dest):
            return s
    if len(bool_specs) == 1:
        return bool_specs[0]
    return None


def _make_widget(spec: ArgSpec):
    """Create a Qt widget for an ArgSpec, returning (widget, getter_fn, setter_fn)."""
    if spec.action in ("store_true", "store_false"):
        w = QCheckBox()
        w.setChecked(bool(spec.default) if spec.action == "store_true"
                     else not bool(spec.default))
        w.setToolTip(spec.help)
        return w, w.isChecked, w.setChecked

    if spec.choices:
        w = QComboBox()
        w.addItems([str(c) for c in spec.choices])
        if spec.default is not None:
            w.setCurrentText(str(spec.default))
        w.setToolTip(spec.help)
        return w, w.currentText, w.setCurrentText

    if spec.type is float or (spec.type is None and isinstance(spec.default, float)):
        w = QDoubleSpinBox()
        w.setRange(-9999.0, 9999.0)
        w.setSingleStep(0.1)
        w.setDecimals(2)
        if spec.default is not None:
            w.setValue(float(spec.default))
        w.setToolTip(spec.help)
        return w, w.value, w.setValue

    if spec.type is int or (spec.type is None and isinstance(spec.default, int)
                            and spec.action == "store"):
        w = QSpinBox()
        w.setRange(-9999, 9999)
        if spec.default is not None:
            w.setValue(int(spec.default))
        w.setToolTip(spec.help)
        return w, w.value, w.setValue

    # Fallback: string input
    w = QLineEdit()
    if spec.default is not None:
        w.setText(str(spec.default))
    w.setPlaceholderText(spec.help[:60] if spec.help else "")
    w.setToolTip(spec.help)
    return w, w.text, w.setText


def _subsection_label(text: str) -> QLabel:
    """Subsection header matching the existing GUI convention (ray_section.py)."""
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#888; font-size:11px; margin-top:4px;")
    return lbl



# ══════════════════════════════════════════════════════════════════════════════
# Main panel widget
# ══════════════════════════════════════════════════════════════════════════════

class PluginPanel(QWidget):
    """Dynamically renders UI controls for all discovered plugin arguments."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # {dest: (widget, getter_fn, setter_fn, type_or_None)}
        self._controls: dict[str, tuple] = {}
        # {mode_dest: QComboBox} for mode selector widgets
        self._mode_combos: dict[str, QComboBox] = {}
        # {mode_dest: {mode_value: [widgets_to_show]}} for mode-dependent visibility
        self._mode_sections: dict[str, dict[str, list[QWidget]]] = {}
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        plugins_found = False
        try:
            from Plugins import (
                data_collection_registry,
                gaze_registry,
                object_detection_registry,
                phenomena_registry,
            )
            registries = [
                ("Gaze Plugins", gaze_registry),
                ("Detection Plugins", object_detection_registry),
                ("Phenomena Plugins", phenomena_registry),
                ("Data Collection Plugins", data_collection_registry),
            ]
        except ImportError:
            lay.addWidget(QLabel("Plugin system not available."))
            return

        for reg_label, registry in registries:
            reg_plugins = []
            for pname in registry.names():
                pcls = registry.get(pname)
                specs = introspect_plugin(pcls)
                specs = [s for s in specs if s.dest not in _HANDLED_DESTS]
                if specs:
                    reg_plugins.append((pname, specs))

            if not reg_plugins:
                continue

            plugins_found = True
            cat_box = QGroupBox(reg_label)
            cat_lay = QVBoxLayout(cat_box)
            cat_lay.setContentsMargins(6, 6, 6, 6)
            cat_lay.setSpacing(4)

            for pname, specs in reg_plugins:
                layout = _PLUGIN_LAYOUTS.get(pname)
                if layout and layout.sections:
                    self._build_structured_group(cat_lay, pname, specs, layout)
                else:
                    self._build_flat_group(cat_lay, pname, specs)

            lay.addWidget(cat_box)

        if not plugins_found:
            lay.addWidget(QLabel("No additional plugin settings discovered."))

        self._sync_all_mode_visibility()

    # ── Structured layout (with subsections + mode visibility) ────────────────

    def _build_structured_group(self, lay: QVBoxLayout, pname: str,
                                specs: list[ArgSpec],
                                plugin_layout: _PluginLayout):
        """Build a plugin group with declared subsections and mode visibility.

        Uses a standalone QCheckBox for the activation toggle and a QGroupBox
        for the content so the entire box disappears when the plugin is off.
        """
        first_title = specs[0].group_title if specs else ""
        display_name = _clean_group_title(first_title, pname)

        activation = _find_activation_spec(specs, pname)
        param_specs = [s for s in specs if s is not activation]
        prefix = _derive_flag_prefix(param_specs)
        spec_by_dest = {s.dest: s for s in param_specs}

        # Activation checkbox -- always visible
        if activation:
            cb = QCheckBox(display_name)
            cb.setChecked(bool(activation.default))
            cb.setToolTip(activation.help)
            self._controls[activation.dest] = (
                cb, cb.isChecked, cb.setChecked, None)
            lay.addWidget(cb)

        # Content box -- hidden when activation is unchecked
        content = QGroupBox()
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(6, 4, 6, 4)
        content_lay.setSpacing(2)

        placed_dests: set[str] = set()

        for section in plugin_layout.sections:
            section_specs = []
            for dest in section.dests:
                if dest in spec_by_dest:
                    section_specs.append(spec_by_dest[dest])
                    placed_dests.add(dest)
            if not section_specs:
                continue

            # Container for mode-dependent show/hide
            section_container = QWidget()
            section_lay = QVBoxLayout(section_container)
            section_lay.setContentsMargins(0, 0, 0, 0)
            section_lay.setSpacing(1)

            section_lay.addWidget(_subsection_label(section.label))

            form = QFormLayout()
            form.setContentsMargins(8, 0, 0, 2)
            form.setSpacing(3)

            for spec in section_specs:
                widget, getter, setter = _make_widget(spec)
                label = _make_label(spec, prefix)
                form.addRow(f"{label}:", widget)
                self._controls[spec.dest] = (widget, getter, setter, spec.type)
                if isinstance(widget, QComboBox) and spec.choices:
                    self._mode_combos[spec.dest] = widget

            section_lay.addLayout(form)
            content_lay.addWidget(section_container)

            if section.mode_dest and section.mode_value:
                if section.mode_dest not in self._mode_sections:
                    self._mode_sections[section.mode_dest] = {}
                mode_map = self._mode_sections[section.mode_dest]
                if section.mode_value not in mode_map:
                    mode_map[section.mode_value] = []
                mode_map[section.mode_value].append(section_container)

        # Leftover params not in any section
        leftover = [s for s in param_specs if s.dest not in placed_dests]
        if leftover:
            content_lay.addWidget(_subsection_label("Other"))
            form = QFormLayout()
            form.setContentsMargins(8, 0, 0, 2)
            form.setSpacing(3)
            for spec in leftover:
                widget, getter, setter = _make_widget(spec)
                label = _make_label(spec, prefix)
                form.addRow(f"{label}:", widget)
                self._controls[spec.dest] = (widget, getter, setter, spec.type)
            content_lay.addLayout(form)

        lay.addWidget(content)

        if activation:
            cb.toggled.connect(content.setVisible)
            content.setVisible(cb.isChecked())

    # ── Flat layout (fallback for plugins without declared layout) ────────────

    def _build_flat_group(self, lay: QVBoxLayout, pname: str,
                          specs: list[ArgSpec]):
        """Build a simple flat plugin group (no subsections)."""
        first_title = specs[0].group_title if specs else ""
        display_name = _clean_group_title(first_title, pname)

        activation = _find_activation_spec(specs, pname)
        param_specs = [s for s in specs if s is not activation]
        prefix = _derive_flag_prefix(param_specs)

        # Activation checkbox -- always visible
        if activation:
            cb = QCheckBox(display_name)
            cb.setChecked(bool(activation.default))
            cb.setToolTip(activation.help)
            self._controls[activation.dest] = (
                cb, cb.isChecked, cb.setChecked, None)
            lay.addWidget(cb)

        # Content box -- hidden when activation is unchecked
        content = QGroupBox()
        form = QFormLayout(content)
        form.setContentsMargins(6, 4, 6, 4)
        form.setSpacing(4)

        for spec in param_specs:
            widget, getter, setter = _make_widget(spec)
            label = _make_label(spec, prefix)
            form.addRow(f"{label}:", widget)
            self._controls[spec.dest] = (widget, getter, setter, spec.type)

        lay.addWidget(content)

        if activation:
            cb.toggled.connect(content.setVisible)
            content.setVisible(cb.isChecked())

    # ── Mode-dependent visibility ─────────────────────────────────────────────

    def _sync_all_mode_visibility(self):
        """Connect mode combos to their dependent sections and set initial state."""
        for mode_dest, mode_map in self._mode_sections.items():
            combo = self._mode_combos.get(mode_dest)
            if combo is None:
                continue

            def _on_mode_changed(value, md=mode_dest):
                self._update_mode_visibility(md)

            combo.currentTextChanged.connect(_on_mode_changed)
            self._update_mode_visibility(mode_dest)

    def _update_mode_visibility(self, mode_dest: str):
        """Show/hide sections based on the current value of a mode combo."""
        combo = self._mode_combos.get(mode_dest)
        if combo is None:
            return
        current = combo.currentText()
        mode_map = self._mode_sections.get(mode_dest, {})
        for value, containers in mode_map.items():
            visible = (value == current)
            for container in containers:
                container.setVisible(visible)

    # ── Namespace interface ───────────────────────────────────────────────────

    def get_values(self) -> dict:
        """Return {dest: value} for all plugin controls."""
        result = {}
        for dest, (widget, getter, setter, typ) in self._controls.items():
            val = getter()
            if typ is float and isinstance(val, str):
                try: val = float(val)
                except ValueError: pass
            elif typ is int and isinstance(val, str):
                try: val = int(val)
                except ValueError: pass
            result[dest] = val
        return result

    def apply_values(self, d: dict):
        """Set widget values from a dict."""
        for dest, (widget, getter, setter, typ) in self._controls.items():
            if dest in d:
                val = d[dest]
                try:
                    setter(val)
                except (TypeError, ValueError):
                    pass
        # Re-sync visibility after applying values (mode combos may have changed)
        for mode_dest in self._mode_sections:
            self._update_mode_visibility(mode_dest)
