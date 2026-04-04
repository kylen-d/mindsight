"""
GUI/plugin_panel.py — Dynamic plugin argument panel.

Discovers installed plugins via the MindSight plugin registries, introspects
their CLI arguments, and renders appropriate Qt widgets for each argument.
New plugins get UI controls automatically without any GUI code changes.
"""
from __future__ import annotations

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


class PluginPanel(QWidget):
    """Dynamically renders UI controls for all discovered plugin arguments."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # {dest: (widget, getter_fn, setter_fn, type_or_None)}
        self._controls: dict[str, tuple] = {}
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Discover plugins from all registries
        plugins_found = False
        try:
            from Plugins import (
                data_collection_registry,
                gaze_registry,
                object_detection_registry,
                phenomena_registry,
            )
            registries = [
                ("Gaze", gaze_registry),
                ("Object Detection", object_detection_registry),
                ("Phenomena", phenomena_registry),
                ("Data Collection", data_collection_registry),
            ]
        except ImportError:
            lay.addWidget(QLabel("Plugin system not available."))
            return

        for reg_label, registry in registries:
            for pname in registry.names():
                pcls = registry.get(pname)
                specs = introspect_plugin(pcls)
                # Filter out already-handled args
                specs = [s for s in specs if s.dest not in _HANDLED_DESTS]
                if not specs:
                    continue

                plugins_found = True
                group_title = f"{pname}"
                grp = QGroupBox(group_title)
                grp.setCheckable(False)
                form = QFormLayout(grp)
                form.setContentsMargins(4, 4, 4, 4)

                for spec in specs:
                    widget, getter, setter = _make_widget(spec)
                    # Use a readable label from the flag name
                    label = spec.flag.lstrip("-").replace("-", " ").title()
                    form.addRow(f"{label}:", widget)
                    self._controls[spec.dest] = (widget, getter, setter, spec.type)

                lay.addWidget(grp)

        if not plugins_found:
            lay.addWidget(QLabel("No additional plugin settings discovered."))

    def get_values(self) -> dict[str, any]:
        """Return {dest: value} for all plugin controls."""
        result = {}
        for dest, (widget, getter, setter, typ) in self._controls.items():
            val = getter()
            # Coerce type if needed
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
