"""
schema_panel.py -- Qt settings panel generated from the ui-spec (SP3.1 Batch F,
D7).

``SchemaPanel(groups, show_advanced)`` renders the ``UiGroup`` tree from
``ui_spec.build_ui_spec()`` into spin / double-spin / checkbox / combo /
line-edit / path (line-edit + Browse button) controls inside (optionally
checkable) QGroupBoxes, and exposes the
SAME namespace contract as the hand-written sections it will replace:
``namespace_values()`` / ``apply_namespace(ns)`` / ``reset_defaults()``.

The checkable groups reproduce the T10 off-value semantics exactly: unchecking a
group writes its owner's off-value (``rf_gazelle_model=None`` /
``adaptive_ray="off"`` / ``smooth_snap="off"`` / ``gaze_lock=False`` /
``depth=False`` / ``gaze_tips=False`` / ``joint_attention=False`` /
``gaze_aversion=False``) while every OTHER field in the group keeps emitting its
widget value -- matching the live ray / performance / phenomena panels
(pinned by tests/test_schema_panel_equivalence.py).

The Qt layer is deliberately thin; all structure/tuning knowledge lives in
``ui_spec`` (pure) and the schema ``ui`` metadata.
"""
from __future__ import annotations

from argparse import Namespace

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mindsight.PostProcessing.RayForming.ray_config import resolve_min_call_gap

from .ui_spec import UiField, UiGroup, build_ui_spec
from .widgets import _browse_btn


class SchemaPanel(QWidget):
    """Settings panel generated from the schema ui-spec."""

    def __init__(self, groups: list[UiGroup] | None = None,
                 show_advanced: bool = False, parent=None):
        super().__init__(parent)
        self._groups = groups if groups is not None else build_ui_spec()
        self._show_advanced = show_advanced
        # dest -> (UiField, control widget)
        self._fields: dict[str, tuple[UiField, QWidget]] = {}
        # toggle owner dest -> {"group": QGroupBox, "widget": str|None,
        #                       "inner": QWidget|None, "off": value}
        self._toggles: dict[str, dict] = {}
        # advanced rows (widget containers) to show/hide
        self._advanced_rows: list[QWidget] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        for group in self._groups:
            self._build_group(group, lay)
        lay.addStretch(1)
        self.set_show_advanced(show_advanced)

    # -- construction ---------------------------------------------------------

    def _make_control(self, f: UiField) -> QWidget:
        if f.widget == "check":
            w = QCheckBox(f.label)
            w.setChecked((not f.default) if f.inverted else bool(f.default))
        elif f.widget == "spin":
            w = QSpinBox()
            if f.minimum is not None and f.maximum is not None:
                w.setRange(int(f.minimum), int(f.maximum))
            if f.step is not None:
                w.setSingleStep(int(f.step))
            w.setValue(int(f.default))
        elif f.widget == "double":
            w = QDoubleSpinBox()
            if f.decimals is not None:
                w.setDecimals(int(f.decimals))
            if f.minimum is not None and f.maximum is not None:
                w.setRange(float(f.minimum), float(f.maximum))
            if f.step is not None:
                w.setSingleStep(float(f.step))
            w.setValue(float(f.default))
        elif f.widget == "combo":
            w = QComboBox()
            w.addItems([str(c) for c in (f.choices or ())])
            if f.default is not None:
                idx = w.findText(str(f.default))
                if idx >= 0:
                    w.setCurrentIndex(idx)
        elif f.widget in ("line", "path"):
            w = QLineEdit()
            w.setText("" if f.default is None else str(f.default))
        else:  # pragma: no cover - guarded by ui_spec tests
            raise ValueError(f"unknown widget {f.widget!r} for {f.dest}")
        if f.tooltip:
            w.setToolTip(f.tooltip)
        return w

    def _path_browse_btn(self, line_edit: QLineEdit, filt: str | None):
        """A shared 'Browse...' button that fills *line_edit* from a file
        dialog (same pattern as the hand-written backend checkpoint field)."""
        btn = _browse_btn()
        f = filt or "*"
        btn.clicked.connect(
            lambda _=False, le=line_edit, ft=f: self._browse_path(le, ft))
        return btn

    def _browse_path(self, line_edit: QLineEdit, filt: str = "*"):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", f"Files ({filt});;All (*)")
        if path:
            line_edit.setText(path)

    def _add_field_row(self, f: UiField, layout: QVBoxLayout):
        control = self._make_control(f)
        row = QWidget()
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(0, 0, 0, 0)
        if f.widget == "check":
            rlay.addWidget(control)          # checkbox carries its own label
        else:
            rlay.addWidget(QLabel(f.label))
            rlay.addWidget(control, 1)
            if f.widget == "path":
                rlay.addWidget(self._path_browse_btn(control, f.file_filter))
        layout.addWidget(row)
        self._fields[f.dest] = (f, control)
        if f.advanced:
            self._advanced_rows.append(row)

    def _build_group(self, group: UiGroup, parent_layout: QVBoxLayout):
        box = QGroupBox(group.title)
        vbox = QVBoxLayout(box)

        if group.toggle_dest is not None:
            box.setCheckable(True)
            owner_w = group.toggle_owner_widget
            # initial checked state: unchecked when the default value == off.
            if owner_w is None:
                box.setChecked(False)     # bool owner defaults off
                self._toggles[group.toggle_dest] = {
                    "group": box, "widget": None, "inner": None,
                    "off": group.toggle_off_value}
            else:
                box.setChecked(False)
                if owner_w == "combo":
                    inner: QWidget = QComboBox()
                    inner.addItems([str(c) for c in (group.toggle_choices or ())])
                    idx = inner.findText(str(group.toggle_on_default))
                    if idx >= 0:
                        inner.setCurrentIndex(idx)
                else:  # line / path
                    inner = QLineEdit()
                    inner.setText("" if group.toggle_on_default is None
                                  else str(group.toggle_on_default))
                if group.toggle_tooltip:
                    inner.setToolTip(group.toggle_tooltip)
                row = QWidget()
                rlay = QHBoxLayout(row)
                rlay.setContentsMargins(0, 0, 0, 0)
                rlay.addWidget(QLabel(group.toggle_label))
                rlay.addWidget(inner, 1)
                if owner_w == "path":
                    rlay.addWidget(
                        self._path_browse_btn(inner, group.toggle_filter))
                vbox.addWidget(row)
                self._toggles[group.toggle_dest] = {
                    "group": box, "widget": owner_w, "inner": inner,
                    "off": group.toggle_off_value}

        for f in group.fields:
            self._add_field_row(f, vbox)
        for sub in group.subgroups:
            self._build_group(sub, vbox)

        parent_layout.addWidget(box)
        if group.advanced:
            self._advanced_rows.append(box)

    # -- show-advanced --------------------------------------------------------

    def set_show_advanced(self, show: bool):
        """Toggle visibility of the deep-tuning tier.  Does NOT affect
        namespace_values() -- hidden controls still report their values (the
        namespace census must be tier-independent)."""
        self._show_advanced = show
        for row in self._advanced_rows:
            row.setVisible(show)

    # -- namespace contract ---------------------------------------------------

    def _read_field(self, f: UiField, w: QWidget):
        if f.widget == "check":
            v = w.isChecked()
            return (not v) if f.inverted else v
        if f.widget in ("spin", "double"):
            return w.value()
        if f.widget == "combo":
            return w.currentText()
        # line
        return w.text().strip() or None

    def _read_toggle(self, tg: dict):
        group = tg["group"]
        if tg["widget"] is None:
            return group.isChecked()
        if not group.isChecked():
            return tg["off"]
        if tg["widget"] == "combo":
            return tg["inner"].currentText()
        # line
        return tg["inner"].text().strip() or None

    def namespace_values(self) -> dict:
        vals: dict = {}
        for dest, (f, w) in self._fields.items():
            vals[dest] = self._read_field(f, w)
        for dest, tg in self._toggles.items():
            vals[dest] = self._read_toggle(tg)
        return vals

    def _apply_field(self, f: UiField, w: QWidget, ns: Namespace):
        if f.dest == "min_call_gap":
            w.setValue(int(resolve_min_call_gap(ns)))
            return
        val = getattr(ns, f.dest, f.default)
        if f.widget == "check":
            checked = (not bool(val)) if f.inverted else bool(val)
            w.setChecked(checked)
        elif f.widget == "spin":
            w.setValue(int(val))
        elif f.widget == "double":
            w.setValue(float(val))
        elif f.widget == "combo":
            idx = w.findText(str(val).lower())
            if idx >= 0:
                w.setCurrentIndex(idx)
        elif f.widget in ("line", "path"):
            w.setText("" if val is None else str(val))

    def _apply_toggle(self, dest: str, tg: dict, ns: Namespace):
        group = tg["group"]
        off = tg["off"]
        if tg["widget"] is None:
            group.setChecked(bool(getattr(ns, dest, off)))
            return
        val = getattr(ns, dest, off)
        if tg["widget"] in ("line", "path"):
            text = val or ""
            group.setChecked(bool(text))
            tg["inner"].setText(str(text))
            return
        # combo owner (adaptive_ray / smooth_snap)
        if isinstance(val, bool):
            val = tg["inner"].itemText(0) if val else off
        val = str(val).lower()
        group.setChecked(val != str(off).lower())
        if val != str(off).lower():
            idx = tg["inner"].findText(val)
            if idx >= 0:
                tg["inner"].setCurrentIndex(idx)

    def apply_namespace(self, ns: Namespace):
        for _dest, (f, w) in self._fields.items():
            self._apply_field(f, w, ns)
        for dest, tg in self._toggles.items():
            self._apply_toggle(dest, tg, ns)

    def reset_defaults(self):
        self.apply_namespace(Namespace())
