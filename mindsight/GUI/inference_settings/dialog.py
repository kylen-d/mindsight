"""
inference_settings/dialog.py -- the Inference Settings dialog (UP2 Batch B).

A modal dialog rendering ``SETTINGS_SPEC`` (seven tabs) into a left tab list +
right scrollable pages, backed by the RunSettings store.  It edits a
``store.working_copy()`` namespace; OK/Apply commit it, Cancel discards.  The
header shows ``Preset: <source_label> (modified)`` with Reset-to-preset; the
footer offers Save-to-project-pipeline (when a project is open), Import/Export
YAML (reusing ``pipeline_dialog``), and a one-click Import-from-Gaze-Tuning
bridge.

Widget metadata is resolved per-dest from the live schema/FlagSpec via
``spec.field_meta`` -- never duplicated here.  Toggle groups reproduce the
schema off-value semantics (bool checkbox IS the owner; combo/path owners carry
an inner control shown when checked); the Gaze-LLE Blend group is a special
bool-style toggle over ``rf_gazelle_model`` whose value lives on the Models
tab.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..widgets import CollapsibleGroupBox, _browse_btn
from .spec import SETTINGS_SPEC, FieldMeta, SpecField, SpecGroup, field_meta
from .widgets import SliderValue

_HINT_AMBER = "#d08a1d"


def _caption(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #888; font-size: 11px;")
    return lbl


class InferenceSettingsDialog(QDialog):
    """Modal inference-settings editor over a RunSettings store."""

    def __init__(self, store, parent=None, *, gaze_tab=None,
                 project_pipeline_path: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Inference Settings")
        self.resize(720, 640)
        self._store = store
        self._gaze_tab = gaze_tab
        self._project_pipeline_path = project_pipeline_path
        self._ns = store.working_copy()

        # dest -> {"kind", "w", "field", "meta"}
        self._controls: dict[str, dict] = {}
        # owner dest -> {"box", "mode", "inner", "off"}
        self._toggles: dict[str, dict] = {}
        # phenomenon enable dests, for the "Enable all phenomena" bulk action.
        self._phenomena_toggles: list[str] = []
        self._blend_hint: QLabel | None = None

        self._build_ui()
        self._apply_ns_to_widgets(self._ns)
        self._refresh_header()

    # ── construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)

        # Header: active preset + modified state + reset.
        header = QHBoxLayout()
        self._header_lbl = QLabel()
        header.addWidget(self._header_lbl, 1)
        reset_btn = QPushButton("Reset to preset")
        reset_btn.clicked.connect(self._on_reset)
        header.addWidget(reset_btn)
        outer.addLayout(header)

        # Body: left tab list + right stacked scroll pages.
        body = QHBoxLayout()
        self._tab_list = QListWidget()
        self._tab_list.setMaximumWidth(190)
        self._stack = QStackedWidget()
        for tab in SETTINGS_SPEC:
            self._tab_list.addItem(tab.title)
            self._stack.addWidget(self._build_page(tab))
        self._tab_list.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._tab_list.setCurrentRow(0)
        body.addWidget(self._tab_list)
        body.addWidget(self._stack, 1)
        outer.addLayout(body, 1)

        # Footer: YAML + bridge actions, then the dialog buttons.
        footer = QHBoxLayout()
        self._save_proj_btn = QPushButton("Save to project pipeline...")
        self._save_proj_btn.setEnabled(self._project_pipeline_path is not None)
        self._save_proj_btn.clicked.connect(self._on_save_project)
        footer.addWidget(self._save_proj_btn)
        imp = QPushButton("Import YAML...")
        imp.clicked.connect(self._on_import_yaml)
        footer.addWidget(imp)
        exp = QPushButton("Export YAML...")
        exp.clicked.connect(self._on_export_yaml)
        footer.addWidget(exp)
        if self._gaze_tab is not None:
            gz = QPushButton("Import from Inference Tuning...")
            gz.clicked.connect(self._on_import_gaze)
            footer.addWidget(gz)
        footer.addStretch(1)
        outer.addLayout(footer)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).clicked.connect(
            self._on_ok)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self._on_apply)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).clicked.connect(
            self.reject)
        outer.addWidget(buttons)

    def _build_page(self, tab) -> QScrollArea:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(6, 6, 6, 6)
        if tab.caption:
            lay.addWidget(_caption(tab.caption))
        if tab.key == "phenomena":
            bulk = QHBoxLayout()
            on = QPushButton("Enable all phenomena")
            on.clicked.connect(lambda: self._set_all_phenomena(True))
            off = QPushButton("Disable all")
            off.clicked.connect(lambda: self._set_all_phenomena(False))
            bulk.addWidget(on)
            bulk.addWidget(off)
            bulk.addStretch(1)
            lay.addLayout(bulk)
        for group in tab.groups:
            lay.addWidget(self._build_group(group))
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    def _build_group(self, group: SpecGroup) -> QGroupBox:
        box = QGroupBox(group.title)
        vbox = QVBoxLayout(box)

        if group.toggle is not None:
            box.setCheckable(True)
            self._register_toggle(group, box, vbox)
        if group.caption:
            vbox.addWidget(_caption(group.caption))

        basic = [f for f in group.fields if f.tier != "A"]
        advanced = [f for f in group.fields if f.tier == "A"]
        for f in basic:
            vbox.addWidget(self._build_field_row(f))
        if advanced:
            adv = CollapsibleGroupBox("Advanced")
            inner = QWidget()
            ilay = QVBoxLayout(inner)
            ilay.setContentsMargins(0, 0, 0, 0)
            for f in advanced:
                ilay.addWidget(self._build_field_row(f))
            adv.set_content(inner)
            vbox.addWidget(adv)
        return box

    def _register_toggle(self, group: SpecGroup, box: QGroupBox,
                         vbox: QVBoxLayout):
        owner = group.toggle
        meta = field_meta(owner)
        box.setToolTip(group.toggle_desc)
        if meta.widget == "check" or owner in _BOOL_TOGGLE_OWNERS:
            # bool owner: the checkbox IS the owner (no inner widget).  The
            # blend group is bool-style too (its model value lives on Tab 1).
            mode = "blend" if owner == "rf_gazelle_model" else "bool"
            self._toggles[owner] = {"box": box, "mode": mode, "inner": None,
                                    "off": meta.off_value
                                    if meta.off_value is not None else False}
            if owner == "rf_gazelle_model":
                self._blend_hint = _caption("")
                self._blend_hint.setStyleSheet(
                    f"color: {_HINT_AMBER}; font-size: 11px;")
                self._blend_hint.setVisible(False)
                vbox.addWidget(self._blend_hint)
                box.toggled.connect(lambda _c: self._update_blend_hint())
        elif meta.widget == "combo":
            inner = QComboBox()
            labels = group.toggle_choice_labels or {}
            for raw in (meta.choices or ()):
                if raw == meta.off_value:
                    continue                    # 'off' == unchecked, not a choice
                inner.addItem(labels.get(raw, str(raw)), raw)
            row = self._labeled_row(group.toggle_label, inner)
            vbox.addWidget(row)
            self._toggles[owner] = {"box": box, "mode": "combo",
                                    "inner": inner, "off": meta.off_value}
        elif meta.widget == "path":
            inner = QLineEdit()
            browse = _browse_btn()
            browse.clicked.connect(
                lambda _c, le=inner: self._browse_into(le, "*.pt"))
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(QLabel(group.toggle_label))
            rl.addWidget(inner, 1)
            rl.addWidget(browse)
            vbox.addWidget(row)
            self._toggles[owner] = {"box": box, "mode": "path",
                                    "inner": inner, "off": meta.off_value}
        if group.key in _PHENOMENA_GROUP_KEYS:
            self._phenomena_toggles.append(owner)

    # -- field rows -----------------------------------------------------------

    def _build_field_row(self, f: SpecField) -> QWidget:
        meta = field_meta(f.dest)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        widget = self._make_control(f, meta)
        if meta.widget == "check":
            widget.setText(f.label)
            rl.addWidget(widget)
        else:
            lbl = QLabel(f.label)
            lbl.setMinimumWidth(180)
            rl.addWidget(lbl)
            rl.addWidget(widget, 1)
            if meta.widget == "path":
                browse = _browse_btn()
                filt = "*.vp.json" if f.dest == "vp_file" else "*"
                browse.clicked.connect(
                    lambda _c, le=widget, ft=filt: self._browse_into(le, ft))
                rl.addWidget(browse)
        tip = f.description or meta.tooltip
        if tip:
            widget.setToolTip(tip)
            row.setToolTip(tip)
        self._controls[f.dest] = {"kind": meta.widget, "w": widget,
                                  "field": f, "meta": meta}
        return row

    def _make_control(self, f: SpecField, meta: FieldMeta) -> QWidget:
        if meta.widget == "check":
            return QCheckBox()
        if meta.widget in ("spin", "double"):
            return SliderValue(
                is_int=(meta.widget == "spin"),
                minimum=meta.minimum, maximum=meta.maximum, step=meta.step,
                decimals=meta.decimals, end_labels=f.end_labels,
                tooltip=f.description or meta.tooltip)
        if meta.widget == "combo":
            combo = QComboBox()
            self._fill_combo(combo, meta, f.choice_labels)
            return combo
        # line / path
        return QLineEdit()

    @staticmethod
    def _fill_combo(combo: QComboBox, meta: FieldMeta,
                    choice_labels: dict | None):
        for raw in (meta.choices or ()):
            disp = (choice_labels or {}).get(raw, str(raw))
            combo.addItem(disp, raw)

    def _labeled_row(self, label: str, control: QWidget) -> QWidget:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel(label))
        rl.addWidget(control, 1)
        return row

    def _browse_into(self, line_edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", f"Files ({filt});;All (*)")
        if path:
            line_edit.setText(path)

    # ── namespace <-> widgets ────────────────────────────────────────────────

    def _apply_ns_to_widgets(self, ns):
        from mindsight.PostProcessing.RayForming.ray_config import (
            resolve_min_call_gap,
        )
        for dest, c in self._controls.items():
            kind = c["kind"]
            meta = c["meta"]
            if dest == "min_call_gap":
                val = getattr(ns, "min_call_gap", None)
                c["w"].setValue(int(resolve_min_call_gap(ns)) if val is None
                                else int(val))
                continue
            val = getattr(ns, dest, meta.default)
            w = c["w"]
            if kind == "check":
                checked = (not bool(val)) if c["field"].inverted else bool(val)
                w.setChecked(checked)
            elif kind in ("spin", "double"):
                if val is None:
                    val = meta.default if meta.default is not None else 0
                w.setValue(val)
            elif kind == "combo":
                idx = w.findData(val)
                if idx < 0 and val is not None:
                    idx = w.findData(str(val))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            else:  # line / path
                if isinstance(meta.default, list):
                    seq = val or []
                    w.setText(", ".join(str(x) for x in seq))
                else:
                    w.setText("" if val is None else str(val))
        for owner, tg in self._toggles.items():
            self._apply_toggle(owner, tg, ns)
        self._update_blend_hint()

    def _apply_toggle(self, owner: str, tg: dict, ns):
        meta = field_meta(owner)
        val = getattr(ns, owner, tg["off"])
        if tg["mode"] in ("bool", "blend"):
            tg["box"].setChecked(bool(val))
        elif tg["mode"] == "combo":
            checked = val is not None and val != tg["off"]
            tg["box"].setChecked(checked)
            if checked:
                idx = tg["inner"].findData(val)
                if idx >= 0:
                    tg["inner"].setCurrentIndex(idx)
            elif meta.choices:
                # default the inner to the first real (non-off) choice
                for i in range(tg["inner"].count()):
                    if tg["inner"].itemData(i) != tg["off"]:
                        tg["inner"].setCurrentIndex(i)
                        break
        elif tg["mode"] == "path":
            tg["box"].setChecked(bool(val))
            tg["inner"].setText("" if val in (None, "") else str(val))

    def _read_widgets_into_ns(self, ns):
        for dest, c in self._controls.items():
            kind = c["kind"]
            meta = c["meta"]
            w = c["w"]
            if kind == "check":
                v = w.isChecked()
                setattr(ns, dest, (not v) if c["field"].inverted else v)
            elif kind in ("spin", "double"):
                setattr(ns, dest, w.value())
            elif kind == "combo":
                setattr(ns, dest, w.currentData())
            else:  # line / path
                text = w.text().strip()
                if isinstance(meta.default, list):
                    setattr(ns, dest,
                            [x.strip() for x in text.split(",") if x.strip()])
                else:
                    setattr(ns, dest, text or None)
        # Toggles override their owner dest last (may re-set rf_gazelle_model).
        for owner, tg in self._toggles.items():
            if tg["mode"] == "bool":
                setattr(ns, owner, tg["box"].isChecked())
            elif tg["mode"] == "combo":
                setattr(ns, owner, tg["inner"].currentData()
                        if tg["box"].isChecked() else tg["off"])
            elif tg["mode"] == "path":
                setattr(ns, owner, (tg["inner"].text().strip() or None)
                        if tg["box"].isChecked() else tg["off"])
            elif tg["mode"] == "blend":
                self._read_blend(ns, tg)
        return ns

    def _read_blend(self, ns, tg: dict):
        """Blend toggle owns ``rf_gazelle_model`` presence; the model VALUE
        comes from the Models tab field (already read into ns above)."""
        if not tg["box"].isChecked():
            ns.rf_gazelle_model = None
            return
        current = getattr(ns, "rf_gazelle_model", None)
        if current:
            return
        resolved = self._resolve_default_gazelle(ns)
        ns.rf_gazelle_model = resolved or ""

    def _resolve_default_gazelle(self, ns) -> str | None:
        """Bare checkpoint filename for the active variant if it exists under
        the shared Weights root, else None (preflight is the hard gate)."""
        from mindsight import constants
        name = getattr(ns, "rf_gazelle_name", None) or "gazelle_dinov2_vitb14"
        candidate = constants.PROJECT_ROOT / "Weights" / "Gazelle" / f"{name}.pt"
        return f"{name}.pt" if candidate.is_file() else None

    def _update_blend_hint(self):
        if self._blend_hint is None:
            return
        tg = self._toggles.get("rf_gazelle_model")
        if tg is None:
            return
        model = self._controls.get("rf_gazelle_model", {}).get("w")
        has_model = bool(model.text().strip()) if model is not None else False
        if tg["box"].isChecked() and not has_model \
                and not self._resolve_default_gazelle(self._ns):
            self._blend_hint.setText(
                "Select a Gaze-LLE model on the Models & Device tab.")
            self._blend_hint.setVisible(True)
        else:
            self._blend_hint.setVisible(False)

    # ── bulk phenomena action ────────────────────────────────────────────────

    def _set_all_phenomena(self, on: bool):
        for owner in self._phenomena_toggles:
            tg = self._toggles.get(owner)
            if tg is not None:
                tg["box"].setChecked(on)

    # ── header ────────────────────────────────────────────────────────────────

    def _refresh_header(self):
        label = self._store.source_label()
        suffix = " (modified)" if self._store.is_modified() else ""
        self._header_lbl.setText(f"Preset: {label}{suffix}")

    # ── button handlers ──────────────────────────────────────────────────────

    def _commit(self):
        self._read_widgets_into_ns(self._ns)
        self._store.commit(self._ns)
        self._refresh_header()

    def _on_ok(self):
        self._commit()
        self.accept()

    def _on_apply(self):
        self._commit()
        # Re-seed the working copy from the committed state so blend resolution
        # / coercion is reflected back into the widgets.
        self._ns = self._store.working_copy()
        self._apply_ns_to_widgets(self._ns)

    def _on_reset(self):
        self._store.reset_to_preset()
        self._ns = self._store.working_copy()
        self._apply_ns_to_widgets(self._ns)
        self._refresh_header()

    def _on_import_yaml(self):
        from ..pipeline_dialog import import_pipeline
        ns = import_pipeline(self)
        if ns is None:
            return
        # import_pipeline returns a Namespace of only the loaded keys; overlay
        # them onto the working copy (the store commits on OK/Apply).
        for k, v in vars(ns).items():
            setattr(self._ns, k, v)
        self._apply_ns_to_widgets(self._ns)

    def _on_export_yaml(self):
        from ..pipeline_dialog import export_pipeline
        self._read_widgets_into_ns(self._ns)
        export_pipeline(self, self._ns)

    def _on_save_project(self):
        if self._project_pipeline_path is None:
            return
        import yaml

        from ..pipeline_dialog import _namespace_to_yaml_dict
        self._read_widgets_into_ns(self._ns)
        target = self._project_pipeline_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.dump(_namespace_to_yaml_dict(self._ns),
                                        default_flow_style=False,
                                        sort_keys=False))
            self._store.apply_yaml(str(target), source_label="project pipeline")
            self._ns = self._store.working_copy()
            self._apply_ns_to_widgets(self._ns)
            self._refresh_header()
            QMessageBox.information(self, "Saved",
                                    f"Wrote {target.name} and applied it.")
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Save error", str(exc))

    def _on_import_gaze(self):
        if self._gaze_tab is None:
            return
        gns = self._gaze_tab._build_namespace()
        for k, v in vars(gns).items():
            setattr(self._ns, k, v)
        self._apply_ns_to_widgets(self._ns)
        # Mark modified: this is a deliberate divergence from the loaded source.
        self._header_lbl.setText(
            f"Preset: {self._store.source_label()} (modified)")


# Toggle owners whose checkbox IS the owner even though the schema/FlagSpec
# type is not a plain bool store_true (rf_gazelle_model is a path dest used as a
# bool-style blend enable; its value lives on the Models tab).
_BOOL_TOGGLE_OWNERS = frozenset({"rf_gazelle_model"})

# Phenomena group keys whose toggle owners participate in "Enable all phenomena".
_PHENOMENA_GROUP_KEYS = frozenset({
    "joint_attention", "mutual_gaze", "social_ref", "gaze_follow",
    "gaze_leader", "gaze_aversion", "scanpath", "attn_span", "eye_movement",
    "novel_salience", "pupillometry",
})
