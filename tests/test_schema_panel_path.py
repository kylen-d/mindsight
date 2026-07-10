"""Path-widget (browse button) support in the schema panel (B4b).

The Gaze-LLE Blend group's Model field (dest ``rf_gazelle_model``) is a
``path`` widget: a QLineEdit paired with a shared Browse button.  It must add
the browse affordance WITHOUT changing the namespace round-trip -- the value
stays a plain string, '' -> None, exactly as the old bare line-edit owner did.
Offscreen Qt only.
"""

import os
from argparse import Namespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

_CKPT = "Weights/Gazelle/gazelle_dinov2_vitb14.pt"


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_rf_gazelle_model_row_has_browse_button(qapp):
    from PyQt6.QtWidgets import QLineEdit, QPushButton
    from mindsight.GUI.schema_panel import SchemaPanel

    panel = SchemaPanel()
    tg = panel._toggles["rf_gazelle_model"]
    inner = tg["inner"]
    assert isinstance(inner, QLineEdit)
    # The Browse button lives in the same row as the line edit.
    buttons = inner.parent().findChildren(QPushButton)
    assert len(buttons) == 1


def test_rf_gazelle_model_path_roundtrips(qapp):
    from mindsight.GUI.schema_panel import SchemaPanel

    panel = SchemaPanel()
    tg = panel._toggles["rf_gazelle_model"]

    # A typed checkpoint path (group enabled) surfaces as a plain string.
    tg["group"].setChecked(True)
    tg["inner"].setText(_CKPT)
    assert panel.namespace_values()["rf_gazelle_model"] == _CKPT

    # apply_namespace mirrors a value back into the widget + checked state.
    panel.apply_namespace(Namespace(rf_gazelle_model="a/b/c.pt"))
    assert tg["inner"].text() == "a/b/c.pt"
    assert tg["group"].isChecked()
    assert panel.namespace_values()["rf_gazelle_model"] == "a/b/c.pt"

    # Empty / None keeps the T10 off-value semantics ('' -> None, unchecked).
    panel.apply_namespace(Namespace(rf_gazelle_model=None))
    assert not tg["group"].isChecked()
    assert panel.namespace_values()["rf_gazelle_model"] is None
