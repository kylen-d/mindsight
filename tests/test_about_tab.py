"""About tab: hero cards, mkdocs down-conversion, in-app reader."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_render_mkdocs_markdown_admonitions_and_tabs():
    from mindsight.GUI.about_tab import render_mkdocs_markdown
    src = (
        '!!! note "What you need"\n'
        "    - A project folder.\n"
        "\n"
        '=== "macOS"\n'
        "    1. Download the zip.\n"
        "\n"
        "plain line\n"
    )
    out = render_mkdocs_markdown(src)
    assert "> **What you need**" in out
    assert "> - A project folder." in out
    assert "**macOS**" in out
    assert "1. Download the zip." in out       # dedented out of the tab
    assert "!!!" not in out and "===" not in out
    assert "plain line" in out                  # untouched content survives


def test_hero_lists_guide_cards_from_local_docs(qapp):
    # Running from the checkout: docs/ exists, so the curated guides that are
    # present must appear as cards (the tutorial at minimum).
    from mindsight.GUI.about_tab import AboutTab, docs_root
    assert docs_root() is not None, "test must run from a checkout"
    tab = AboutTab()
    assert tab._card_count >= 1
    assert tab._stack.currentIndex() == 0


def test_open_doc_renders_and_switches_to_reader(qapp, tmp_path):
    from mindsight.GUI.about_tab import AboutTab
    md = tmp_path / "guide.md"
    md.write_text("# Hello\n\nSome **bold** text.\n")
    tab = AboutTab()
    tab.open_doc(md, "Hello Guide")
    assert tab._stack.currentIndex() == 1
    assert tab._reader_title.text() == "Hello Guide"
    assert "Hello" in tab._browser.toPlainText()
    assert "bold" in tab._browser.toPlainText()
