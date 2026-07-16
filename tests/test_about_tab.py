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


def test_render_mkdocs_markdown_hardened_constructs():
    """v1.1 W0.6: collapsibles render like admonitions; mermaid fences and
    grid-card div wrappers degrade instead of leaking raw syntax."""
    from mindsight.GUI.about_tab import render_mkdocs_markdown
    src = (
        '??? tip "Fold me"\n'
        "    hidden body\n"
        "\n"
        '```mermaid\n'
        "graph TD; A-->B;\n"
        "```\n"
        '<div class="grid cards" markdown>\n'
        "- a card entry\n"
        "</div>\n"
    )
    out = render_mkdocs_markdown(src)
    assert "> **Fold me**" in out
    assert "> hidden body" in out
    assert "mermaid" not in out and "A-->B" not in out
    assert "documentation site" in out           # diagram pointer left behind
    assert "<div" not in out and "</div>" not in out
    assert "- a card entry" in out               # inner markdown survives


def test_hero_lists_guide_cards_from_local_docs(qapp):
    # Running from the checkout: docs/ exists, so every curated guide must
    # appear as a card (all GUIDES entries ship in the checkout).
    from mindsight.GUI.about_tab import GUIDES, AboutTab, docs_root
    assert docs_root() is not None, "test must run from a checkout"
    tab = AboutTab()
    assert tab._card_count == len(GUIDES)
    assert tab._stack.currentIndex() == 0


def test_all_guides_pages_exist_and_render_clean():
    """Every GUIDES page exists in the checkout and down-converts without
    leaking raw mkdocs syntax into the reader."""
    from mindsight.GUI.about_tab import GUIDES, docs_root, render_mkdocs_markdown
    root = docs_root()
    assert root is not None
    for _title, _sub, rel in GUIDES:
        page = root / rel
        assert page.is_file(), f"GUIDES names a missing page: {rel}"
        out = render_mkdocs_markdown(page.read_text(encoding="utf-8"))
        for token in ("\n!!!", "\n===", "\n???", "```mermaid", "<div"):
            assert token not in out, f"raw mkdocs syntax leaks in {rel}: {token!r}"


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
