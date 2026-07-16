"""
about_tab.py -- the About tab: program identity + in-app documentation reader.

Eyes-on session request (2026-07-11): a landing page with the logo, version,
and tagline, plus "guide cards" that open rendered tutorials INSIDE the app.
Two stacked pages: a hero landing and a full-width markdown reader with a back
button.

Docs come from the repo's ``docs/`` tree when running from a checkout, or from
the copy bundled into the wheel as package data (``mindsight/_bundled/docs``).
When neither is present the cards give way to a button that opens the hosted
documentation site.  The reader understands enough of the mkdocs dialect
(admonitions, content tabs) to render the committed pages faithfully through
Qt's GitHub-flavored markdown support.
"""

from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QFont, QPixmap, QTextDocument
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

#: Published documentation site (mirrors README / docs config).
DOCS_URL = "https://kylen-d.github.io/mindsight-docs/"

#: Curated in-app guides: (card title, card subtitle, path relative to docs/).
#: Cards only show for files that exist, so this list can stay ahead of what a
#: given install actually ships.
GUIDES = (
    ("Run a Study",
     "Start-to-finish walkthrough for research assistants",
     "studies/run-a-study-tutorial.md"),
    ("Analyze Footage",
     "Project, video-file, and camera runs",
     "guides/analyze-footage.md"),
    ("Projects & Sessions",
     "Build projects; plan and record sessions",
     "guides/projects-and-sessions.md"),
    ("Visual Prompts",
     "Teach the detector your study's objects",
     "guides/visual-prompts.md"),
    ("Crop & Adjust",
     "Re-frame footage before analysis",
     "guides/crop-and-adjust.md"),
    ("Settings vs Tuning",
     "The dialog governs runs; the tab is a sandbox",
     "guides/inference-settings-and-tuning.md"),
    ("Inference Settings",
     "Every setting in the dialog, tab by tab",
     "reference/inference-settings.md"),
    ("Where Things Live",
     "Folders, outputs, and weights on disk",
     "guides/where-things-live.md"),
    ("About & Theming",
     "This tab, the reader, and the theme control",
     "guides/about-and-theming.md"),
    ("What's New",
     "Release changelog",
     "changelog.md"),
)

_TAGLINE = "Gaze-based social attention analysis for developmental research."


def repo_root() -> Path:
    """The checkout root (parent of the ``mindsight`` package)."""
    return Path(__file__).resolve().parents[2]


def docs_root() -> Path | None:
    """The local docs tree: checkout copy, else the wheel's bundled copy."""
    d = repo_root() / "docs"
    if d.is_dir():
        return d
    from mindsight.resources import bundled_path
    b = bundled_path("docs")
    return b if b is not None and b.is_dir() else None


def render_mkdocs_markdown(text: str) -> str:
    """Down-convert mkdocs-material syntax to plain GitHub markdown.

    Qt's markdown engine renders GitHub dialect; the committed docs also use
    mkdocs admonitions (``!!! note "Title"``) and content tabs
    (``=== "macOS"``).  Both become emphasized blocks their indented bodies
    dedent into, which reads correctly even if it loses the chrome.

    Constructs Qt cannot render at all are degraded rather than shown raw:
    ``???``/``???+`` collapsibles render like admonitions (always expanded),
    ``mermaid`` fences become a pointer to the documentation site, and
    grid-card ``<div>`` wrappers are stripped (their inner markdown stays).
    """
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        adm = re.match(r'^(?:!!!|\?\?\?\+?)\s+\w+(?:\s+"([^"]*)")?\s*$', line)
        tab = re.match(r'^===\s+"([^"]*)"\s*$', line)
        if adm or tab:
            title = (adm or tab).group(1) or "Note"
            out.append(f"> **{title}**" if adm else f"**{title}**")
            out.append(">" if adm else "")
            i += 1
            while i < len(lines):
                body = lines[i]
                if body.strip() == "":
                    out.append(">" if adm else "")
                elif body.startswith("    "):
                    out.append(("> " + body[4:]) if adm else body[4:])
                else:
                    break
                i += 1
            continue
        if re.match(r'^```\s*mermaid\s*$', line):
            out.append("> **Diagram** -- rendered on the documentation site.")
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                i += 1
            i += 1                      # skip the closing fence
            continue
        if re.match(r'^\s*</?div[^>]*>\s*$', line):
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


class AboutTab(QWidget):
    """Hero landing (logo / version / guide cards) + full-width doc reader."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack = QStackedWidget()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._stack)
        self._stack.addWidget(self._build_hero())    # page 0
        self._stack.addWidget(self._build_reader())  # page 1

    # ── page 0: hero ─────────────────────────────────────────────────────────

    def _build_hero(self) -> QWidget:
        from mindsight import __version__

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.addStretch(2)

        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pm = self._logo_pixmap()
        if pm is not None:
            logo.setPixmap(pm)
        lay.addWidget(logo)

        name = QLabel("MindSight")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont()
        f.setPointSize(26)
        f.setBold(True)
        name.setFont(f)
        lay.addWidget(name)

        version = QLabel(f"version {__version__}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: #888;")
        lay.addWidget(version)

        tagline = QLabel(_TAGLINE)
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tagline.setStyleSheet("color: #aaa; font-style: italic;")
        lay.addWidget(tagline)
        lay.addSpacing(24)

        cards = QHBoxLayout()
        cards.addStretch(1)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        root = docs_root()
        self._card_count = 0
        if root is not None:
            for title, subtitle, rel in GUIDES:
                path = root / rel
                if not path.is_file():
                    continue
                btn = QPushButton(f"{title}\n{subtitle}")
                btn.setMinimumSize(220, 72)
                btn.setStyleSheet(
                    "QPushButton {text-align: center; padding: 10px;"
                    " font-size: 13px;}")
                btn.setToolTip(f"Open '{title}' in the reader")
                btn.clicked.connect(
                    lambda _c, p=path, t=title: self.open_doc(p, t))
                row, col = divmod(self._card_count, 3)
                grid.addWidget(btn, row, col)
                self._card_count += 1
        if self._card_count:
            cards.addLayout(grid)
        if self._card_count == 0:
            # No checkout docs and no bundled copy (wheel built without the
            # resource-staging step): the hosted site is the reader.
            site = QPushButton("Open the documentation site")
            site.setMinimumHeight(40)
            site.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
            cards.addWidget(site)
        cards.addStretch(1)
        lay.addLayout(cards)
        lay.addSpacing(16)

        links = QLabel(
            f'<a href="{DOCS_URL}">Documentation site</a> &nbsp;·&nbsp; '
            f'<a href="https://github.com/kylen-d/MindSight">GitHub</a>')
        links.setAlignment(Qt.AlignmentFlag.AlignCenter)
        links.setOpenExternalLinks(True)
        lay.addWidget(links)
        lay.addStretch(3)
        return page

    def _logo_pixmap(self) -> QPixmap | None:
        for cand in (repo_root() / "mindsightlogo.png",
                     repo_root() / "assets" / "mindsight_icon.png"):
            if cand.exists():
                pm = QPixmap(str(cand))
                if not pm.isNull():
                    return pm.scaledToHeight(
                        160, Qt.TransformationMode.SmoothTransformation)
        return None

    # ── page 1: reader ───────────────────────────────────────────────────────

    def _build_reader(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        bar = QHBoxLayout()
        back = QPushButton("‹  About")
        back.setToolTip("Back to the About page")
        back.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        bar.addWidget(back)
        self._reader_title = QLabel("")
        self._reader_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        bar.addWidget(self._reader_title, 1)
        bar.addStretch(0)
        bar_host = QWidget()
        bar_host.setLayout(bar)
        lay.addWidget(bar_host)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(False)
        self._browser.setOpenLinks(False)
        self._browser.anchorClicked.connect(self._follow_link)
        lay.addWidget(self._browser, 1)
        return page

    def open_doc(self, path: Path, title: str | None = None):
        """Render *path* (markdown) in the reader and switch to it."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            QDesktopServices.openUrl(QUrl(DOCS_URL))
            return
        self._current_doc = Path(path)
        self._reader_title.setText(title or Path(path).stem)
        doc = self._browser.document()
        # Relative image/link paths resolve against the doc's own folder.
        doc.setBaseUrl(QUrl.fromLocalFile(str(Path(path).parent) + "/"))
        doc.setMarkdown(render_mkdocs_markdown(text),
                        QTextDocument.MarkdownFeature.MarkdownDialectGitHub)
        self._browser.verticalScrollBar().setValue(0)
        self._stack.setCurrentIndex(1)

    def _follow_link(self, url: QUrl):
        """Internal .md links open in the reader; everything else, outside."""
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
            return
        # Same-page anchor (href="#section"): scroll, never leave the reader.
        if not url.path() and url.hasFragment():
            self._browser.scrollToAnchor(url.fragment())
            return
        if url.isLocalFile():
            target = Path(url.toLocalFile())
            # Markdown links may carry an anchor; the path part decides.
            if target.suffix == ".md" and target.exists():
                self.open_doc(target)
                return
            # Anchor resolved against baseUrl lands on the doc's folder:
            # still a same-page anchor, not something to hand to Finder.
            if url.hasFragment() and target.is_dir():
                self._browser.scrollToAnchor(url.fragment())
                return
            if target.is_file():
                QDesktopServices.openUrl(url)
                return
        # Unresolvable relative link (docs page not shipped locally).
        QDesktopServices.openUrl(QUrl(DOCS_URL))
