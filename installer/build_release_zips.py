#!/usr/bin/env python3
"""Build the MindSight release-mode installer zips (Windows + macOS).

Unlike ``make_release_zip.py`` (which bundles a full ``app/`` source tree for a
local/editable install), a *release-mode* zip carries ONLY the launcher script
and its one-page install guide. There is no ``app/`` tree, so the launcher runs
in release mode: it installs MindSight from the wheel published on the GitHub
Release and fetches the weights manifest and the ``pipeline_known_good.yaml``
preset from that same release into ``$APP_DIR`` (see the release-mode branch of
Install-MindSight.command / .bat).

Layout:

    MindSight-1.0.0-indev-mac.zip
    |-- Install-MindSight.command   right-click > Open entry point (exec bit set)
    `-- INSTALL-MACOS.md

    MindSight-1.0.0-indev-win.zip
    |-- Install-MindSight.bat       double-click entry point (CRLF)
    `-- INSTALL-WINDOWS.md

The release URLs (wheel, manifest, preset) are baked into the committed launcher
scripts as overridable defaults, so no build-time substitution is needed; this
builder just packages the committed installer files verbatim. A content scan
fails the build if any forbidden dev-venv token leaks in.

Usage:
    python installer/build_release_zips.py [--platform win|mac|all] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALLER_DIR = REPO_ROOT / "installer"

# Unix mode bits for the executable launcher inside the mac zip.
EXEC_EXTERNAL_ATTR = (0o100755 << 16)


def _package_version() -> str:
    """The version from mindsight/__init__.py (regex, no heavy import)."""
    import re
    text = (REPO_ROOT / "mindsight" / "__init__.py").read_text()
    m = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    if not m:
        raise SystemExit("cannot find __version__ in mindsight/__init__.py")
    return m.group(1)


VERSION = _package_version()

PLATFORMS = {
    "win": {
        "zip_name": f"MindSight-{VERSION}-win.zip",
        "launcher": "Install-MindSight.bat",
        "guide": "INSTALL-WINDOWS.md",
    },
    "mac": {
        "zip_name": f"MindSight-{VERSION}-mac.zip",
        "launcher": "Install-MindSight.command",
        "guide": "INSTALL-MACOS.md",
    },
}

FORBIDDEN_CONTENT_TOKENS = ["claudeyolo", "miniconda"]


def build_zip(out_dir: Path, platform: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = PLATFORMS[platform]
    zip_path = out_dir / spec["zip_name"]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (spec["launcher"], spec["guide"]):
            src = INSTALLER_DIR / name
            if not src.exists():
                raise SystemExit(f"ERROR: missing required file {src}")
            data = src.read_bytes()
            if name.endswith(".bat"):
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                zf.writestr(name, data)
            elif name.endswith(".command"):
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = EXEC_EXTERNAL_ATTR
                zf.writestr(info, data)
            else:
                zf.writestr(name, data)
    return zip_path


def census(zip_path: Path) -> int:
    """Print a content census; return the number of violations (0 == pass)."""
    violations: list[str] = []
    has_release_fetch = False
    with zipfile.ZipFile(zip_path) as zf:
        names = [i.filename for i in zf.infolist() if not i.filename.endswith("/")]
        for name in names:
            if name.startswith("app/") or name == "app":
                violations.append(f"release zip must not bundle an app/ tree: {name}")
            data = zf.read(name)
            if b"\x00" in data[:8192]:
                continue
            text = data.decode("utf-8", errors="ignore")
            for token in FORBIDDEN_CONTENT_TOKENS:
                if token in text:
                    violations.append(f"forbidden token {token!r} in: {name}")
            if "pipeline_known_good.yaml" in text and "weights_manifest.json" in text:
                has_release_fetch = True
    if not has_release_fetch:
        violations.append(
            "no launcher fetches both the weights manifest and the preset "
            "(release-mode manifest/preset fetch missing?)")

    print("=" * 60)
    print(f"CENSUS for {zip_path.name}")
    print("=" * 60)
    print(f"  files: {', '.join(names)}")
    print(f"  on-disk zip size: {zip_path.stat().st_size} bytes")
    print("  checked content tokens:", ", ".join(FORBIDDEN_CONTENT_TOKENS))
    print(f"  release manifest+preset fetch present: {has_release_fetch}")
    if violations:
        print(f"  RESULT: FAIL ({len(violations)} violation(s))")
        for v in violations:
            print(f"    - {v}")
    else:
        print("  RESULT: PASS")
    print("=" * 60)
    return len(violations)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build MindSight release-mode installer zips.")
    parser.add_argument("--platform", choices=[*sorted(PLATFORMS), "all"], default="all",
                        help="Target platform (default: all).")
    parser.add_argument("--out", default=str(REPO_ROOT / "dist"),
                        help="Output directory for the zips (default: ./dist).")
    args = parser.parse_args(argv)

    plats = sorted(PLATFORMS) if args.platform == "all" else [args.platform]
    out_dir = Path(args.out).expanduser().resolve()
    violations = 0
    for plat in plats:
        zip_path = build_zip(out_dir, plat)
        print(f"Built {zip_path}")
        violations += census(zip_path)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
