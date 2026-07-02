#!/usr/bin/env python3
"""Build a MindSight SP4.0 release zip (Windows or macOS).

The zip is assembled from ``git archive`` (the committed tree only) so that no
untracked working-tree junk -- weights, outputs, local reference material, dev
virtualenv paths -- can ever leak into a release. Layout (D7):

    MindSight-SP4.0-win.zip
    |-- Install-MindSight.bat       double-click entry point
    |-- INSTALL-WINDOWS.md          one-page install / troubleshooting guide
    `-- app/                        git-archive of the repo (source tree)

    MindSight-SP4.0-mac.zip
    |-- Install-MindSight.command   right-click > Open entry point (exec bit set)
    |-- INSTALL-MACOS.md            one-page install / troubleshooting guide
    `-- app/                        git-archive of the repo (source tree)

After building, a content census runs over the finished zip and fails the
build if any forbidden pattern is present (weights, .git, localref, dev-venv
paths). Absence is the pass.

Usage:
    python installer/make_release_zip.py [--platform win|mac] [--out DIR] [--ref GITREF]
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALLER_DIR = REPO_ROOT / "installer"

# Per-platform zip name and the files copied to the zip root (outside app/).
# The .bat is rewritten with CRLF line endings so cmd.exe parses labels/goto
# reliably on Windows; the .command keeps LF and is given a Unix exec bit so
# the double-click / right-click > Open flow works after extraction.
PLATFORMS = {
    "win": {
        "zip_name": "MindSight-SP4.0-win.zip",
        "root_files": ["Install-MindSight.bat", "INSTALL-WINDOWS.md"],
    },
    "mac": {
        "zip_name": "MindSight-SP4.0-mac.zip",
        "root_files": ["Install-MindSight.command", "INSTALL-MACOS.md"],
    },
}

# Unix mode bits for the executable launcher script inside the mac zip.
EXEC_EXTERNAL_ATTR = (0o100755 << 16)

# --- Census rules ----------------------------------------------------------
# A path fails if any of its components matches one of these names exactly, or
# if it ends with one of these extensions. Component matching avoids false
# positives on legitimate source (e.g. mindsight/outputs/ is source; a
# top-level Outputs/ runtime dir is not).
FORBIDDEN_PATH_COMPONENTS = {"Weights", "Outputs", "localref", ".git"}
FORBIDDEN_EXTENSIONS = {".pt", ".onnx", ".ts", ".pkl"}

# Text tokens that must not appear in ANY file's contents. These mark dev-venv
# leakage: the dev interpreter is Python 3.14 under ~/claudeyolo / miniconda,
# and the installer targets a uv-managed Python 3.12 that must never reference
# it. NOTE: we do NOT forbid "cp314"/"3.14" -- the committed uv.lock is a
# universal lock that legitimately lists wheels for every Python version,
# including 3.14; blocking that would be wrong. Instead we positively assert
# below that 3.12 wheels (cp312) are present, which is what the installer uses.
FORBIDDEN_CONTENT_TOKENS = ["claudeyolo", "miniconda"]

# The builder's own source (archived into app/installer/) necessarily contains
# the token strings above as literals; exempt it from the content scan.
CONTENT_SCAN_EXEMPT_BASENAMES = {"make_release_zip.py"}


def run(cmd: list[str]) -> bytes:
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE).stdout


def build_zip(out_dir: Path, ref: str, platform: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = PLATFORMS[platform]
    zip_path = out_dir / spec["zip_name"]

    # Snapshot the tracked tree at `ref` as a tar stream (respects the index --
    # gitignored weights/outputs/localref/test_data are absent by construction).
    tar_bytes = run(["git", "archive", "--format=tar", ref])

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                data = tf.extractfile(member).read()
                zf.writestr(f"app/{member.name}", data)

        for name in spec["root_files"]:
            src = INSTALLER_DIR / name
            if not src.exists():
                raise SystemExit(f"ERROR: missing required file {src}")
            data = src.read_bytes()
            if name.endswith(".bat"):
                # Normalise to CRLF for cmd.exe.
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if name.endswith(".command"):
                # Keep LF; carry the Unix exec bit so extraction preserves +x.
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = EXEC_EXTERNAL_ATTR
                zf.writestr(info, data)
            else:
                zf.writestr(name, data)

    return zip_path


def census(zip_path: Path) -> int:
    """Print a content census. Return the number of violations (0 == pass)."""
    violations: list[str] = []
    total_bytes = 0
    file_count = 0
    top_level: dict[str, int] = {}
    cp312_hits = 0
    cp314_hits = 0

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            file_count += 1
            total_bytes += info.file_size
            top = name.split("/", 1)[0]
            top_level[top] = top_level.get(top, 0) + 1

            parts = Path(name).parts
            if any(p in FORBIDDEN_PATH_COMPONENTS for p in parts):
                violations.append(f"forbidden path component: {name}")
            if Path(name).suffix in FORBIDDEN_EXTENSIONS:
                violations.append(f"forbidden extension: {name}")

            # Content scan of text-ish files.
            data = zf.read(name)
            if b"\x00" in data[:8192]:
                continue  # binary; skip content grep
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            cp312_hits += text.count("cp312")
            cp314_hits += text.count("cp314")
            if Path(name).name in CONTENT_SCAN_EXEMPT_BASENAMES:
                continue
            for token in FORBIDDEN_CONTENT_TOKENS:
                if token in text:
                    violations.append(f"forbidden token {token!r} in: {name}")

    # Positive assertion: the universal lock must carry Python 3.12 wheels
    # (that is what the installer resolves against). Absence would mean the
    # lock cannot install on the 3.12 target.
    if cp312_hits == 0:
        violations.append("no cp312 wheels found in the tree (uv.lock lost its 3.12 target?)")

    print("=" * 60)
    print(f"CENSUS for {zip_path.name}")
    print("=" * 60)
    print(f"  files: {file_count}")
    print(f"  total uncompressed size: {total_bytes / 1024 / 1024:.2f} MiB")
    print(f"  on-disk zip size: {zip_path.stat().st_size / 1024 / 1024:.2f} MiB")
    print("  top-level entries (file counts):")
    for top in sorted(top_level):
        print(f"    {top}: {top_level[top]}")
    print("  checked path components:", ", ".join(sorted(FORBIDDEN_PATH_COMPONENTS)))
    print("  checked extensions:", ", ".join(sorted(FORBIDDEN_EXTENSIONS)))
    print("  checked content tokens:", ", ".join(FORBIDDEN_CONTENT_TOKENS))
    print(f"  info: wheel-tag refs -- cp312={cp312_hits} (required present), cp314={cp314_hits} "
          "(universal lock, informational)")
    if violations:
        print(f"  RESULT: FAIL ({len(violations)} violation(s))")
        for v in violations:
            print(f"    - {v}")
    else:
        print("  RESULT: PASS (zero forbidden hits)")
    print("=" * 60)
    return len(violations)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a MindSight SP4.0 release zip.")
    parser.add_argument("--platform", choices=sorted(PLATFORMS), default="win",
                        help="Target platform for the zip (default: win).")
    parser.add_argument("--out", default=str(REPO_ROOT / "dist"),
                        help="Output directory for the zip (default: ./dist).")
    parser.add_argument("--ref", default="HEAD",
                        help="Git ref to archive (default: HEAD).")
    args = parser.parse_args(argv)

    zip_path = build_zip(Path(args.out).expanduser().resolve(), args.ref, args.platform)
    print(f"Built {zip_path}")
    violations = census(zip_path)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
