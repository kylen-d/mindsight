"""
weights.py — Weight-file resolution, checksummed manifest, and verified downloads.

Two responsibilities live here:

1. :func:`resolve_weight` -- every backend resolves model paths through it so
   bare filenames (e.g. ``"yolov8n.pt"``) land in ``Weights/{backend}/`` while
   absolute/relative paths with directory components are respected as-is.
2. The **weights manifest** (``weights_manifest.json`` at the repo root): the
   single source of truth for each downloadable weight's upstream URL, sha256,
   size, license, and required flag.  :func:`load_manifest`,
   :func:`find_entry`, :func:`verify`, and :func:`download` implement
   checksum-verified, plain-English downloads used by both the ``mindsight-weights``
   console script (installer first-run fetch) and the preflight checksum check.

:func:`sha256_file` is the ONE implementation of the file-hashing loop in the
codebase; ``mindsight.outputs.provenance`` imports it (wrapping it in a
(path, size, mtime) cache) so run manifests and weight verification agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from mindsight.constants import PROJECT_ROOT

WEIGHTS_ROOT = PROJECT_ROOT / "Weights"
MANIFEST_PATH = PROJECT_ROOT / "weights_manifest.json"

# Manifest sources: a real GitHub-release asset (url + sha256), or an asset
# Ultralytics auto-fetches on first use (url/sha256 null + an explanatory note).
SOURCE_GITHUB = "github-release"
SOURCE_ULTRALYTICS_AUTO = "ultralytics-auto"


class WeightsError(Exception):
    """A weight download or verification failed (plain-English ``str``)."""


# ══════════════════════════════════════════════════════════════════════════════
# Path resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_weight(backend: str, filename: str) -> Path:
    """Resolve a weight file path, preferring ``Weights/{backend}/``.

    Parameters
    ----------
    backend : str
        Subdirectory name under ``Weights/`` (e.g. ``"YOLO"``, ``"MGaze"``).
    filename : str
        Model filename or path.  Bare filenames are resolved against
        ``Weights/{backend}/``.  Paths with directory components (relative
        or absolute) are returned unchanged.

    Returns
    -------
    Path
        Resolved path to the weight file.
    """
    p = Path(filename)
    if p.is_absolute() or p.parent != Path("."):
        return p
    target_dir = WEIGHTS_ROOT / backend
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / p.name


# ══════════════════════════════════════════════════════════════════════════════
# Hashing (single source; provenance wraps this in a cache)
# ══════════════════════════════════════════════════════════════════════════════

def sha256_file(path) -> str:
    """Streaming sha256 hex digest of *path* (1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Manifest load + lookup
# ══════════════════════════════════════════════════════════════════════════════

def load_manifest(path=None) -> dict:
    """Load and lightly validate ``weights_manifest.json``.

    Raises :class:`WeightsError` when the file is missing or malformed.
    """
    path = Path(path) if path is not None else MANIFEST_PATH
    if not path.exists():
        raise WeightsError(
            f"weights manifest not found at {path} -- the install is incomplete")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WeightsError(f"weights manifest {path.name} is unreadable: {exc}")
    if not isinstance(data, dict) or "weights" not in data:
        raise WeightsError(f"weights manifest {path.name} has no 'weights' list")
    return data


def manifest_entries(path=None) -> list[dict]:
    """The manifest's list of weight entries."""
    return list(load_manifest(path)["weights"])


def find_entry(filename: str, *, backend=None, path=None):
    """Return the manifest entry for *filename* (optionally scoped to *backend*).

    Returns ``None`` when no entry matches (a user/custom weight).
    """
    name = Path(str(filename)).name
    for entry in manifest_entries(path):
        if entry.get("filename") == name and (
                backend is None or entry.get("backend") == backend):
            return entry
    return None


def entry_dest(entry: dict) -> Path:
    """Resolved local path for a manifest *entry* (``Weights/{backend}/{file}``)."""
    return resolve_weight(entry["backend"], entry["filename"])


# ══════════════════════════════════════════════════════════════════════════════
# Verify + download
# ══════════════════════════════════════════════════════════════════════════════

# verify() outcomes
OK = "ok"
MISMATCH = "mismatch"
MISSING = "missing"


def verify(path, entry: dict) -> str:
    """Compare *path* against the manifest *entry*'s sha256.

    Returns ``"missing"`` (no file), ``"mismatch"`` (present but wrong bytes),
    or ``"ok"``.  Entries with no ``sha256`` (e.g. ``ultralytics-auto``) can
    only ever be ``"missing"`` or ``"ok"`` on presence.
    """
    path = Path(path)
    if not path.exists():
        return MISSING
    expected = entry.get("sha256")
    if not expected:
        return OK
    return OK if sha256_file(path) == expected else MISMATCH


def download(entry: dict, *, dest=None, progress=print, retries: int = 2) -> Path:
    """Download a manifest *entry* to its resolved path, verifying the sha256.

    A partial/failed download is deleted; a checksum mismatch deletes the file
    and raises.  Network failures raise :class:`WeightsError` with readable text
    (never a bare traceback -- G-OFFLINE).  Returns the destination path.
    """
    url = entry.get("url")
    if not url or entry.get("source") == SOURCE_ULTRALYTICS_AUTO:
        raise WeightsError(
            f"{entry.get('filename', '?')} has no download URL "
            f"({entry.get('source', 'unknown')} source); "
            "it is fetched automatically on first use")
    dest = Path(dest) if dest is not None else entry_dest(entry)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    last_err: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            progress(f"  Downloading {dest.name} (attempt {attempt}) ...")
            urllib.request.urlretrieve(url, str(tmp))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_err = exc
            if tmp.exists():
                tmp.unlink()
            continue
        # Verify before moving into place.
        expected = entry.get("sha256")
        if expected:
            got = sha256_file(tmp)
            if got != expected:
                tmp.unlink()
                raise WeightsError(
                    f"{dest.name} downloaded but its checksum does not match the "
                    f"published weight (expected {expected[:12]}..., got "
                    f"{got[:12]}...) -- the file was deleted; try again")
        tmp.replace(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        progress(f"  Saved {dest.name} ({size_mb:.1f} MB) OK")
        return dest

    raise WeightsError(
        f"could not download {dest.name} from {url} -- {last_err}. "
        "Check your internet connection and try again")


# ══════════════════════════════════════════════════════════════════════════════
# CLI (console script: mindsight-weights)
# ══════════════════════════════════════════════════════════════════════════════

def _select(entries, args) -> list[dict]:
    """Filter manifest *entries* per --required / --backend."""
    sel = entries
    if args.backend:
        wanted = set(args.backend)
        sel = [e for e in sel if e.get("backend") in wanted]
    if args.required:
        sel = [e for e in sel if e.get("required")]
    return sel


def _downloadable(entry: dict) -> bool:
    return bool(entry.get("url")) and entry.get("source") != SOURCE_ULTRALYTICS_AUTO


def downloadable_missing(names, *, path=None) -> list[dict]:
    """Manifest entries for *names* that are missing on disk AND downloadable.

    The one-click preflight fetch consumes this (SP3 D11 -- consume, don't
    compute): the caller passes the weight filenames a run needs; this returns
    exactly the subset the manager can fetch (a real upstream URL, and the
    resolved file is currently :data:`MISSING`).  Auto-fetch entries and files
    already present are excluded.  Order and de-duplication follow *names*.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for name in names:
        base = Path(str(name)).name
        if base in seen:
            continue
        seen.add(base)
        entry = find_entry(base, path=path)
        if (entry is not None and _downloadable(entry)
                and verify(entry_dest(entry), entry) == MISSING):
            out.append(entry)
    return out


def main(argv=None) -> int:
    """``mindsight-weights`` -- verify and download weights from the manifest."""
    parser = argparse.ArgumentParser(
        prog="mindsight-weights",
        description="Download and verify MindSight model weights from the manifest.")
    parser.add_argument("--required", action="store_true",
                        help="Operate only on required weights (installer default).")
    parser.add_argument("--all", action="store_true",
                        help="Operate on every downloadable weight in the manifest.")
    parser.add_argument("--backend", action="append", default=None,
                        help="Limit to backend(s) (e.g. MGaze, Gazelle, YOLO). Repeatable.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only report present/verified/missing; download nothing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even when the file is already present and valid.")
    parser.add_argument("--manifest", default=None,
                        help="Path to a weights_manifest.json (default: repo root).")
    args = parser.parse_args(argv)

    if not (args.required or args.all or args.backend or args.verify_only):
        # Bare invocation defaults to the required set (installer contract).
        args.required = True

    try:
        entries = manifest_entries(args.manifest)
    except WeightsError as exc:
        print(f"ERROR: {exc}")
        return 1

    selected = _select(entries, args)
    if not selected:
        print("No matching weights in the manifest.")
        return 0

    if args.verify_only:
        bad = 0
        for e in selected:
            state = verify(entry_dest(e), e)
            tag = {OK: "OK     ", MISMATCH: "MISMATCH", MISSING: "MISSING "}[state]
            print(f"  [{tag}] {e['backend']}/{e['filename']}")
            if state != OK:
                bad += 1
        print(f"\n{len(selected)} checked, {bad} need attention.")
        return 1 if bad else 0

    downloaded = skipped = failed = would = 0
    for e in selected:
        dest = entry_dest(e)
        if not _downloadable(e):
            print(f"  [skip] {e['backend']}/{e['filename']} "
                  f"({e.get('source')}: fetched automatically on first use)")
            skipped += 1
            continue
        if not args.force and verify(dest, e) == OK:
            print(f"  [have] {e['backend']}/{e['filename']} already present and verified")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  [dry-run] {e['backend']}/{e['filename']}  <-  {e['url']}")
            would += 1
            continue
        try:
            download(e)
            downloaded += 1
        except WeightsError as exc:
            print(f"  FAILED: {e['backend']}/{e['filename']} -- {exc}")
            failed += 1

    if args.dry_run:
        print(f"\n{would} file(s) would be downloaded.")
        return 0

    parts = []
    if downloaded:
        parts.append(f"{downloaded} downloaded")
    if skipped:
        parts.append(f"{skipped} skipped")
    if failed:
        parts.append(f"{failed} failed")
    print(f"\nWeights: {', '.join(parts) or 'nothing to do'}.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
