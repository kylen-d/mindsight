"""Stage shipped resources into mindsight/_bundled/ for wheel builds.

The wheel declares ``mindsight/_bundled/**`` as package data, but the
directory is gitignored: run this script immediately before ``uv build`` so
the wheel carries the docs tree, config presets, weights manifest, and app
icon.  Wheels built without running it install fine but fall back to hosted
docs and downloaded resources -- the census below is the guard against
shipping such a wheel by accident (the release procedure runs this script
and fails loudly if staging is incomplete).

Usage:
    python scripts/sync_bundled_resources.py [--target DIR] [--clean]

--target defaults to <repo>/mindsight/_bundled (override for tests).
--clean removes the staging tree instead of building it.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: (source relative to repo root, destination relative to the staging root).
#: Directories copy recursively; files copy one-to-one.
_ITEMS = [
    ("docs", "docs"),
    ("configs/pipeline_known_good.yaml", "configs/pipeline_known_good.yaml"),
    ("configs/pipeline_low_power.yaml", "configs/pipeline_low_power.yaml"),
    ("configs/KNOWN_GOOD.md", "configs/KNOWN_GOOD.md"),
    ("weights_manifest.json", "weights_manifest.json"),
    ("assets/mindsight_icon.png", "assets/mindsight_icon.png"),
]

#: Optional items: staged when present, no census failure when absent.
_OPTIONAL = {"assets/mindsight_icon.png"}

#: Files whose presence proves the staging is usable.
_CENSUS = [
    "docs/index.md",
    "docs/studies/run-a-study-tutorial.md",
    "configs/pipeline_known_good.yaml",
    "weights_manifest.json",
]


def sync(target: Path) -> int:
    if target.exists():
        shutil.rmtree(target)
    staged = 0
    for src_rel, dst_rel in _ITEMS:
        src = REPO / src_rel
        dst = target / dst_rel
        if not src.exists():
            if src_rel in _OPTIONAL:
                print(f"  skip (absent, optional): {src_rel}")
                continue
            print(f"FAIL: required source missing: {src_rel}")
            return 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
            n = sum(1 for f in dst.rglob("*") if f.is_file())
            staged += n
            print(f"  {src_rel}/ -> {n} files")
        else:
            shutil.copy2(src, dst)
            staged += 1
            print(f"  {src_rel}")
    missing = [rel for rel in _CENSUS if not (target / rel).is_file()]
    if missing:
        print(f"FAIL: census missing after sync: {missing}")
        return 1
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"PASS: {staged} files staged into {target} ({size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, default=REPO / "mindsight" / "_bundled")
    ap.add_argument("--clean", action="store_true", help="remove the staging tree")
    args = ap.parse_args()
    if args.clean:
        if args.target.exists():
            shutil.rmtree(args.target)
            print(f"removed {args.target}")
        return 0
    return sync(args.target)


if __name__ == "__main__":
    sys.exit(main())
