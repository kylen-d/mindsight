"""
vp_archive.py -- portable, self-contained Visual Prompt archives (MP4).

A plain ``.vp.json`` stores ABSOLUTE reference-image paths, so it breaks the
moment it moves to another machine (or the images do).  A ``.vp.zip`` archive
bundles ``vp.json`` (with archive-relative image paths) together with every
referenced image, so one file carries the whole prompt.

Importing EXTRACTS the archive into a folder next to it and materializes a
normal ``.vp.json`` with absolute paths -- everything downstream (the VP
builder, ``--vp-file``, project ``Inputs/Prompts/``) keeps consuming plain
VP files and never learns about zips.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

VP_ARCHIVE_EXT = ".vp.zip"


def export_vp_archive(zip_path, classes: list, references: list) -> Path:
    """Write a self-contained ``.vp.zip`` from in-memory VP data.

    ``references`` entries carry absolute image paths (the builder's format);
    each image is stored under ``images/<idx>_<name>`` so same-named files
    from different folders never collide.  Raises ``ValueError`` when a
    referenced image is missing on disk.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        new_refs = []
        for i, ref in enumerate(references):
            src = Path(ref["image"])
            if not src.is_file():
                raise ValueError(f"reference image not found: {src}")
            arcname = f"images/{i:03d}_{src.name}"
            zf.write(src, arcname)
            new_refs.append({"image": arcname,
                             "annotations": ref["annotations"]})
        zf.writestr("vp.json", json.dumps(
            {"version": 1, "classes": classes, "references": new_refs},
            indent=2))
    return zip_path


def import_vp_archive(zip_path, dest_dir=None) -> Path:
    """Extract a ``.vp.zip`` and return the materialized ``.vp.json`` path.

    Extraction lands in *dest_dir* (default: a collision-safe folder named
    after the archive, next to it); image paths in the returned VP file are
    absolute.  Only ``images/...`` members are extracted (path-traversal
    safe).  Raises ``ValueError`` on a malformed archive.
    """
    zip_path = Path(zip_path)
    name = zip_path.name
    stem = (name[:-len(VP_ARCHIVE_EXT)] if name.endswith(VP_ARCHIVE_EXT)
            else zip_path.stem)
    if dest_dir is None:
        dest = zip_path.parent / stem
        n = 2
        while dest.exists():
            dest = zip_path.parent / f"{stem}_{n}"
            n += 1
    else:
        dest = Path(dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        try:
            data = json.loads(zf.read("vp.json"))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"not a valid VP archive (no readable vp.json): {zip_path}"
            ) from exc
        (dest / "images").mkdir(parents=True, exist_ok=True)
        for member in zf.namelist():
            # Only the flat images/ payload; refuse traversal shenanigans.
            if (not member.startswith("images/") or ".." in member
                    or member.endswith("/")):
                continue
            target = dest / "images" / Path(member).name
            target.write_bytes(zf.read(member))
        for ref in data.get("references", []):
            ref["image"] = str(
                (dest / "images" / Path(ref["image"]).name).resolve())
    missing = [r["image"] for r in data.get("references", [])
               if not Path(r["image"]).is_file()]
    if missing:
        raise ValueError(
            f"VP archive is missing {len(missing)} image(s): "
            f"{Path(missing[0]).name}...")
    vp_json = dest / f"{stem}.vp.json"
    vp_json.write_text(json.dumps(data, indent=2))
    return vp_json
