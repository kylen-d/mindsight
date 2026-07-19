"""
validation/store.py — Validation-set data model + on-disk store.

One JSON file per set.  The file IS a valid eval-harness labels file
(``labels`` carries ``{frame: {participant: {x, y} | state}}`` exactly
like ``eval_data/{stem}_labels.json``; ``scripts/eval_gaze.py score``
reads it unchanged) extended with set metadata and per-frame labeled
object boxes for the IoU metric:

    {
      "format": 1,
      "name": "office-a",
      "video": "/abs/path/clip.mp4",
      "every": 10,
      "note": "",
      "labels":  {"120": {"P0": {"x": 451, "y": 475}, "P1": "offscreen"}},
      "objects": {"120": [{"name": "plate", "x1": .., "y1": .., "x2": .., "y2": ..}]}
    }

Sets live in ``<project>/validation/`` when a project is open, else
``~/.mindsight/validation/`` (via :func:`validation_root`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

FORMAT_VERSION = 1

#: Non-point label states (mirroring scripts/eval_annotate.py):
#: "offscreen" scores via the in/out head, "uncertain"/"skip" are
#: excluded from scoring.
LABEL_STATES = ("offscreen", "uncertain", "skip")


class ValidationSetError(Exception):
    """A validation set could not be read or written (plain-English str)."""


def validation_root(project_dir=None) -> Path:
    """Directory holding validation sets.

    ``<project>/validation`` when *project_dir* is given, else the
    per-user state dir (``~/.mindsight/validation``; honors the
    MINDSIGHT_STATE_DIR/MINDSIGHT_HOME overrides via constants).
    """
    if project_dir:
        return Path(project_dir) / "validation"
    from mindsight.constants import state_dir
    return state_dir() / "validation"


def _slug(name: str) -> str:
    """Filesystem-safe stem for a set name."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    if not s:
        raise ValidationSetError(f"Set name {name!r} has no usable characters.")
    return s


def _valid_point(v) -> bool:
    return (isinstance(v, dict) and "x" in v and "y" in v
            and all(isinstance(v[k], (int, float)) for k in ("x", "y")))


@dataclass
class ValidationSet:
    """In-memory validation set.  Frame keys are ints; JSON uses strings."""

    name: str
    video: str = ""
    every: int | None = None
    note: str = ""
    #: {frame: {participant: {"x": int, "y": int} | one of LABEL_STATES}}
    labels: dict = field(default_factory=dict)
    #: {frame: [{"name": str, "x1": int, "y1": int, "x2": int, "y2": int}]}
    objects: dict = field(default_factory=dict)

    # ── Editing ──────────────────────────────────────────────────────────────

    def frames(self) -> list[int]:
        return sorted(set(self.labels) | set(self.objects))

    def add_frame(self, frame: int) -> None:
        self.labels.setdefault(int(frame), {})

    def remove_frame(self, frame: int) -> None:
        self.labels.pop(int(frame), None)
        self.objects.pop(int(frame), None)

    def set_label(self, frame: int, participant: str, value) -> None:
        """Set a participant's label: an ``{"x", "y"}`` point (coerced to
        int) or one of :data:`LABEL_STATES`."""
        if isinstance(value, str):
            if value not in LABEL_STATES:
                raise ValidationSetError(
                    f"Unknown label state {value!r}; known: {LABEL_STATES}")
        elif _valid_point(value):
            value = {"x": int(value["x"]), "y": int(value["y"])}
        else:
            raise ValidationSetError(
                f"Label must be an x/y point or one of {LABEL_STATES}, "
                f"got {value!r}")
        self.labels.setdefault(int(frame), {})[str(participant)] = value

    def clear_label(self, frame: int, participant: str) -> None:
        self.labels.get(int(frame), {}).pop(str(participant), None)

    def add_object(self, frame: int, name: str, box) -> None:
        x1, y1, x2, y2 = (int(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            raise ValidationSetError(f"Degenerate object box {box!r}")
        self.objects.setdefault(int(frame), []).append(
            {"name": str(name), "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    def remove_object(self, frame: int, index: int) -> None:
        boxes = self.objects.get(int(frame), [])
        if 0 <= index < len(boxes):
            boxes.pop(index)
        if not boxes:
            self.objects.pop(int(frame), None)

    def point_label_count(self) -> int:
        return sum(1 for f in self.labels.values()
                   for v in f.values() if isinstance(v, dict))

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "format": FORMAT_VERSION,
            "name": self.name,
            "video": self.video,
            "every": self.every,
            "note": self.note,
            "labels": {str(k): self.labels[k] for k in sorted(self.labels)},
            "objects": {str(k): self.objects[k] for k in sorted(self.objects)},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationSet":
        if not isinstance(data, dict) or "labels" not in data:
            raise ValidationSetError("Not a validation set (no 'labels' key).")
        labels: dict = {}
        for k, per_frame in (data.get("labels") or {}).items():
            entry: dict = {}
            for pid, v in (per_frame or {}).items():
                if isinstance(v, str) and v in LABEL_STATES:
                    entry[str(pid)] = v
                elif _valid_point(v):
                    entry[str(pid)] = {"x": int(v["x"]), "y": int(v["y"])}
                else:
                    raise ValidationSetError(
                        f"Frame {k} participant {pid}: bad label {v!r}")
            labels[int(k)] = entry
        objects: dict = {}
        for k, boxes in (data.get("objects") or {}).items():
            objects[int(k)] = [
                {"name": str(b.get("name", "")),
                 "x1": int(b["x1"]), "y1": int(b["y1"]),
                 "x2": int(b["x2"]), "y2": int(b["y2"])}
                for b in (boxes or [])
            ]
        return cls(
            name=str(data.get("name", "")),
            video=str(data.get("video", "") or ""),
            every=data.get("every"),
            note=str(data.get("note", "") or ""),
            labels=labels,
            objects=objects,
        )


class ValidationStore:
    """Loads/saves validation sets under one root directory."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path_for(self, name: str) -> Path:
        return self.root / f"{_slug(name)}.json"

    def list_sets(self) -> list[dict]:
        """[{name, path, frames, points}] for every readable set, sorted
        by name.  Unreadable files are skipped (never crash the GUI on a
        stray file)."""
        out = []
        if not self.root.is_dir():
            return out
        for path in sorted(self.root.glob("*.json")):
            try:
                vset = self.load_path(path)
            except (ValidationSetError, OSError):
                continue
            out.append({"name": vset.name or path.stem, "path": path,
                        "frames": len(vset.frames()),
                        "points": vset.point_label_count()})
        return out

    def load_path(self, path: Path) -> ValidationSet:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationSetError(f"Cannot read {path}: {exc}") from exc
        vset = ValidationSet.from_dict(data)
        if not vset.name:
            vset.name = Path(path).stem
        return vset

    def load(self, name: str) -> ValidationSet:
        return self.load_path(self.path_for(name))

    def save(self, vset: ValidationSet) -> Path:
        """Atomic write (tmp + replace); returns the set's path."""
        if not vset.name:
            raise ValidationSetError("Set has no name.")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(vset.name)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(vset.to_dict(), indent=2) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
        return path

    def delete(self, name: str) -> None:
        self.path_for(name).unlink(missing_ok=True)
