"""
mindsight.io.video_edit -- ffmpeg-backed source-video editing (UP4 / HP4).

Backs the Projects tab's Crop & Adjust tool: crop a raw study video to a
pixel rectangle and/or resample its frame rate, re-encoding with the same
H.264 settings the annotated-output remux uses.

ffmpeg resolution order (:func:`ffmpeg_exe`):
1. a system ``ffmpeg`` on PATH (keeps behavior byte-stable on machines that
   already had one -- the golden baselines were encoded with it), then
2. the static binary bundled by the ``imageio-ffmpeg`` wheel, so RA machines
   need nothing installed.

The same resolver now feeds the post-run H.264 remux in ``writers.py``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_exe() -> str | None:
    """Resolve an ffmpeg executable: system PATH first, bundled wheel second."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 -- optional dependency / download failure
        return None


def _even(v: int) -> int:
    """libx264 with yuv420p needs even frame dimensions."""
    return max(2, int(v) - (int(v) % 2))


def crop_video(src, dst, *, rect=None, fps=None) -> Path:
    """Re-encode *src* to *dst*, cropped to *rect* and/or resampled to *fps*.

    ``rect`` is ``(x, y, w, h)`` in source pixels (width/height are rounded
    DOWN to even values for the encoder); ``fps`` is the target frame rate.
    At least one of the two must be given.  Audio streams are copied through
    untouched.  Raises ``RuntimeError`` with the tail of ffmpeg's stderr on
    failure, ``ValueError`` on bad arguments.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise ValueError(f"video not found: {src}")
    if rect is None and fps is None:
        raise ValueError("nothing to do: give a crop rectangle and/or a "
                         "target frame rate")
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError(
            "ffmpeg is not available (neither on PATH nor via the bundled "
            "imageio-ffmpeg binary)")

    filters = []
    if rect is not None:
        x, y, w, h = (int(v) for v in rect)
        if w <= 0 or h <= 0:
            raise ValueError(f"empty crop rectangle: {rect}")
        filters.append(f"crop={_even(w)}:{_even(h)}:{max(0, x)}:{max(0, y)}")
    if fps is not None:
        if fps <= 0:
            raise ValueError(f"bad target frame rate: {fps}")
        filters.append(f"fps={fps:g}")

    cmd = [exe, "-y", "-i", str(src), "-vf", ",".join(filters),
           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-c:a", "copy", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode(errors="replace")[-400:]
        if dst.exists():
            dst.unlink()
        raise RuntimeError(f"ffmpeg failed on {src.name}:\n{tail}") from exc
    return dst


def apply_edit(src, *, rect=None, fps=None, overwrite=False) -> Path | None:
    """Edit *src* IN PLACE (same filename, so project discovery and run.yaml
    are untouched), keeping a backup unless *overwrite*.

    Non-destructive default: the untouched original moves to an ``original/``
    folder beside the video (a subfolder is invisible to run discovery, which
    only looks at top-level files).  Returns the backup path, or None when
    overwriting.
    """
    src = Path(src)
    tmp = src.with_name(src.stem + "._editing_tmp" + src.suffix)
    crop_video(src, tmp, rect=rect, fps=fps)
    backup = None
    if not overwrite:
        backup_dir = src.parent / "original"
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / src.name
        n = 2
        while backup.exists():
            backup = backup_dir / f"{src.stem}_{n}{src.suffix}"
            n += 1
        shutil.move(str(src), str(backup))
    tmp.replace(src)
    return backup


def extract_frames(video, out_dir, *, count: int = 8) -> list[Path]:
    """Extract *count* evenly spaced frames from *video* as JPEGs (MP2).

    Frames are taken at the midpoints of *count* equal segments, so the
    fade-in first frame and near-duplicate last frame are naturally avoided.
    Returns the written paths (named ``<stem>_f<frame#>.jpg``).  Raises
    ``ValueError`` when the video cannot be opened or reports no frames.
    """
    import cv2
    video, out_dir = Path(video), Path(out_dir)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            raise ValueError(f"video reports no frames: {video}")
        count = max(1, min(int(count), total))
        indices = [min(int((i + 0.5) * total / count), total - 1)
                   for i in range(count)]
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            p = out_dir / f"{video.stem}_f{idx:06d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written.append(p)
        return written
    finally:
        cap.release()
