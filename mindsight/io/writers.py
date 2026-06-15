"""
mindsight.io.writers -- Output sinks for the pipeline.

Owns the annotated-video writer (mp4v + optional ffmpeg H.264 remux) and the
per-frame event-CSV handle.  ``open_video_writer`` and ``finalize_video`` were
historically defined in ``mindsight.outputs.dashboard_output``; they are pure
IO (not drawing) and moved here as part of the SP1.2 io extraction.
"""

import csv
from pathlib import Path

import cv2

from mindsight.constants import OUTPUTS_ROOT as _OUTPUTS_ROOT


def open_video_writer(save_arg, source, cap, *, no_dashboard=False):
    """Create and return a (VideoWriter, path) tuple for the annotated output.

    Parameters
    ----------
    save_arg     : True  → write to Outputs/Video/[stem]_Video_Output.mp4
                   str   → write to that path
                   None/False → do not record; returns (None, None)
    source       : video file path (str/Path) or webcam index (int).
    cap          : open cv2.VideoCapture used to query FPS and frame size.
    no_dashboard : if True, frames are raw (no side panels), so the writer
                   is sized to the original video dimensions.

    Returns
    -------
    (cv2.VideoWriter, str) or (None, None)
    """
    if not save_arg:
        return None, None
    if save_arg is True:
        stem = Path(str(source)).stem if not isinstance(source, int) else "webcam"
        path = str(_OUTPUTS_ROOT / "Video" / f"{stem}_Video_Output.mp4")
    else:
        path = save_arg
    fps0   = cap.get(cv2.CAP_PROP_FPS) or 30
    fw, fh = int(cap.get(3)), int(cap.get(4))
    if no_dashboard:
        out_w = fw
    else:
        panel_w = max(280, int(fw * 0.22))
        out_w = fw + 2 * panel_w
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps0, (out_w, fh))
    print(f"Saving → {path}")
    return writer, path


def finalize_video(path):
    """Remux mp4v video to H.264 via ffmpeg for broad player compatibility.

    If ffmpeg is not available, the original mp4v file is kept as-is
    (playable in VLC and most players, but not QuickTime on macOS).
    """
    if path is None:
        return
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        print("Note: ffmpeg not found; video saved as MPEG-4 Part 2 (mp4v).\n"
              "      Install ffmpeg for H.264 output (QuickTime compatible).")
        return

    tmp = path + ".h264.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-preset", "fast",
             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             tmp],
            check=True, capture_output=True,
        )
        shutil.move(tmp, path)
        print(f"Video remuxed to H.264 → {path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg failed — keep the original mp4v file
        if Path(tmp).exists():
            Path(tmp).unlink()
        print("Note: H.264 remux failed; video saved as mp4v.")


def open_event_log(output_cfg):
    """Open the per-frame event CSV, write its header, and return (fh, writer).

    Returns ``(None, None)`` when no log path is configured.  The header layout
    and the optional ``video_name``/``conditions`` prefix columns mirror the
    setup previously inline in the run loop, byte-for-byte.
    """
    if not output_cfg.log_path:
        return None, None
    log_fh  = open(output_cfg.log_path, "w", newline="")
    log_csv = csv.writer(log_fh)
    header = ["frame","t_seconds","face_idx","object","object_conf",
              "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
              "joint_attention","joint_attention_confirmed",
              "participant_label"]
    if output_cfg.video_name is not None:
        header = ["video_name", "conditions"] + header
    log_csv.writerow(header)
    print(f"Logging → {output_cfg.log_path}")
    return log_fh, log_csv
