"""Camera enumeration that matches cv2's device indexing.

The GUI shows camera *names* but the capture pipeline opens cameras by
``cv2.VideoCapture(index)``.  Those two worlds do not order devices the same
way, and mapping combo position to cv2 index opens the WRONG camera (eyes-on
A3, 2026-07-10: picking "MacBook Pro Camera" opened the iPhone Continuity
stub).

On macOS, OpenCV's AVFoundation backend enumerates
``AVCaptureDevice devicesWithMediaType:Video`` (+ Muxed) and then SORTS the
list by ``uniqueID`` for stable indices across launches.  We reproduce that
exact ordering via PyObjC when available; every passive alternative
(QtMultimedia, ffmpeg -list_devices, system_profiler) uses a different order
and is wrong.

Fallback chain: PyObjC AVFoundation (darwin) -> QtMultimedia order (other
platforms, where backend order generally matches) -> cv2 open/close probe ->
blind "Camera 0-3" list.  Every entry is ``(cv2_index, display_name)``.
"""

from __future__ import annotations

import sys


def _avf_sorted(devices: list[tuple[str, str]]) -> list[str]:
    """Names ordered the way cv2 indexes them: sorted by AVFoundation
    uniqueID.  *devices* is (uniqueID, localizedName) pairs."""
    return [name for _uid, name in sorted(devices, key=lambda d: d[0])]


def _list_avfoundation() -> list[tuple[int, str]]:
    import AVFoundation as AV  # pyobjc-framework-AVFoundation (darwin dep)

    devs = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
    devs += list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
    names = _avf_sorted([(str(d.uniqueID()), str(d.localizedName()))
                         for d in devs])
    return list(enumerate(names))


def _list_qt() -> list[tuple[int, str]]:
    from PyQt6.QtMultimedia import QMediaDevices

    return [(i, d.description() or f"Camera {i}")
            for i, d in enumerate(QMediaDevices.videoInputs())]


def _list_cv2_probe(limit: int = 6) -> list[tuple[int, str]]:
    import cv2

    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, f"Camera {i} ({w}x{h})"))
        cap.release()
    return found


def list_cameras() -> list[tuple[int, str]]:
    """Enumerate cameras as ``(cv2_index, display_name)`` pairs.

    May trigger the OS camera-permission prompt; call on demand (Refresh),
    never at startup.
    """
    if sys.platform == "darwin":
        try:
            cams = _list_avfoundation()
            if cams:
                return cams
        except Exception:
            pass
    else:
        try:
            cams = _list_qt()
            if cams:
                return cams
        except Exception:
            pass
    try:
        cams = _list_cv2_probe()
        if cams:
            return cams
    except Exception:
        pass
    return [(i, f"Camera {i}") for i in range(4)]
