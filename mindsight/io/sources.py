"""
mindsight.io.sources -- Input sources for the pipeline.

Owns the primary capture (video file / image / webcam) and the auxiliary
video-stream lifecycle: open, per-frame read, and face-detection enrichment.
Extracted from mindsight.cli / mindsight.pipeline as part of the SP1.2 io extraction; the
aux-stream helpers are moved verbatim, so behavior is byte-identical.
"""

import cv2


def open_video_source(source):
    """Open the primary video/webcam capture; return ``(cap, fps)``.

    Raises ``RuntimeError`` if the source cannot be opened.  ``fps`` falls back
    to 30.0 when the backend reports 0 (typical for webcams / some codecs).

    A digit-only *source* string (e.g. ``"0"``) is normalized to the ``int``
    camera index cv2 needs -- ``cv2.VideoCapture("0")`` does NOT open a webcam,
    only ``cv2.VideoCapture(0)`` does.  File paths are passed through unchanged
    (they never satisfy ``str.isdigit``), so this is the single seam that makes
    a GUI/CLI ``source = "0"`` open camera 0.
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, fps


def read_image_source(source):
    """Read a single still image; raise ``FileNotFoundError`` if unreadable."""
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Cannot read: {source}")
    return frame


def open_aux_streams(output_cfg, main_fps):
    """Open auxiliary video captures and return (captures_dict, ended_set).

    Each AuxStreamConfig may list multiple participants.  A single
    VideoCapture is opened per config; ``read_aux_frames`` then
    populates keys for every participant in the config.
    """
    # canonical_key -> (VideoCapture, AuxStreamConfig)
    aux_captures: dict[str, tuple] = {}
    _aux_ended: set[str] = set()

    if output_cfg.aux_streams:
        for aux in output_cfg.aux_streams:
            ckey = f"{aux.stream_label}:{aux.source}"
            ac = cv2.VideoCapture(aux.source)
            if not ac.isOpened():
                print(f"Warning: cannot open aux stream "
                      f"{aux.stream_label} ({aux.source}) -- skipping")
                continue
            aux_fps = ac.get(cv2.CAP_PROP_FPS) or 30.0
            if abs(aux_fps - main_fps) > 1.0:
                print(f"Warning: aux stream {aux.stream_label} "
                      f"FPS ({aux_fps:.1f}) differs from main ({main_fps:.1f}) "
                      f"-- frames may drift")
            aux_captures[ckey] = (ac, aux)
        if aux_captures:
            print(f"Opened {len(aux_captures)} auxiliary stream(s)")
    return aux_captures, _aux_ended


def read_aux_frames(aux_captures, _aux_ended, frame_no):
    """Read one frame from each auxiliary stream.

    Returns a dict keyed by ``(pid, stream_label, video_type)`` so that
    ``find_aux_frame()`` can search by any combination of fields.
    For multi-participant streams, the same frame is stored under each
    participant's key.
    """
    aux_frames: dict[tuple, object] = {}
    for ckey, (ac, cfg) in aux_captures.items():
        ret_a, frame_a = ac.read()
        if not ret_a:
            frame_a = None
            if ckey not in _aux_ended:
                _aux_ended.add(ckey)
                print(f"Warning: aux stream {cfg.stream_label} "
                      f"ended at frame {frame_no}")

        # Register under each participant listed in the config
        for pid in cfg.participants:
            aux_frames[(pid, cfg.stream_label, cfg.video_type)] = frame_a

    return aux_frames


def enrich_aux_with_face_detection(aux_frames, aux_captures, face_det,
                                   pid_map):
    """Run face detection on WIDE_CLOSEUP/FACE_CLOSEUP aux streams.

    For streams with ``auto_detect_faces=True``, detect faces in the aux
    frame and match them to known participants using spatial ordering
    (left-to-right).  Populates per-participant face crops from the
    wide/face stream so plugins can access individual participant data.
    """
    from mindsight.pipeline_config import VideoType
    if not aux_captures:
        return

    for ckey, (ac, cfg) in aux_captures.items():
        if not cfg.auto_detect_faces:
            continue
        if cfg.video_type not in (VideoType.WIDE_CLOSEUP,
                                  VideoType.FACE_CLOSEUP):
            continue

        # Find the frame in aux_frames for this config
        ref_frame = None
        for pid in cfg.participants:
            key = (pid, cfg.stream_label, cfg.video_type)
            if key in aux_frames and aux_frames[key] is not None:
                ref_frame = aux_frames[key]
                break

        if ref_frame is None:
            continue

        # Detect faces in the aux frame
        try:
            detected = face_det.detect(ref_frame)
        except Exception:
            continue

        if not detected:
            continue

        # Sort detected faces left-to-right by x1
        detected = sorted(detected, key=lambda f: f.get('bbox', [0])[0]
                          if 'bbox' in f else 0)

        # Match to participants: positional mapping (left-to-right order
        # matches the order in cfg.participants)
        for i, pid in enumerate(cfg.participants):
            if i >= len(detected):
                break
            face = detected[i]
            bbox = face.get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = ref_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            face_crop = ref_frame[y1:y2, x1:x2]
            # Add per-participant face crop as an additional aux entry
            crop_key = (pid, f"{cfg.stream_label}_face",
                        VideoType.FACE_CLOSEUP)
            aux_frames[crop_key] = face_crop
