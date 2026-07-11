"""camera_enum: name ordering must match cv2's AVFoundation indexing."""

from mindsight.GUI.camera_enum import _avf_sorted, list_cameras


def test_avf_sorted_orders_by_unique_id_not_list_position():
    # Real-world repro (eyes-on A3): AVFoundation enumerates the MacBook
    # camera first, but cv2 sorts by uniqueID, putting the iPhone (UID 53...)
    # at index 0.  Mapping list position to cv2 index opened the wrong camera.
    devices = [
        ("6C707041-05AC-0010-0008-000000000001", "MacBook Pro Camera"),
        ("53237E66-949F-4896-85F2-31A100000001", "Kylen's iPhone Camera"),
    ]
    assert _avf_sorted(devices) == ["Kylen's iPhone Camera",
                                    "MacBook Pro Camera"]


def test_list_cameras_returns_index_name_pairs():
    cams = list_cameras()
    assert cams, "must always return at least the blind fallback"
    for idx, name in cams:
        assert isinstance(idx, int) and isinstance(name, str) and name
