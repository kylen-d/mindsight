"""W3Z item 6: --overlay-theme (classic | mindsight).

Classic is the byte-exact historical look -- the smoke/CSV/SSIM goldens pin
it end-to-end; here we pin its palette values and the default plumbing so a
theme regression is caught fast.  The mindsight theme must actually change
pixels, and unknown names must fall back to classic.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from mindsight.outputs import dashboard_output as do


@pytest.fixture(autouse=True)
def _reset_theme():
    yield
    do.set_overlay_theme("classic")


def _ctx(frame):
    return {
        "frame": frame,
        "persons_gaze": [(np.array([50.0, 50.0]), np.array([150.0, 100.0]),
                          (0.1, 0.2))],
        "all_targets": [{"class_name": "cup", "conf": 0.9, "cls_id": 41,
                         "x1": 100, "y1": 120, "x2": 160, "y2": 170}],
        "hits": set(), "lock_info": None, "ray_snapped": [False],
        "ray_extended": [False], "tip_convergences": None,
        "face_bboxes": [(40, 40, 80, 90)], "face_track_ids": [0],
        "pid_map": None,
    }


def _cfg(theme):
    return types.SimpleNamespace(gaze_debug=False, tip_radius=80,
                                 gaze_cone_angle=0.0, _lite_overlay=False,
                                 _overlay_theme=theme)


def test_classic_palette_is_pinned():
    c = do.OVERLAY_THEMES["classic"]
    assert c["face_cols"][0] == (100, 100, 255)
    assert c["joint"] == (0, 200, 255)
    assert c["lock"] == (0, 215, 255)
    assert c["conv"] == (0, 220, 180)
    assert c["label_fill"] is None and c["label_text"] == (255, 255, 255)
    assert c["badge_bg"] == (20, 20, 20)
    assert c["dash_bg"] == (18, 18, 18)


def test_default_is_classic_and_unknown_falls_back():
    assert do._ACTIVE_THEME == "classic"
    do.set_overlay_theme("no-such-theme")
    assert do._ACTIVE_THEME == "classic"
    assert do._FACE_COLS[0] == (100, 100, 255)


def test_set_theme_rebinds_and_restores():
    do.set_overlay_theme("mindsight")
    assert do._FACE_COLS[0] == do.OVERLAY_THEMES["mindsight"]["face_cols"][0]
    assert do._BADGE_BG == do.OVERLAY_THEMES["mindsight"]["badge_bg"]
    do.set_overlay_theme("classic")
    assert do._FACE_COLS[0] == (100, 100, 255)
    assert do._BADGE_BG == (20, 20, 20)


def test_draw_overlay_theme_changes_pixels_only_when_selected():
    base = np.full((240, 320, 3), 90, dtype=np.uint8)
    classic1 = do.draw_overlay(_ctx(base.copy()), gaze_cfg=_cfg("classic"))
    themed = do.draw_overlay(_ctx(base.copy()), gaze_cfg=_cfg("mindsight"))
    classic2 = do.draw_overlay(_ctx(base.copy()), gaze_cfg=_cfg("classic"))
    assert not np.array_equal(classic1, themed)
    # classic after a themed run is byte-identical to classic before it --
    # theme state never leaks between runs.
    np.testing.assert_array_equal(classic1, classic2)


def test_no_gaze_cfg_means_classic():
    base = np.full((240, 320, 3), 90, dtype=np.uint8)
    do.set_overlay_theme("mindsight")       # stale state from a prior run
    bare = do.draw_overlay(_ctx(base.copy()), gaze_cfg=None)
    classic = do.draw_overlay(_ctx(base.copy()), gaze_cfg=_cfg("classic"))
    np.testing.assert_array_equal(bare, classic)


def test_labelled_box_styles():
    do.set_overlay_theme("classic")
    f = np.zeros((100, 200, 3), dtype=np.uint8)
    do._draw_labelled_box(f, 20, 40, 120, 90, (100, 100, 255), "cup 0.90")
    # Classic: solid box-colour tab above the box.
    assert tuple(f[35, 60]) == (100, 100, 255)
    do.set_overlay_theme("mindsight")
    f2 = np.zeros((100, 200, 3), dtype=np.uint8)
    do._draw_labelled_box(f2, 20, 40, 120, 90, (100, 100, 255), "cup 0.90")
    # Mindsight: indigo-ink tab interior instead of the solid colour fill.
    assert tuple(f2[35, 60]) == do.OVERLAY_THEMES["mindsight"]["badge_bg"]


def test_flag_plumbing():
    from mindsight.cli_flags import parse_cli
    from mindsight.pipeline import RunOptions

    assert RunOptions().overlay_theme == "classic"
    assert parse_cli([]).overlay_theme == "classic"
    assert parse_cli(["--overlay-theme", "mindsight"]).overlay_theme == "mindsight"
