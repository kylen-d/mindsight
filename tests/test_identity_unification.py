"""Cross-file face-identity unification (v1.1 W1.1 + W1.2).

Before v1.1 the ``hits`` set was keyed by list position while ``hit_events``
used stable track IDs, and mutual_gaze / social_ref / gaze_follow /
gaze_leadership(tip) kept their own positional identities.  When the
left-to-right face order differed from track-ID order, "P0" meant different
people in different output files and pid_map labels attached to the wrong
faces.  These tests drive the trackers with face_track_ids DIFFERENT from
list positions and assert every identity that reaches outputs is the track
ID.  Also pins the gaze_following episode/summary leader-follower
convention (participant = follower in BOTH).
"""

import numpy as np

from mindsight.Phenomena.Default.gaze_following import GazeFollowingTracker
from mindsight.Phenomena.Default.mutual_gaze import MutualGazeTracker
from mindsight.Phenomena.Default.social_referencing import SocialReferenceTracker
from mindsight.PostProcessing.RayForming.hit_detection import (
    compute_ray_intersections,
)


class _Cfg:
    """Minimal gaze_cfg for compute_ray_intersections."""
    hit_conf_gate = 0.0
    detect_extend = 0.0
    detect_extend_scope = 'objects'
    gaze_cone_angle = 0.0
    gaze_tips = False
    tip_radius = 80
    forward_gaze_threshold = 0.0


def _obj(name, x1, y1, x2, y2):
    return {'class_name': name, 'conf': 0.9,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


def test_hits_keyed_by_track_id():
    """The hits set and hit_events agree on the face identity: the track ID."""
    persons_gaze = [((0.0, 50.0), (100.0, 50.0), None)]   # ray crosses the box
    objects = [_obj('cup', 40, 30, 60, 70)]
    _targets, hits, hit_events = compute_ray_intersections(
        persons_gaze, [0.9], [7], [], objects, _Cfg())

    assert hits == {(7, 0)}                    # track ID 7, not position 0
    assert hit_events[0]['face_idx'] == 7


def test_mutual_gaze_pairs_use_track_ids():
    t = MutualGazeTracker()
    persons_gaze = [
        (np.array([100.0, 200.0]), np.array([300.0, 200.0]), 0.9),
        (np.array([300.0, 200.0]), np.array([100.0, 200.0]), 0.9),
    ]
    face_bboxes = [(70, 160, 130, 240), (270, 160, 330, 240)]
    # Positions 0/1 belong to track IDs 5/2 (order churned earlier in the run).
    result = t.update(persons_gaze=persons_gaze, face_bboxes=face_bboxes,
                      face_track_ids=[5, 2], frame_no=0)

    assert result['pairs'] == {(2, 5)}         # tid pair, normalized ascending
    assert t.pair_counts == {(2, 5): 1}
    t.finalize(1)
    row = t._episodes.rows[0]
    assert {row['participant'], row['partner']} == {2, 5}


def test_social_ref_state_survives_position_swap():
    """A face-look then object-look across a face-order swap still counts:
    the state machine is keyed by track ID, not by list position."""
    t = SocialReferenceTracker(window_frames=30)
    face_a = (70, 160, 130, 240)     # face at x~100
    face_b = (270, 160, 330, 240)    # face at x~300

    # Frame 0: track 9 at position 0 looks at track 4's face.
    t.update(frame_no=0,
             persons_gaze=[
                 (np.array([100.0, 200.0]), np.array([300.0, 200.0]), 0.9),
                 (np.array([300.0, 200.0]), np.array([400.0, 200.0]), 0.9)],
             face_bboxes=[face_a, face_b],
             face_track_ids=[9, 4], dets=[], hits=set())

    # Frame 1: order SWAPPED (track 9 now at position 1); track 9's ray now
    # points above both faces at object 0, delivered via the tid-keyed hits
    # set (the ray must not cross any face bbox or the state re-arms).
    dets = [_obj('toy', 500, 80, 560, 120)]
    t.update(frame_no=1,
             persons_gaze=[
                 (np.array([300.0, 200.0]), np.array([400.0, 200.0]), 0.9),
                 (np.array([100.0, 100.0]), np.array([520.0, 100.0]), 0.9)],
             face_bboxes=[face_b, face_a],
             face_track_ids=[4, 9], dets=dets, hits={(9, 0)})

    assert len(t.event_log) == 1
    ev = t.event_log[0]
    assert ev['face_idx'] == 9                 # the referencer, by track ID
    assert ev['prior_face_targets'] == [4]     # the referenced face, by tid
    assert ev['object_names'] == ['toy']


def test_gaze_following_episode_and_summary_share_convention():
    """participant = FOLLOWER, partner = LEADER in BOTH the episode rows and
    the summary rows (W1.2: the summary side used to be swapped)."""
    t = GazeFollowingTracker(lag_frames=10)
    t.update(frame_no=0, hits={(3, 0)})        # leader tid 3 acquires obj 0
    t.update(frame_no=2, hits={(3, 0), (8, 0)})  # follower tid 8 follows

    assert len(t.event_log) == 1
    assert t.event_log[0] == {'leader': 3, 'follower': 8, 'obj_idx': 0,
                              'lag_frames': 2, 'frame': 2}

    ep = t._episodes.rows[0]
    assert ep['participant'] == 8              # follower (test-locked shape)
    assert ep['partner'] == 3                  # leader

    summary = t.summary_metrics(total_frames=10, fps=30.0)
    count_row = next(r for r in summary if r['metric'] == 'event_count')
    assert count_row['participant'] == 'P8'    # follower -- was 'P3' pre-W1.2
    assert count_row['partner'] == 'P3'        # leader
    assert count_row['value'] == 1
