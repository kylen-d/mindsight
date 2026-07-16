"""DataCollection plugin lifecycle wiring (v1.1 W0.5).

In 1.0.0 DataCollection plugin instances were built (factory.build_data_plugins)
and seeded into ctx['data_plugins'], but ``on_frame`` and ``on_run_complete``
had zero call sites and ``data_collection_registry`` was omitted from the
parser's add_arguments loop -- the documented lifecycle was dead.  These tests
pin the repaired wiring:

* collect_frame_data fires ``on_frame`` once per frame with the frame context;
* finalize_run fires ``on_run_complete`` once with the summary kwargs;
* build_parser gives DataCollection plugins an add_arguments call.
"""

import numpy as np

from mindsight.outputs.data_pipeline import collect_frame_data, finalize_run
from mindsight.pipeline_config import FrameContext
from Plugins import DataCollectionPlugin, data_collection_registry


class _RecorderPlugin(DataCollectionPlugin):
    name = "recorder"

    def __init__(self):
        self.frames = []
        self.run_completes = []

    def on_frame(self, **kwargs):
        self.frames.append(kwargs)

    def on_run_complete(self, **kwargs):
        self.run_completes.append(kwargs)


def _ctx(plugin, **extra):
    return FrameContext(frame=np.zeros((4, 4, 3), dtype=np.uint8),
                        data_plugins=[plugin], **extra)


def test_on_frame_fires_with_frame_context():
    plugin = _RecorderPlugin()
    ctx = _ctx(plugin, confirmed_objs={"cup"})
    events = [{'face_idx': 3, 'object': 'cup', 'object_conf': 0.9,
               'bbox': (1, 2, 3, 4)}]

    collect_frame_data(ctx, log_csv=None, frame_no=17, hit_events=events,
                       face_track_ids=[3], persons_gaze=[])

    assert len(plugin.frames) == 1
    seen = plugin.frames[0]
    assert seen['frame_no'] == 17
    assert seen['hit_events'] == events
    assert seen['face_track_ids'] == [3]
    assert seen['confirmed_objs'] == {"cup"}
    assert 'frame' in seen  # full context, including the image


def test_on_frame_not_called_without_plugins():
    ctx = FrameContext(frame=np.zeros((4, 4, 3), dtype=np.uint8))
    collect_frame_data(ctx, log_csv=None, frame_no=0, hit_events=[],
                       face_track_ids=[], persons_gaze=[])
    # no data_plugins key: must not raise, nothing to assert beyond survival


def test_on_run_complete_fires_once_with_summary_kwargs():
    plugin = _RecorderPlugin()
    ctx = _ctx(plugin, total_frames=100, frame_no=99, total_hits=7,
               look_counts={(0, 'cup'): 7}, source='vid.mp4',
               video_name='vid', conditions='A', video_fps=30.0)

    finalize_run(ctx)

    assert len(plugin.run_completes) == 1
    seen = plugin.run_completes[0]
    assert seen['total_frames'] == 100
    assert seen['total_hits'] == 7
    assert seen['look_counts'] == {(0, 'cup'): 7}
    assert seen['source'] == 'vid.mp4'
    assert seen['video_name'] == 'vid'
    assert seen['conditions'] == 'A'
    assert seen['fps'] == 30.0


def test_parser_wires_data_collection_add_arguments(monkeypatch):
    class _FlaggedPlugin(DataCollectionPlugin):
        name = "flagged"
        called_with = []

        @classmethod
        def add_arguments(cls, parser):
            cls.called_with.append(parser)
            parser.add_argument("--recorder-out", default=None)

    monkeypatch.setitem(data_collection_registry._plugins, "flagged",
                        _FlaggedPlugin)
    from mindsight.cli_flags import build_parser
    parser = build_parser()

    assert _FlaggedPlugin.called_with, \
        "data_collection_registry.add_arguments never invoked by build_parser"
    ns = parser.parse_args(["--recorder-out", "x.json"])
    assert ns.recorder_out == "x.json"
