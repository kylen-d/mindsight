"""Equivalence tests: unified schema vs the existing config construction.

Two boundaries are pinned:

1. Namespace equivalence (the core guarantee).  For the default parser
   namespace and a battery of representative flag sets, the dataclasses the
   pipeline consumes today (built by direct ``from_namespace`` calls, as
   ``cli._build_from_args`` does) must equal the ones built via
   ``PipelineConfig.from_namespace(ns)`` -> ``config_compat.to_dataclasses``.

2. YAML equivalence.  For every pipeline YAML the legacy loader accepts,
   the schema built from a REAL ``load_pipeline`` merge must line up with
   ``config_compat.load_yaml``:

   * over an EMPTY namespace (the GUI pipeline_dialog route), the two must
     be IDENTICAL -- this pins alias, shim, and precedence semantics 1:1
     against the live loader;
   * over the DEFAULT PARSER namespace WITH ``_explicit_cli=frozenset()``
     (the CLI route after the YAML-precedence hotfix), the loader now honors
     every YAML value on any dest the user did NOT type -- so the two routes
     match EXACTLY.  This flips the SP1.1 frozen divergences: the old
     ``_is_default`` heuristic silently DROPPED any YAML value whose namespace
     attribute was truthy (e.g. conf=0.35 blocked ``detection.conf: 0.05``);
     with explicit-flag detection (ms.cli._args) that gating is gone.  The
     per-file ``EXPECTED_GATED`` sets below are therefore now empty.  Fix 2
     additionally corrected the ``detection.merge_overlap_strategy`` schema
     default ('filter' -> 'dynamic'), so ``ALWAYS_DIVERGENT`` is empty too and
     the CLI route matches ``load_yaml`` with zero divergences.
"""

import copy
from argparse import Namespace
from pathlib import Path

import pytest

from mindsight.config import PipelineConfig
from mindsight.config_compat import load_yaml, to_dataclasses
from mindsight.Phenomena.phenomena_config import PhenomenaConfig
from mindsight.pipeline_config import (
    AuxStreamConfig,
    DepthConfig,
    DetectionConfig,
    GazeConfig,
    OutputConfig,
    ProjectConfig,
    TrackerConfig,
    VideoType,
)
from mindsight.config_compat import load_pipeline
from mindsight.PostProcessing.RayForming.ray_config import RayFormingConfig
from tests.test_config_schema import get_parser

REPO_ROOT = Path(__file__).resolve().parents[1]

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def build_direct(ns, class_ids=None, blacklist=None):
    """Build the 8 dataclasses the way cli._build_from_args does today."""
    return (
        GazeConfig.from_namespace(ns),
        DetectionConfig.from_namespace(
            ns, class_ids, blacklist if blacklist is not None else set()),
        TrackerConfig.from_namespace(ns),
        RayFormingConfig.from_namespace(ns),
        DepthConfig.from_namespace(ns),
        PhenomenaConfig.from_namespace(ns),
        OutputConfig.from_namespace(ns),
        ProjectConfig(),
    )


def build_via_schema(ns, class_ids=None, blacklist=None):
    cfg = PipelineConfig.from_namespace(ns, class_ids=class_ids,
                                        blacklist=blacklist)
    return to_dataclasses(cfg)


def apply_build_preprocessing(ns):
    """Replicate the ns mutations cli._build_from_args performs before the
    from_namespace calls (--no-depth flips ns.depth)."""
    if getattr(ns, "no_depth", False):
        ns.depth = False
    return ns


def diff_paths(a: dict, b: dict, prefix: str = "") -> set[str]:
    """Dotted paths at which two nested model dumps differ."""
    out: set[str] = set()
    for key in a.keys() | b.keys():
        pa, pb = a.get(key), b.get(key)
        path = f"{prefix}{key}"
        if isinstance(pa, dict) and isinstance(pb, dict):
            out |= diff_paths(pa, pb, f"{path}.")
        elif pa != pb:
            out.add(path)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. Namespace equivalence
# ══════════════════════════════════════════════════════════════════════════════

FLAG_SETS = {
    "defaults": [],
    "legacy_rf_interval": ["--rf-gazelle-interval", "10"],
    "min_call_gap_wins": ["--min-call-gap", "45", "--rf-gazelle-interval", "10"],
    "merge_overlaps": ["--merge-overlaps", "--merge-overlap-strategy", "merge",
                       "--merge-overlap-threshold", "0.5"],
    "all_phenomena": ["--all-phenomena", "--ja-window", "30"],
    "phenomena_mix": ["--joint-attention", "--ja-quorum", "0.6",
                      "--mutual-gaze", "--aversion-conf", "0.8",
                      "--gaze-leader-tips", "--scanpath-dwell", "12"],
    "snap_and_tips": ["--conf-ray", "--gaze-tips", "--tip-radius", "40",
                      "--adaptive-ray", "snap", "--smooth-snap", "all",
                      "--smooth-snap-alpha", "0.5", "--snap-gate-angle", "45"],
    "rayforming_knobs": ["--len-hold-tau", "2.5", "--dir-beta", "0.9",
                         "--fixation-v-threshold", "0.1",
                         "--depth-ray-length", "--depth-belief-boost", "0.2"],
    "depth_on": ["--depth", "--depth-input-size", "256",
                 "--depth-w-depth", "0.7", "--depth-sample-radius", "4",
                 "--depth-aware-scoring", "--depth-skip-frames", "3"],
    "no_depth_override": ["--depth", "--no-depth"],
    "outputs": ["--anonymize", "blur", "--anonymize-padding", "0.5",
                "--save", "x.mp4", "--log", "e.csv", "--summary", "s.csv",
                "--heatmap", "h", "--charts"],
    "tracker_knobs": ["--gaze-lock", "--dwell-frames", "30", "--lock-dist",
                      "50", "--skip-frames", "3", "--obj-persistence", "5",
                      "--reid-grace-seconds", "2.0", "--gaze-cone", "12.5",
                      "--forward-gaze-threshold", "0"],
}


@pytest.mark.parametrize("name", list(FLAG_SETS), ids=list(FLAG_SETS))
def test_namespace_equivalence(name):
    ns = get_parser().parse_args(FLAG_SETS[name])
    apply_build_preprocessing(ns)
    class_ids = [0, 63]
    blacklist = {"tv", "laptop"}
    direct = build_direct(ns, class_ids, blacklist)
    schema = build_via_schema(ns, class_ids, blacklist)
    for got, want in zip(schema, direct):
        assert got == want, f"[{name}] {type(want).__name__} mismatch"


def test_namespace_equivalence_default_none_classes():
    ns = get_parser().parse_args([])
    direct = build_direct(ns, None, set())
    schema = build_via_schema(ns, None, set())
    for got, want in zip(schema, direct):
        assert got == want


def test_namespace_equivalence_with_aux_streams_and_pid_map():
    """aux_streams / pid_map are attached to ns by _build_from_args; the
    schema must round-trip them through AuxStream models unchanged."""
    ns = get_parser().parse_args([])
    ns.aux_streams = [
        AuxStreamConfig(source="eye.mp4", video_type=VideoType.EYE_ONLY,
                        stream_label="left_eye", participants=["S1"]),
        AuxStreamConfig(source="wide.mp4", video_type=VideoType.WIDE_CLOSEUP,
                        stream_label="room", participants=["S1", "S2"],
                        auto_detect_faces=False),
    ]
    ns.pid_map = {0: "S1", 1: "S2"}
    direct = build_direct(ns)
    schema = build_via_schema(ns)
    for got, want in zip(schema, direct):
        assert got == want


# ══════════════════════════════════════════════════════════════════════════════
# 2. YAML equivalence
# ══════════════════════════════════════════════════════════════════════════════

REPO_YAMLS = [
    "test_pipeline.yaml",
    "Projects/ExampleProject/Pipeline/pipeline.yaml",
    "Projects/ExampleProject/Pipeline/pipeline_example.yaml",
    # NOT Projects/ExampleProject/project.yaml -- that is a project.yaml
    # consumed by project.runner.load_project_config, not a pipeline YAML.
]

SYNTHETIC_YAMLS = {
    "legacy_bool_snap.yaml": """
gaze:
  adaptive_ray: true
  adaptive_snap: true
""",
    "legacy_bool_extend.yaml": """
gaze:
  adaptive_ray: true
""",
    "legacy_bool_off.yaml": """
gaze:
  adaptive_ray: false
""",
    "rf_interval.yaml": """
plugins:
  rf_gazelle_interval: 10
""",
    "rf_precedence.yaml": """
plugins:
  rf_gazelle_interval: 10
  min_call_gap: 45
""",
    "aux_streams.yaml": """
aux_streams:
  - source: eye.mp4
    video_type: eye_only
    stream_label: left_eye
    participants: [S1]
  - source: wide.mp4
    video_type: wide_closeup
    stream_label: room
    participants: [S1, S2]
    auto_detect_faces: false
""",
    "kitchen_sink.yaml": """
detection:
  conf: 0.2
  obj_persistence: 7
gaze:
  snap_bbox_scale: 0.25
  snap_w_size: 0.6
  hit_conf_gate: 0.3
  detect_extend: 25
  detect_extend_scope: both
  gaze_lock: true
  dwell_frames: 30
  gaze_debug: true
  snap_release_frames: 9
  snap_engage_frames: 3
depth:
  enabled: true
  depth_aware_scoring: true
  snap_w_depth: 0.9
  gaze_sample_radius: 5
  input_size: 256
output:
  anonymize: blur
  anonymize_padding: 0.5
  save_video: out.mp4
  log_csv: true
phenomena:
  ja_window: 12
  ja_quorum: 0.5
  ja_window_thresh: 0.9
""",
    "unknown_keys.yaml": """
totally_unknown: 1
detection:
  conf: 0.2
  nonsense_key: true
gaze:
  ray_length: 1.5
whatever:
  nested: {deep: value}
""",
}

# No always-divergent keys remain: Fix 2 flipped the schema default of
# merge_overlap_strategy from the dead 'filter' fallback to the runtime-true
# 'dynamic', so load_yaml and the CLI route now agree on it too.
ALWAYS_DIVERGENT: set = set()

# HOTFIX (YAML precedence): with explicit-flag detection the CLI route no
# longer drops YAML values on truthy-default dests, so the frozen per-file
# gated divergences of SP1.1 are now all EMPTY -- every YAML value lands and
# the two routes match exactly (apart from ALWAYS_DIVERGENT above).  The keys
# below are kept only to enumerate the files under test.
EXPECTED_GATED = {name: {} for name in [
    "test_pipeline.yaml",
    "Projects/ExampleProject/Pipeline/pipeline.yaml",
    "Projects/ExampleProject/Pipeline/pipeline_example.yaml",
    "legacy_bool_snap.yaml",
    "legacy_bool_extend.yaml",
    "legacy_bool_off.yaml",
    "rf_interval.yaml",
    "rf_precedence.yaml",
    "aux_streams.yaml",
    "kitchen_sink.yaml",
    "unknown_keys.yaml",
]}


def _yaml_path(name: str, tmp_path: Path) -> Path:
    if name in SYNTHETIC_YAMLS:
        p = tmp_path / name
        p.write_text(SYNTHETIC_YAMLS[name])
        return p
    return REPO_ROOT / name


ALL_YAMLS = REPO_YAMLS + list(SYNTHETIC_YAMLS)


@pytest.mark.parametrize("name", ALL_YAMLS, ids=[Path(p).name for p in ALL_YAMLS])
def test_yaml_equivalence_empty_namespace(name, tmp_path):
    """GUI route: legacy merge into an EMPTY namespace == load_yaml.  Exact."""
    path = _yaml_path(name, tmp_path)
    ns = load_pipeline(path, Namespace())
    via_loader = PipelineConfig.from_namespace(ns)
    via_compat = load_yaml(path)
    assert diff_paths(via_loader.model_dump(), via_compat.model_dump()) == set()
    assert via_loader.canonical_hash() == via_compat.canonical_hash()


@pytest.mark.parametrize("name", ALL_YAMLS, ids=[Path(p).name for p in ALL_YAMLS])
def test_yaml_equivalence_default_namespace(name, tmp_path):
    """CLI route: merge into the DEFAULT parser namespace carrying
    ``_explicit_cli=frozenset()`` (nothing typed) vs load_yaml.

    HOTFIX flip: with explicit-flag detection the loader honors every YAML
    value on any unset dest, so the SP1.1 gated divergences are gone.  The
    only remaining divergence is merge_overlap_strategy (ALWAYS_DIVERGENT),
    which Fix 2 removes by flipping the schema default.
    """
    path = _yaml_path(name, tmp_path)
    ns = copy.deepcopy(get_parser().parse_args([]))
    ns._explicit_cli = frozenset()  # CLI route: user typed nothing
    load_pipeline(path, ns)
    via_loader = PipelineConfig.from_namespace(ns)
    via_compat = load_yaml(path)

    expected = EXPECTED_GATED[name]
    diffs = diff_paths(via_loader.model_dump(), via_compat.model_dump())
    assert diffs == set(expected) | ALWAYS_DIVERGENT, (
        f"unexpected divergence set for {name}")

    dump_a, dump_b = via_loader.model_dump(), via_compat.model_dump()
    for dotted, (legacy_val, compat_val) in expected.items():
        section, field = dotted.split(".", 1)
        assert dump_a[section][field] == legacy_val, dotted
        assert dump_b[section][field] == compat_val, dotted
    # Fix 2: both routes now resolve the runtime-true 'dynamic'.
    assert dump_a["detection"]["merge_overlap_strategy"] == "dynamic"
    assert dump_b["detection"]["merge_overlap_strategy"] == "dynamic"


def test_yaml_to_dataclasses_roundtrip():
    """load_yaml output converts to the existing dataclass types cleanly."""
    cfg = load_yaml(REPO_ROOT / "test_pipeline.yaml")
    gaze, det, tracker, ray, depth, phen, out, proj = to_dataclasses(cfg)
    assert isinstance(gaze, GazeConfig)
    assert isinstance(det, DetectionConfig)
    assert isinstance(tracker, TrackerConfig)
    assert isinstance(ray, RayFormingConfig)
    assert isinstance(depth, DepthConfig)
    assert isinstance(phen, PhenomenaConfig)
    assert isinstance(out, OutputConfig)
    assert isinstance(proj, ProjectConfig)
    # Spot checks against the YAML contents.
    assert det.conf == 0.05
    assert gaze.gaze_tips is True and tracker.skip_frames == 1
    assert gaze.adaptive_ray == "extend" and ray.snap_mode == "extend"
    assert phen.joint_attention is True and phen.ja_window == 10
    assert out.heatmap_path == "heatmaps/"


def test_yaml_aux_streams_parsed(tmp_path):
    path = _yaml_path("aux_streams.yaml", tmp_path)
    cfg = load_yaml(path)
    assert cfg.output.aux_streams is not None
    a, b = cfg.output.aux_streams
    assert a.video_type is VideoType.EYE_ONLY
    assert b.participants == ["S1", "S2"] and b.auto_detect_faces is False
    out = to_dataclasses(cfg)[6]
    assert out.aux_streams == [
        AuxStreamConfig(source="eye.mp4", video_type=VideoType.EYE_ONLY,
                        stream_label="left_eye", participants=["S1"]),
        AuxStreamConfig(source="wide.mp4", video_type=VideoType.WIDE_CLOSEUP,
                        stream_label="room", participants=["S1", "S2"],
                        auto_detect_faces=False),
    ]


def test_yaml_rf_interval_precedence(tmp_path):
    legacy = load_yaml(_yaml_path("rf_interval.yaml", tmp_path))
    assert legacy.rayforming.min_call_gap == 10
    both = load_yaml(_yaml_path("rf_precedence.yaml", tmp_path))
    assert both.rayforming.min_call_gap == 45


def test_yaml_unknown_keys_silently_ignored(tmp_path):
    """Unknown keys neither raise nor leak -- matching the legacy loader
    (which also ignores them silently), while DIRECT construction with an
    unknown key raises (extra='forbid')."""
    path = _yaml_path("unknown_keys.yaml", tmp_path)
    cfg = load_yaml(path)  # must not raise
    assert cfg.detection.conf == 0.2
    assert cfg.gaze.ray_length == 1.5
    # Everything else stays at defaults.
    assert cfg.tracker == PipelineConfig().tracker
    assert cfg.output == PipelineConfig().output


def test_yaml_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_yaml(REPO_ROOT / "no_such_pipeline.yaml")
