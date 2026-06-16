"""Tests for outputs/global_csv.py -- tidy-aware global + condition aggregation."""

import csv

from mindsight.outputs.global_csv import (
    GLOBAL_TABLES,
    generate_condition_csvs,
    generate_global_csv,
)

_SUMMARY_HEADER = ["video_name", "conditions", "phenomenon", "participant",
                   "partner", "object", "metric", "value"]


def _write(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def _make_two_summaries(csv_dir):
    _write(csv_dir / "a_summary.csv", _SUMMARY_HEADER, [
        ["a", "Cond1", "object_look_time", "P0", "", "cup", "frames_active", "5"],
        ["a", "Cond1", "joint_attention", "all", "", "", "frames_active", "10"],
    ])
    _write(csv_dir / "b_summary.csv", _SUMMARY_HEADER, [
        ["b", "Cond2", "object_look_time", "P1", "", "plate", "frames_active", "7"],
    ])


class TestGenerateGlobalCsv:

    def test_concat_header_once(self, tmp_path):
        _make_two_summaries(tmp_path)
        out = generate_global_csv(tmp_path, "_summary.csv", "Global_summary.csv")
        assert out == tmp_path / "Global_summary.csv"
        rows = _read(out)
        # 1 header + 3 data rows, header not repeated.
        assert rows[0] == _SUMMARY_HEADER
        assert len(rows) == 4
        assert rows.count(_SUMMARY_HEADER) == 1

    def test_empty_dir_returns_none(self, tmp_path):
        assert generate_global_csv(
            tmp_path, "_summary.csv", "Global_summary.csv") is None

    def test_excludes_existing_global(self, tmp_path):
        _make_two_summaries(tmp_path)
        # A pre-existing Global file must not be re-ingested.
        _write(tmp_path / "Global_summary.csv", _SUMMARY_HEADER, [
            ["x", "", "stale", "P0", "", "", "m", "99"]])
        out = generate_global_csv(tmp_path, "_summary.csv", "Global_summary.csv")
        flat = [c for row in _read(out) for c in row]
        assert "stale" not in flat

    def test_stream_suffix_isolation(self, tmp_path):
        # _summary.csv must not swallow _novel_salience_events.csv etc.
        _make_two_summaries(tmp_path)
        _write(tmp_path / "a_novel_salience_events.csv",
               ["video_name", "conditions", "frame", "participant"],
               [["a", "Cond1", "4", "P0"]])
        out = generate_global_csv(tmp_path, "_summary.csv", "Global_summary.csv")
        rows = _read(out)
        assert len(rows) == 4  # only the two summary files aggregated

    def test_events_suffix_case_distinct(self, tmp_path):
        # Capitalised _Events.csv (frame log) is distinct from lowercase
        # _novel_salience_events.csv streams.
        _write(tmp_path / "a_Events.csv",
               ["video_name", "conditions", "frame"], [["a", "", "6"]])
        _write(tmp_path / "a_novel_salience_events.csv",
               ["video_name", "conditions", "frame"], [["a", "", "4"]])
        out = generate_global_csv(tmp_path, "_Events.csv", "Global_Events.csv")
        rows = _read(out)
        assert len(rows) == 2
        assert rows[1] == ["a", "", "6"]


class TestGenerateConditionCsvs:

    def test_split_on_conditions(self, tmp_path):
        _make_two_summaries(tmp_path)
        gpath = generate_global_csv(
            tmp_path, "_summary.csv", "Global_summary.csv")
        cond_dir = tmp_path / "By Condition"
        written = generate_condition_csvs(gpath, cond_dir, "_summary.csv")
        names = sorted(p.name for p in written)
        assert names == ["Cond1_summary.csv", "Cond2_summary.csv"]
        c1 = _read(cond_dir / "Cond1_summary.csv")
        assert c1[0] == _SUMMARY_HEADER
        assert len(c1) == 3  # header + 2 Cond1 rows

    def test_multi_tag_video_in_each(self, tmp_path):
        _write(tmp_path / "m_summary.csv", _SUMMARY_HEADER, [
            ["m", "A|B", "joint_attention", "all", "", "", "frames_active", "3"],
        ])
        gpath = generate_global_csv(
            tmp_path, "_summary.csv", "Global_summary.csv")
        cond_dir = tmp_path / "By Condition"
        generate_condition_csvs(gpath, cond_dir, "_summary.csv")
        assert (cond_dir / "A_summary.csv").exists()
        assert (cond_dir / "B_summary.csv").exists()

    def test_no_conditions_returns_empty(self, tmp_path):
        _write(tmp_path / "n_summary.csv", _SUMMARY_HEADER, [
            ["n", "", "joint_attention", "all", "", "", "frames_active", "3"],
        ])
        gpath = generate_global_csv(
            tmp_path, "_summary.csv", "Global_summary.csv")
        assert generate_condition_csvs(
            gpath, tmp_path / "By Condition", "_summary.csv") == []


def test_global_tables_registry_shape():
    # Every entry is (per-video suffix, Global_ output name).
    for suffix, out_name in GLOBAL_TABLES:
        assert suffix.endswith(".csv")
        assert out_name.startswith("Global_")
    suffixes = [s for s, _ in GLOBAL_TABLES]
    assert "_summary.csv" in suffixes
    assert "_Events.csv" in suffixes
