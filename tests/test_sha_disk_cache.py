"""Persistent weight-hash cache (v1.1 W2.5).

The (path, size, mtime_ns) sha256 cache in outputs.provenance now persists to
the per-user state dir, so the first preflight of a NEW app launch no longer
re-hashes unchanged multi-hundred-MB weights (the cold-start GUI freeze).
MINDSIGHT_NO_HASH_CACHE=1 disables persistence; a corrupt cache file is
ignored; touching a file invalidates its entry.
"""

import json
import os

import pytest

import mindsight.outputs.provenance as prov


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINDSIGHT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("MINDSIGHT_NO_HASH_CACHE", raising=False)
    _reset()
    yield tmp_path / "state"
    _reset()


def _reset():
    prov._SHA_CACHE.clear()
    prov._sha_disk_loaded = False


def _computes():
    return prov._sha_compute_count


def test_cache_survives_process_restart(state_dir, tmp_path):
    f = tmp_path / "weight.bin"
    f.write_bytes(b"x" * 4096)

    d1 = prov._sha256_file(f)
    assert (state_dir / prov._SHA_CACHE_FILENAME).is_file()
    baseline = _computes()

    _reset()                      # simulate a fresh process
    d2 = prov._sha256_file(f)
    assert d2 == d1
    assert _computes() == baseline   # served from disk, no re-hash


def test_touch_invalidates_entry(state_dir, tmp_path):
    f = tmp_path / "weight.bin"
    f.write_bytes(b"x" * 4096)
    prov._sha256_file(f)
    baseline = _computes()

    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    _reset()
    prov._sha256_file(f)
    assert _computes() == baseline + 1   # mtime moved -> genuine re-hash


def test_no_hash_cache_env_disables_persistence(state_dir, tmp_path,
                                                monkeypatch):
    monkeypatch.setenv("MINDSIGHT_NO_HASH_CACHE", "1")
    f = tmp_path / "weight.bin"
    f.write_bytes(b"x" * 4096)
    prov._sha256_file(f)
    assert not (state_dir / prov._SHA_CACHE_FILENAME).exists()


def test_corrupt_cache_file_is_ignored(state_dir, tmp_path):
    f = tmp_path / "weight.bin"
    f.write_bytes(b"x" * 4096)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / prov._SHA_CACHE_FILENAME).write_text("{not json")

    digest = prov._sha256_file(f)     # must not raise
    assert len(digest) == 64
    # and the corrupt file was replaced with a valid one
    data = json.loads((state_dir / prov._SHA_CACHE_FILENAME).read_text())
    assert any(str(f) in k for k in data)
