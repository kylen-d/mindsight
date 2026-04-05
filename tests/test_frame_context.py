"""Tests for pipeline_config.FrameContext -- the mutable per-frame data carrier."""

import numpy as np
import pytest

from ms.pipeline_config import FrameContext


class TestFrameContextInit:
    """Tests for FrameContext construction."""

    def test_default_init(self):
        """Default construction sets frame=None and frame_no=0."""
        ctx = FrameContext()
        assert ctx['frame'] is None
        assert ctx['frame_no'] == 0

    def test_init_with_positional_args(self):
        """Positional frame and frame_no are stored."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = FrameContext(frame, 42)
        assert ctx['frame'] is frame
        assert ctx['frame_no'] == 42

    def test_init_with_kwargs(self):
        """Extra keyword arguments flow into the context."""
        ctx = FrameContext(frame=None, frame_no=1, objects=[], hits=set())
        assert ctx['objects'] == []
        assert ctx['hits'] == set()

    def test_init_kwargs_override_nothing(self):
        """Kwargs do not conflict with the explicit frame/frame_no params."""
        ctx = FrameContext(frame="img", frame_no=10, extra="value")
        assert ctx['frame'] == "img"
        assert ctx['frame_no'] == 10
        assert ctx['extra'] == "value"


class TestFrameContextGetSetItem:
    """Tests for __getitem__ and __setitem__."""

    def test_setitem_and_getitem(self):
        """Setting and getting a key works."""
        ctx = FrameContext()
        ctx['detections'] = [1, 2, 3]
        assert ctx['detections'] == [1, 2, 3]

    def test_overwrite_existing_key(self):
        """Overwriting an existing key replaces the value."""
        ctx = FrameContext(frame_no=5)
        ctx['frame_no'] = 99
        assert ctx['frame_no'] == 99

    def test_getitem_missing_key_raises(self):
        """Accessing a missing key raises KeyError."""
        ctx = FrameContext()
        with pytest.raises(KeyError):
            _ = ctx['nonexistent']

    def test_set_none_value(self):
        """None is a valid value."""
        ctx = FrameContext()
        ctx['result'] = None
        assert ctx['result'] is None


class TestFrameContextContains:
    """Tests for __contains__."""

    def test_contains_existing_key(self):
        """'frame' is always present after init."""
        ctx = FrameContext()
        assert 'frame' in ctx
        assert 'frame_no' in ctx

    def test_contains_missing_key(self):
        """Missing key returns False."""
        ctx = FrameContext()
        assert 'missing' not in ctx

    def test_contains_after_setitem(self):
        """Key is present after being set."""
        ctx = FrameContext()
        assert 'x' not in ctx
        ctx['x'] = 42
        assert 'x' in ctx


class TestFrameContextGet:
    """Tests for the get() method."""

    def test_get_existing_key(self):
        """get() returns the value for an existing key."""
        ctx = FrameContext(frame_no=7)
        assert ctx.get('frame_no') == 7

    def test_get_missing_key_default_none(self):
        """get() returns None by default for missing keys."""
        ctx = FrameContext()
        assert ctx.get('missing') is None

    def test_get_missing_key_custom_default(self):
        """get() returns the provided default for missing keys."""
        ctx = FrameContext()
        assert ctx.get('missing', 42) == 42

    def test_get_existing_key_ignores_default(self):
        """get() ignores the default when the key exists."""
        ctx = FrameContext()
        ctx['val'] = 10
        assert ctx.get('val', 999) == 10


class TestFrameContextUpdate:
    """Tests for the update() method."""

    def test_update_adds_new_keys(self):
        """update() adds new keys from a mapping."""
        ctx = FrameContext()
        ctx.update({'a': 1, 'b': 2})
        assert ctx['a'] == 1
        assert ctx['b'] == 2

    def test_update_overwrites_existing(self):
        """update() overwrites existing keys."""
        ctx = FrameContext(frame_no=0)
        ctx.update({'frame_no': 100})
        assert ctx['frame_no'] == 100

    def test_update_preserves_unmentioned_keys(self):
        """Keys not in the mapping are preserved."""
        ctx = FrameContext(frame="img", frame_no=1)
        ctx.update({'frame_no': 2})
        assert ctx['frame'] == "img"


class TestFrameContextAsKwargs:
    """Tests for the as_kwargs() method."""

    def test_as_kwargs_returns_dict(self):
        """as_kwargs() returns a plain dict."""
        ctx = FrameContext(frame="img", frame_no=5)
        kw = ctx.as_kwargs()
        assert isinstance(kw, dict)
        assert kw['frame'] == "img"
        assert kw['frame_no'] == 5

    def test_as_kwargs_is_shallow_copy(self):
        """Mutating the returned dict does not affect the context."""
        ctx = FrameContext(frame_no=1)
        kw = ctx.as_kwargs()
        kw['frame_no'] = 999
        assert ctx['frame_no'] == 1

    def test_as_kwargs_includes_extra_keys(self):
        """Extra kwargs from init appear in as_kwargs()."""
        ctx = FrameContext(frame=None, frame_no=0, objects=[1, 2])
        kw = ctx.as_kwargs()
        assert kw['objects'] == [1, 2]

    def test_as_kwargs_after_mutations(self):
        """as_kwargs() reflects all mutations."""
        ctx = FrameContext()
        ctx['hits'] = {(0, 1)}
        ctx.update({'dets': ['a']})
        kw = ctx.as_kwargs()
        assert kw['hits'] == {(0, 1)}
        assert kw['dets'] == ['a']

    def test_kwargs_unpacking(self):
        """as_kwargs() output can be used with **kwargs unpacking."""
        ctx = FrameContext(frame="img", frame_no=3, extra="val")

        def receiver(**kwargs):
            return kwargs

        result = receiver(**ctx.as_kwargs())
        assert result['frame'] == "img"
        assert result['frame_no'] == 3
        assert result['extra'] == "val"
