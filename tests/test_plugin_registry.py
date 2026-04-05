"""Tests for Plugins/__init__.py -- PluginRegistry and plugin lifecycle."""

import pytest

from Plugins import (
    GazePlugin,
    PhenomenaPlugin,
    PluginRegistry,
)


class TestPluginRegistry:

    def test_register_and_get(self):
        reg = PluginRegistry()

        class FakePlugin:
            name = "test_plugin"

        reg.register(FakePlugin)
        assert reg.get("test_plugin") is FakePlugin

    def test_register_empty_name_raises(self):
        reg = PluginRegistry()

        class BadPlugin:
            name = ""

        with pytest.raises(ValueError):
            reg.register(BadPlugin)

    def test_get_missing_raises(self):
        reg = PluginRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_names_returns_sorted(self):
        reg = PluginRegistry()

        class P1:
            name = "charlie"

        class P2:
            name = "alpha"

        class P3:
            name = "bravo"

        reg.register(P1)
        reg.register(P2)
        reg.register(P3)
        assert reg.names() == ["alpha", "bravo", "charlie"]

    def test_contains(self):
        reg = PluginRegistry()

        class FakePlugin:
            name = "exists"

        reg.register(FakePlugin)
        assert "exists" in reg
        assert "nope" not in reg

    def test_register_as_decorator(self):
        reg = PluginRegistry()

        @reg.register
        class DecoratedPlugin:
            name = "decorated"

        assert "decorated" in reg
        assert reg.get("decorated") is DecoratedPlugin

    def test_discover_nonexistent_dir(self, tmp_path):
        """Discover should no-op when directory doesn't exist."""
        reg = PluginRegistry()
        reg.discover(tmp_path / "nonexistent")
        assert reg.names() == []

    def test_discover_skips_underscore_dirs(self, tmp_path):
        """Directories starting with _ should be skipped."""
        reg = PluginRegistry()
        hidden = tmp_path / "_hidden"
        hidden.mkdir()
        (hidden / "__init__.py").write_text("PLUGIN_CLASS = None\n")
        reg.discover(tmp_path)
        assert reg.names() == []


class TestPluginBaseClasses:

    def test_gaze_plugin_has_mode(self):
        """GazePlugin subclass must declare mode."""
        assert hasattr(GazePlugin, 'mode')

    def test_phenomena_plugin_interface(self):
        """PhenomenaPlugin has the expected abstract interface."""
        assert hasattr(PhenomenaPlugin, 'update')
        assert hasattr(PhenomenaPlugin, 'csv_rows')
        assert hasattr(PhenomenaPlugin, 'draw_frame')

    def test_phenomena_plugin_has_name(self):
        assert hasattr(PhenomenaPlugin, 'name')


class TestPluginDiscovery:

    def test_real_gaze_registry_discovers_backends(self):
        """The actual gaze_registry should find at least mgaze and l2cs."""
        from Plugins import gaze_registry
        names = gaze_registry.names()
        assert "mgaze" in names
        assert "l2cs" in names

    def test_real_phenomena_registry_has_entries(self):
        """The actual phenomena_registry should discover at least one tracker."""
        from Plugins import phenomena_registry
        names = phenomena_registry.names()
        assert len(names) >= 1
