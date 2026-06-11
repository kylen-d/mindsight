"""[SP1.5 shim] contract tests for the ms/ -> mindsight/ re-export shims.

SP1.6 deletes the ms/ shim tree and this file with it. Fast, no model loads.
"""


def test_module_shim_identity():
    """Template M: the ms shim IS the moved mindsight module (sys.modules alias)."""
    import ms.pipeline_config
    import mindsight.pipeline_config
    assert ms.pipeline_config is mindsight.pipeline_config


def test_package_shim_reexport():
    """Template P: package shims re-export the moved package's public names."""
    from ms.PostProcessing.RayForming import RayFormingConfig as A
    from mindsight.PostProcessing.RayForming import RayFormingConfig as B
    assert A is B


def test_backend_discovery_intact():
    """T1/Template B: the Backends discover still registers mgaze + l2cs."""
    from Plugins import gaze_registry
    assert {"mgaze", "l2cs"} <= set(gaze_registry.names())


def test_user_plugin_contract():
    """The documented third-party import surface stays importable via shims."""
    from ms.pipeline_config import resolve_display_pid  # noqa: F401
    from ms.cli import main  # noqa: F401
