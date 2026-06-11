"""SP1.6: MGaze is a core backend resolved without the plugin registry."""


def test_backends_absent_from_registry():
    from Plugins import gaze_registry
    names = gaze_registry.names()
    assert "mgaze" not in names
    assert "l2cs" not in names


def test_explicit_mgaze_resolves_to_core_class():
    from mindsight.GazeTracking.Backends.MGaze.MGaze_Tracking import MGazePlugin
    from mindsight.GazeTracking.gaze_factory import create_gaze_engine
    eng = create_gaze_engine(backend="mgaze", engine=None)
    assert isinstance(eng, MGazePlugin)
