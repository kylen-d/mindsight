"""[SP1.5 shim] moved to mindsight.GazeTracking.Backends.L2CS.L2CS_Tracking; delete in SP1.6.

Star-import (not a sys.modules alias): Plugins/__init__.py discover()
loads this file by path and reads PLUGIN_CLASS off the resulting module object."""
from mindsight.GazeTracking.Backends.L2CS.L2CS_Tracking import *  # noqa: F401,F403
from mindsight.GazeTracking.Backends.L2CS.L2CS_Tracking import PLUGIN_CLASS  # noqa: F401
