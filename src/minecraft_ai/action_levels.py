"""Compatibility shim. Implementation lives in `minecraft_ai.control.action_levels`."""
from __future__ import annotations

import sys

from minecraft_ai.control import action_levels as _impl

sys.modules[__name__] = _impl
