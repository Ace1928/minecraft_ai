"""Compatibility shim. Implementation lives in `minecraft_ai.control.outcomes`."""
from __future__ import annotations

import sys

from minecraft_ai.control import outcomes as _impl

sys.modules[__name__] = _impl
