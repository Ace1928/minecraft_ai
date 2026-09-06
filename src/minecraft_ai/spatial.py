"""Compatibility shim. Implementation lives in `minecraft_ai.memory.spatial`."""
from __future__ import annotations

import sys

from minecraft_ai.memory import spatial as _impl

sys.modules[__name__] = _impl
