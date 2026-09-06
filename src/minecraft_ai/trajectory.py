"""Compatibility shim. Implementation lives in `minecraft_ai.memory.trajectory`."""
from __future__ import annotations

import sys

from minecraft_ai.memory import trajectory as _impl

sys.modules[__name__] = _impl
