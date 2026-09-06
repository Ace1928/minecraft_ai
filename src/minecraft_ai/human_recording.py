"""Compatibility shim. Implementation lives in `minecraft_ai.operator.human_recording`."""
from __future__ import annotations

import sys

from minecraft_ai.operator import human_recording as _impl

sys.modules[__name__] = _impl
