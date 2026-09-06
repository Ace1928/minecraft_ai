"""Compatibility shim. Implementation lives in `minecraft_ai.control.execution`."""
from __future__ import annotations

import sys

from minecraft_ai.control import execution as _impl

sys.modules[__name__] = _impl
