"""Compatibility shim. Implementation lives in `minecraft_ai.control.crafting`."""
from __future__ import annotations

import sys

from minecraft_ai.control import crafting as _impl

sys.modules[__name__] = _impl
