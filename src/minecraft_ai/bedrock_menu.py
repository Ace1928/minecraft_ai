"""Compatibility shim. Implementation lives in `minecraft_ai.operator.menu`."""
from __future__ import annotations

import sys

from minecraft_ai.operator import menu as _impl

sys.modules[__name__] = _impl
