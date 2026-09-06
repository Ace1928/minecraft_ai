"""Compatibility shim. Implementation lives in `minecraft_ai.skills.builtin`."""
from __future__ import annotations

import sys

from minecraft_ai.skills import builtin as _impl

sys.modules[__name__] = _impl
