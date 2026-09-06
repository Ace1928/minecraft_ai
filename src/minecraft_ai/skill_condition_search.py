"""Compatibility shim. Implementation lives in `minecraft_ai.skills.conditions`."""
from __future__ import annotations

import sys

from minecraft_ai.skills import conditions as _impl

sys.modules[__name__] = _impl
