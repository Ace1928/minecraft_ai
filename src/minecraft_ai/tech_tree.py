"""Compatibility shim. Implementation lives in `minecraft_ai.skills.tech_tree`."""
from __future__ import annotations

import sys

from minecraft_ai.skills import tech_tree as _impl

sys.modules[__name__] = _impl
