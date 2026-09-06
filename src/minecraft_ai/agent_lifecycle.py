"""Compatibility shim. Implementation lives in `minecraft_ai.agent.lifecycle`."""
from __future__ import annotations

import sys

from minecraft_ai.agent import lifecycle as _impl

sys.modules[__name__] = _impl
