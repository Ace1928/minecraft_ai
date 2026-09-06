"""Compatibility shim. Implementation lives in `minecraft_ai.agent.daemon_executor`."""
from __future__ import annotations

import sys

from minecraft_ai.agent import daemon_executor as _impl

sys.modules[__name__] = _impl
