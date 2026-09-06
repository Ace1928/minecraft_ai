"""Compatibility shim. Implementation lives in `minecraft_ai.agent.factory`."""
from __future__ import annotations

import sys

from minecraft_ai.agent import factory as _impl

sys.modules[__name__] = _impl
