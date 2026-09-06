"""Compatibility shim. Implementation lives in `minecraft_ai.perception.grounded`."""
from __future__ import annotations

import sys

from minecraft_ai.perception import grounded as _impl

sys.modules[__name__] = _impl
