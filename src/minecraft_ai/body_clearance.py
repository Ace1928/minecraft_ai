"""Compatibility shim. Implementation lives in `minecraft_ai.perception.body_clearance`."""
from __future__ import annotations

import sys

from minecraft_ai.perception import body_clearance as _impl

sys.modules[__name__] = _impl
