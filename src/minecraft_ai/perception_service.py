"""Compatibility shim. Implementation lives in `minecraft_ai.perception.service`."""
from __future__ import annotations

import sys

from minecraft_ai.perception import service as _impl

sys.modules[__name__] = _impl
