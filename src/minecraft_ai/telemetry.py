"""Compatibility shim. Implementation lives in `minecraft_ai.operator.telemetry`."""
from __future__ import annotations

import sys

from minecraft_ai.operator import telemetry as _impl

sys.modules[__name__] = _impl
