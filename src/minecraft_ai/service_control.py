"""Compatibility shim. Implementation lives in `minecraft_ai.operator.service_control`."""
from __future__ import annotations

import sys

from minecraft_ai.operator import service_control as _impl

sys.modules[__name__] = _impl
