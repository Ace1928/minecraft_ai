"""Compatibility shim. Implementation lives in `minecraft_ai.control.policy_timing`."""
from __future__ import annotations

import sys

from minecraft_ai.control import policy_timing as _impl

sys.modules[__name__] = _impl
