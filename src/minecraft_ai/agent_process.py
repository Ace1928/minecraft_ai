"""Compatibility shim. Implementation lives in `minecraft_ai.agent.process`."""
from __future__ import annotations

import sys

from minecraft_ai.agent import process as _impl

if __name__ != "__main__":
    sys.modules[__name__] = _impl
else:
    raise SystemExit(_impl.main())
