"""Platform/runtime adapters.

Bedrock Edition running on Linux through a Wine/WineGDK launcher is the
reference runtime. Java Edition adapters remain optional compatibility paths.
"""

from .bedrock_linux import (
    BedrockBuild,
    BedrockLinuxInstall,
    BedrockLinuxInstance,
    discover_bedrock_linux_install,
    find_bedrock_linux_instances,
)
from .bedrock_x11 import (
    CapturedFrame,
    IsolatedX11Capture,
    IsolatedX11InputBackend,
    IsolationError,
    find_minecraft_window,
    require_isolated_display,
)

__all__ = [
    "BedrockBuild",
    "BedrockLinuxInstall",
    "BedrockLinuxInstance",
    "CapturedFrame",
    "IsolatedX11Capture",
    "IsolatedX11InputBackend",
    "IsolationError",
    "discover_bedrock_linux_install",
    "find_bedrock_linux_instances",
    "find_minecraft_window",
    "require_isolated_display",
]
