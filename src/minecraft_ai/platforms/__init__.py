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

__all__ = [
    "BedrockBuild",
    "BedrockLinuxInstall",
    "BedrockLinuxInstance",
    "discover_bedrock_linux_install",
    "find_bedrock_linux_instances",
]
