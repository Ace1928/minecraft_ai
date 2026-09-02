"""Per-instance Minecraft capture/input bridge contracts."""

from .protocol import (
    PROTOCOL_VERSION,
    BridgeAck,
    BridgeCapability,
    BridgeHello,
    ChatEvent,
    FrameDescriptor,
    Heartbeat,
    InputCommand,
    InstanceIdentity,
    ReleaseAll,
)

__all__ = [
    "PROTOCOL_VERSION",
    "BridgeAck",
    "BridgeCapability",
    "BridgeHello",
    "ChatEvent",
    "FrameDescriptor",
    "Heartbeat",
    "InputCommand",
    "InstanceIdentity",
    "ReleaseAll",
]
