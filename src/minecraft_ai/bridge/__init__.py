"""Per-instance Minecraft capture/input bridge contracts."""

from .client import BridgeEndpoint, ScopedBridgeBackend
from .protocol import (
    PROTOCOL_VERSION,
    Authenticate,
    BridgeAck,
    BridgeCapability,
    BridgeError,
    BridgeHello,
    ChatEvent,
    FrameDescriptor,
    Heartbeat,
    InputCommand,
    InstanceIdentity,
    LeaseBind,
    LeaseClear,
    ReleaseAll,
)

__all__ = [
    "PROTOCOL_VERSION",
    "Authenticate",
    "BridgeAck",
    "BridgeCapability",
    "BridgeEndpoint",
    "BridgeError",
    "BridgeHello",
    "ChatEvent",
    "FrameDescriptor",
    "Heartbeat",
    "InputCommand",
    "InstanceIdentity",
    "LeaseBind",
    "LeaseClear",
    "ReleaseAll",
    "ScopedBridgeBackend",
]
