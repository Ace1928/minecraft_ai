"""Per-instance Minecraft capture/input bridge contracts."""

from .client import BridgeEndpoint, ScopedBridgeBackend
from .discovery import (
    BridgeDiscoveryError,
    DiscoveredBridge,
    discover_bridge_descriptors,
    select_bridge,
)
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
    "BridgeDiscoveryError",
    "BridgeEndpoint",
    "BridgeError",
    "BridgeHello",
    "ChatEvent",
    "DiscoveredBridge",
    "FrameDescriptor",
    "Heartbeat",
    "InputCommand",
    "InstanceIdentity",
    "LeaseBind",
    "LeaseClear",
    "ReleaseAll",
    "ScopedBridgeBackend",
    "discover_bridge_descriptors",
    "select_bridge",
]
