"""Business integration adapter for paid player directions.

Provides a direct Python client adapter wrapping DirectionsGateway to integrate
seamlessly with private business accounting (such as neuroforge_business).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .gateway import DirectionsGateway, SUPPORTED_INSTRUCTIONS
from .models import (
    DirectionsDiscovery,
    DirectionsReceipt,
    DirectionsRequest,
    DirectionsState,
)


class MinecraftDirectionsAdapter:
    """Client adapter for executing and verifying paid Minecraft player directions."""

    identity: str = "minecraft_bedrock_live_v1"

    def __init__(
        self,
        state_db_path: str | Path = "/home/lloyd/.local/share/minecraft-ai/state.sqlite3",
        *,
        gateway: DirectionsGateway | None = None,
    ) -> None:
        self.gateway = gateway or DirectionsGateway(state_db_path)

    def discover(self) -> DirectionsDiscovery:
        """Query agent readiness, supervisor epoch, and supported instruction catalog."""
        return self.gateway.discover()

    def validate(self, text: str) -> str:
        """Validate that the submitted text matches an admitted bounded instruction."""
        text = text.strip()
        if not text:
            raise ValueError("Direction text cannot be empty.")
        if len(text) > 256:
            raise ValueError("Direction text exceeds maximum allowed length of 256 characters.")

        # Find matching supported instruction
        for spec in SUPPORTED_INSTRUCTIONS:
            if spec.canonical_text.lower() in text.lower() or spec.instruction_id in text.lower():
                return spec.instruction_id
            if "open inventory" in text.lower() and "close" in text.lower():
                return "open_observe_close_inventory"
            if "open inventory" in text.lower():
                return "observe_inventory"
            if "explore forward" in text.lower():
                return "explore_forward"

        raise ValueError(
            f"Text {text!r} does not match any admitted bounded instruction in the catalog."
        )

    def enqueue(
        self,
        *,
        request_id: str,
        member_id: str,
        server_id: str,
        expected_session_id: str,
        expected_epoch: int,
        text: str,
        deadline_s: float = 60.0,
        arguments: dict[str, Any] | None = None,
    ) -> DirectionsReceipt:
        """Enqueue an authenticated player direction with full idempotency and quota checks."""
        instruction_id = self.validate(text)
        request = DirectionsRequest(
            request_id=request_id,
            member_id=member_id,
            server_id=server_id,
            expected_session_id=expected_session_id,
            expected_epoch=expected_epoch,
            instruction_id=instruction_id,
            instruction_text=text,
            arguments=arguments or {},
            deadline_s=deadline_s,
        )
        return self.gateway.submit(request)

    def poll_outcome(
        self,
        request_id: str,
        *,
        member_id: str | None = None,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.5,
    ) -> DirectionsReceipt:
        """Synchronously poll until terminal outcome or timeout."""
        start = time.time()
        while time.time() - start < timeout_s:
            receipt = self.gateway.status(request_id, member_id=member_id)
            if receipt.state in {
                DirectionsState.SUCCEEDED,
                DirectionsState.FAILED,
                DirectionsState.CANCELLED,
                DirectionsState.EXPIRED,
            }:
                return receipt
            time.sleep(poll_interval_s)
        return self.gateway.status(request_id, member_id=member_id)

    def cancel(self, request_id: str, *, member_id: str | None = None) -> DirectionsReceipt:
        """Cancel a pending direction."""
        return self.gateway.cancel(request_id, member_id=member_id)
