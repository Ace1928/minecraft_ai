"""Eidos OpenAI-compatible Cognition Daemon for minecraft_ai.

Serves as the high-level cognition brain for Minecraft AI, listening on loopback
(default: http://127.0.0.1:8080/v1) and responding with schema-adherent
_CognitionWireDecision payloads to guide the embodied player loop.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field

from minecraft_ai.cognition.types import _CognitionWireDecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [EidosDaemon] %(message)s",
)
logger = logging.getLogger("eidos_daemon")

MODEL_ID = "eidos-nexus-cognition-v1"

app = FastAPI(
    title="Eidos Minecraft Cognition Daemon",
    version="1.0.0",
    description="OpenAI-compatible local brain for minecraft_ai",
)

# Operational Metrics
_METRICS = {
    "started_at": time.time(),
    "total_completions": 0,
    "last_decision": None,
    "last_latency_ms": 0.0,
    "errors": 0,
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int | None = 256
    temperature: float | None = 0.0
    response_format: dict[str, Any] | None = None
    grammar: str | None = None
    thinking: dict[str, Any] | None = None


def eidos_reason(payload: dict[str, Any]) -> _CognitionWireDecision:
    """Eidosian recursive reasoning layer over normalized perception facts."""
    raw_facts = payload.get("fresh_facts", {})
    facts: dict[str, Any] = {}
    for k, v in raw_facts.items():
        facts[k] = v[0] if isinstance(v, list) and v else v

    operator_msgs = payload.get("operator_messages", [])
    recent_runs = payload.get("recent_skill_runs", [])

    health = facts.get("player.health", 20)
    hunger = facts.get("player.hunger", 20)
    oak_logs = facts.get("inventory.oak_log_count", 0)
    wood_planks = facts.get("inventory.oak_planks_count", 0)
    build_blocks = facts.get("inventory.build_blocks", 0)
    danger_immediate = facts.get("danger.immediate", False)
    target_block = facts.get("target.block_type") or facts.get("block.crosshair_type")
    target_visible = facts.get("target.visible", False)
    target_mineable = facts.get("target.mineable", False)

    # 1. Critical Survival Override
    if bool(facts.get("environment.underwater")) or bool(facts.get("player.submerged")) or bool(facts.get("danger.drowning")):
        return _CognitionWireDecision(
            r="Submerged underwater / air loss; swim to surface",
            g="survive",
            s="escape_submersion",
            p={"allow_jump": True},
            o="Submerged! Swimming to surface.",
            c=None,
            x=False,
            q=(),
            d="Swim up to surface and reach dry ground",
            n=("escape_submersion", "explore_forward"),
        )

    if danger_immediate or (isinstance(health, (int, float)) and health < 8):
        return _CognitionWireDecision(
            r="Immediate danger / critical health; disengage and retreat",
            g="survive",
            s="retreat_from_danger",
            p={},
            o="Danger detected! Retreating to safe ground.",
            c=None,
            x=False,
            q=(),
            d="Move away from hostile entities and stabilize",
            n=("retreat_from_danger", "establish_basic_shelter"),
        )

    # 2. Operator Directives (Instruction / Correction)
    if operator_msgs:
        active_op = operator_msgs[-1]
        op_text = active_op.get("text", "")
        op_id = active_op.get("message_id", "op-0")
        lower_text = op_text.lower()
        if any(kw in lower_text for kw in ("shelter", "place", "build", "pillar", "dirt")):
            return _CognitionWireDecision(
                r=f"Operator requested building / block placement ({op_text})",
                g=f"operator:{op_id}",
                s="place_block",
                p={"block": "dirt", "allow_use": True, "allow_jump": True},
                o="Executing operator directive: placing blocks.",
                c="Understood. Placing blocks.",
                x=False,
                q=(),
                d="Place block beneath feet / adjacent voxel to build or step up",
                n=("place_block", "explore_forward"),
            )
        if "wood" in lower_text or "tree" in lower_text or "mine" in lower_text:
            return _CognitionWireDecision(
                r=f"Operator requested wood gathering ({op_text})",
                g=f"operator:{op_id}",
                s="mine_visible_block" if target_visible and target_mineable else "explore_forward",
                p={"target": "oak_log"} if target_visible else {},
                o="Executing operator directive: harvesting wood.",
                c="Acknowledged. Locating and harvesting timber.",
                x=False,
                q=("target.block_type", "target.visible"),
                d="Harvest tree trunk at crosshair",
                n=("mine_visible_block", "collect_dropped_items"),
            )

    # 3. Progression Chain: Wood Logs -> Wood Planks -> Crafting Table
    if oak_logs >= 3 and wood_planks == 0:
        return _CognitionWireDecision(
            r=f"Acquired {oak_logs} oak logs; proceed to crafting wood planks",
            g="progress",
            s="craft_wood_planks",
            p={},
            o=f"Crafting wood planks from {oak_logs} harvested logs.",
            c=None,
            x=False,
            q=(),
            d="Open crafting 2x2 grid and convert logs to planks",
            n=("craft_wood_planks", "craft_crafting_table"),
        )
    if wood_planks >= 4 and facts.get("inventory.crafting_table", 0) == 0:
        return _CognitionWireDecision(
            r=f"Have {wood_planks} planks; crafting crafting table",
            g="progress",
            s="craft_crafting_table",
            p={},
            o="Crafting a workbench/crafting table.",
            c=None,
            x=False,
            q=(),
            d="Arrange 4 planks in grid to craft crafting table",
            n=("craft_crafting_table",),
        )

    # 4. Target Acquisition & Mining
    if target_visible and target_mineable and target_block in {"oak_log", "birch_log", "spruce_log", "log"}:
        return _CognitionWireDecision(
            r=f"Log aligned at crosshair ({target_block}); execute block mining",
            g="gather_materials",
            s="mine_visible_block",
            p={"target": str(target_block)},
            o=f"Mining visible {target_block}.",
            c=None,
            x=False,
            q=(),
            d=f"Hold attack on the {target_block} until broken",
            n=("mine_visible_block", "collect_dropped_items"),
        )

    # 5. Default Autonomous Exploration
    return _CognitionWireDecision(
        r="Exploring terrain to locate harvestable wood and resource nodes",
        g="explore",
        s="explore_forward",
        p={"allow_jump": True, "allow_attack": True, "allow_use": True},
        o="Exploring forward for trees and resources.",
        c=None,
        x=False,
        q=("target.block_type", "target.visible"),
        d="Walk forward across safe terrain toward foliage",
        n=("explore_forward", "approach_visible_target"),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "online",
        "service": "eidos_cognition_daemon",
        "model_id": MODEL_ID,
        "uptime_s": round(time.time() - _METRICS["started_at"], 2),
        "total_completions": _METRICS["total_completions"],
        "last_latency_ms": round(_METRICS["last_latency_ms"], 3),
        "errors": _METRICS["errors"],
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(_METRICS["started_at"]),
                "owned_by": "eidos-nexus",
            }
        ],
    }


@app.get("/props")
def get_props() -> dict[str, Any]:
    """Llama.cpp compatibility probe endpoint."""
    return {
        "build_info": "eidos-nexus-1.0",
        "model": MODEL_ID,
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}



@app.post("/v1/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        user_msg = next((m for m in reversed(req.messages) if m.role == "user"), None)
        payload: dict[str, Any] = {}
        if user_msg and user_msg.content:
            try:
                payload = json.loads(user_msg.content)
            except Exception:
                pass

        wire_decision = eidos_reason(payload)
        decision_json = wire_decision.model_dump_json()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        _METRICS["total_completions"] += 1
        _METRICS["last_latency_ms"] = latency_ms
        _METRICS["last_decision"] = wire_decision.model_dump()

        logger.info(
            "Decision emitted: skill=%s goal=%s latency=%.2fms",
            wire_decision.s,
            wire_decision.g,
            latency_ms,
        )

        return {
            "id": f"chatcmpl-eidos-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": decision_json,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_msg.content if user_msg else "") // 4,
                "completion_tokens": len(decision_json) // 4,
                "total_tokens": (len(user_msg.content if user_msg else "") + len(decision_json)) // 4,
            },
        }
    except Exception as exc:
        _METRICS["errors"] += 1
        logger.error("Error generating decision: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Eidos Cognition Daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    args = parser.parse_args()

    logger.info("Starting Eidos Cognition Daemon on http://%s:%d/v1", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
