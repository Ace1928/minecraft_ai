from __future__ import annotations

import json
import time

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionContext, HighLevelController
from minecraft_ai.models import ModelMessage, ModelResponse
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.roles import get_role


class _ShelterSelectingModel:
    model_id = "contract-test"

    def complete(self, messages: tuple[ModelMessage, ...]) -> ModelResponse:
        return self.complete_structured(messages, name="decision", schema={})

    def complete_structured(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        del messages, name, schema
        return ModelResponse(
            text=json.dumps(
                {
                    "reasoning_summary": "Build shelter now",
                    "chosen_goal_id": "survive",
                    "skill_id": "establish_basic_shelter",
                    "skill_parameters": {},
                    "say": None,
                    "request_replan": False,
                    "ask_perception": [],
                    "research_query": None,
                }
            ),
            model=self.model_id,
            latency_ms=1.0,
        )


def _context() -> CognitionContext:
    return CognitionContext(
        role=get_role("generalist"),
        goals=(),
        memories=(),
        promises=(),
        wiki=(),
    )


def _board(*facts: PerceptionFact) -> PerceptionBlackboard:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=time.monotonic_ns(),
            instance_id="bedrock:test",
            width=1280,
            height=720,
            facts=facts,
        )
    )
    return board


def test_high_level_cannot_execute_unobserved_skill_preconditions() -> None:
    controller = HighLevelController(_ShelterSelectingModel(), build_bootstrap_skill_library())

    decision = controller.decide(_board(), _context())

    assert decision.skill_id is None
    assert decision.request_replan is True
    assert decision.ask_perception == ("inventory.build_blocks",)


def test_high_level_can_execute_observably_feasible_skill() -> None:
    now = time.monotonic_ns()
    build_blocks = PerceptionFact(
        key="inventory.build_blocks",
        value=20,
        confidence=0.95,
        observed_ns=now,
        source="vlm:test",
        expires_after_ms=10_000,
    )
    controller = HighLevelController(_ShelterSelectingModel(), build_bootstrap_skill_library())

    decision = controller.decide(_board(build_blocks), _context())

    assert decision.skill_id == "establish_basic_shelter"
    assert decision.request_replan is False
