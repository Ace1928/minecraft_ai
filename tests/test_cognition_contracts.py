from __future__ import annotations

import json
import time

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionContext, HighLevelController
from minecraft_ai.models import ModelMessage, ModelResponse
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.roles import get_role
from minecraft_ai.social import OperatorMessage, OperatorMessageStatus


class _ShelterSelectingModel:
    model_id = "contract-test"

    def __init__(self, *, chosen_goal_id: str = "survive", say: str | None = None) -> None:
        self.chosen_goal_id = chosen_goal_id
        self.say = say

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
                    "chosen_goal_id": self.chosen_goal_id,
                    "skill_id": "establish_basic_shelter",
                    "skill_parameters": {},
                    "say": self.say,
                    "request_replan": False,
                    "ask_perception": [],
                    "research_query": None,
                }
            ),
            model=self.model_id,
            latency_ms=1.0,
        )


class _CapturingModel(_ShelterSelectingModel):
    def __init__(self) -> None:
        super().__init__(chosen_goal_id="role:generalist:0:survive")
        self.initial_messages: tuple[ModelMessage, ...] = ()

    def complete_structured(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        if not self.initial_messages:
            self.initial_messages = messages
        del name, schema
        return ModelResponse(
            text=json.dumps(
                {
                    "reasoning_summary": "Traverse the hill for the operator",
                    "chosen_goal_id": self.chosen_goal_id,
                    "skill_id": "explore_forward",
                    "skill_parameters": {},
                    "say": "I am climbing the hill now.",
                    "request_replan": False,
                    "ask_perception": [],
                    "research_query": None,
                }
            ),
            model=self.model_id,
            latency_ms=1.0,
        )


class _RepairingModel(_ShelterSelectingModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete_structured(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return super().complete_structured(messages, name=name, schema=schema)
        return ModelResponse(
            text=json.dumps(
                {
                    "reasoning_summary": "Use the feasible learned exploration option",
                    "chosen_goal_id": "survive",
                    "skill_id": "explore_forward",
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


def test_high_level_repairs_infeasible_choice_with_model_selected_feasible_option() -> None:
    model = _RepairingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    decision = controller.decide(_board(), _context())

    assert decision.skill_id == "explore_forward"
    assert model.calls == 2
    assert controller.metrics.repairs == 1
    assert controller.metrics.last_error is None


def test_private_cognition_cannot_leak_into_game_chat() -> None:
    controller = HighLevelController(
        _ShelterSelectingModel(say="private chain of thought"),
        build_bootstrap_skill_library(),
    )

    decision = controller.decide(_board(), _context())

    assert decision.say is None


def test_explicit_operator_reply_remains_available_as_social_output() -> None:
    message = OperatorMessage(
        message_id="message-1",
        created_ns=time.time_ns(),
        text="What are you doing?",
    )
    context = _context()
    context.operator_messages = (message,)
    controller = HighLevelController(
        _ShelterSelectingModel(chosen_goal_id="operator:message-1", say="Gathering wood."),
        build_bootstrap_skill_library(),
    )

    decision = controller.decide(_board(), context)

    assert decision.say == "Gathering wood."


def test_high_level_receives_explicit_active_operator_correction() -> None:
    older = OperatorMessage(
        message_id="old",
        created_ns=1,
        text="Keep gathering logs",
    )
    correction = OperatorMessage(
        message_id="correction",
        created_ns=2,
        text="Stop and climb the hill",
    )
    context = _context()
    context.operator_messages = (correction, older)
    model = _CapturingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    decision = controller.decide(_board(), context)

    payload = json.loads(model.initial_messages[1].content)
    assert payload["active_operator_message"]["message_id"] == "correction"
    assert "Stop and climb the hill" in model.initial_messages[-1].content
    assert "explore_forward" in {skill["skill_id"] for skill in payload["skills"]}
    assert "establish_basic_shelter" not in {
        skill["skill_id"] for skill in payload["skills"]
    }
    assert (
        "must be addressed before any conflicting standing goal"
        in model.initial_messages[0].content
    )
    assert decision.chosen_goal_id == "operator:correction"
    assert decision.say == "I am climbing the hill now."


def test_prior_agent_response_is_not_replayed_as_operator_instruction() -> None:
    message = OperatorMessage(
        message_id="current",
        created_ns=2,
        text="Leave the pit and traverse open ground",
        status=OperatorMessageStatus.ACKNOWLEDGED,
        response_text="Restart the old gather_nearby_wood task",
    )
    context = _context()
    context.operator_messages = (message,)
    model = _CapturingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    controller.decide(_board(), context)

    payload = json.loads(model.initial_messages[1].content)
    assert payload["active_operator_message"]["text"] == message.text
    assert "response_text" not in payload["active_operator_message"]
    assert "Restart the old" not in model.initial_messages[-1].content
