from __future__ import annotations

import json
import time

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionContext, CognitionDecision, HighLevelController
from minecraft_ai.models import ModelMessage, ModelResponse
from minecraft_ai.perception import (
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.roles import get_role
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
from minecraft_ai.skills import SkillLibrary, SkillOutcome, SkillRun


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


class _IdleCapturingModel(_ShelterSelectingModel):
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
                    "reasoning_summary": "No option selected yet",
                    "chosen_goal_id": "role:generalist:0:survive",
                    "skill_id": None,
                    "skill_parameters": {},
                    "say": None,
                    "request_replan": True,
                    "ask_perception": ["terrain.safe_direction"],
                    "research_query": None,
                }
            ),
            model=self.model_id,
            latency_ms=1.0,
        )


class _WireCapturingModel(_ShelterSelectingModel):
    def __init__(self) -> None:
        super().__init__()
        self.schema: dict[str, object] = {}

    def complete_structured(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        del messages, name
        self.schema = schema
        return ModelResponse(
            text=json.dumps(
                {
                    "r": "Explore while gathering evidence",
                    "g": "role:generalist:0:survive",
                    "s": "explore_forward",
                    "p": {},
                    "o": None,
                    "c": None,
                    "x": False,
                    "q": [],
                    "w": None,
                }
            ),
            model=self.model_id,
            latency_ms=1.0,
        )


class _GrammarCapturingModel:
    model_id = "grammar-contract-test"

    def __init__(
        self,
        *,
        skill_id: str = "explore_forward",
        chosen_goal_id: str = "operator:move-now",
    ) -> None:
        self.grammar = ""
        self.messages: tuple[ModelMessage, ...] = ()
        self.skill_id = skill_id
        self.chosen_goal_id = chosen_goal_id

    def complete_constrained(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
        grammar: str,
    ) -> ModelResponse:
        del name, schema
        self.messages = messages
        self.grammar = grammar
        return ModelResponse(
            text=json.dumps(
                {
                    "r": "Follow the direct movement instruction",
                    "g": self.chosen_goal_id,
                    "s": self.skill_id,
                    "p": {},
                    "o": None,
                    "c": None,
                    "x": False,
                    "q": [],
                    "w": None,
                    "d": "move forward through safe open terrain",
                    "n": ["move forward"],
                }
            ),
            model=self.model_id,
            latency_ms=1.0,
        )


class _TargetSelectingModel(_ShelterSelectingModel):
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
                    "reasoning_summary": "Find the selected target",
                    "chosen_goal_id": "old-goal",
                    "skill_id": "reacquire_target",
                    "skill_parameters": {"target": "selected terrain"},
                    "say": None,
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


class _RetryRepairingModel(_ShelterSelectingModel):
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
        del messages, name, schema
        self.calls += 1
        skill_id = "explore_forward" if self.calls == 1 else "reacquire_target"
        return ModelResponse(
            text=json.dumps(
                {
                    "reasoning_summary": "Use a different learned option after timeout",
                    "chosen_goal_id": "survive",
                    "skill_id": skill_id,
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


class _MalformedThenRepairingModel(_ShelterSelectingModel):
    def __init__(self, *, repair_text: str | None = None) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[ModelMessage, ...], str]] = []
        self.repair_text = repair_text or json.dumps(
            {
                "r": "Recovered the explicit option",
                "g": "survive",
                "s": "explore_forward",
                "p": {},
            }
        )

    def complete_structured(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        del schema
        self.calls.append((messages, name))
        text = (
            '{"r":"Continue","g":"survive","s":"explore_forward","p":{'
            if len(self.calls) == 1
            else self.repair_text
        )
        return ModelResponse(text=text, model=self.model_id, latency_ms=1.0)


class _AlwaysMalformedModel(_MalformedThenRepairingModel):
    def complete_structured(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        response = super().complete_structured(
            messages,
            name=name,
            schema=schema,
        )
        return response.model_copy(update={"text": '{"r":"still truncated"'})


class _AuthorityRepairingModel(_ShelterSelectingModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[ModelMessage, ...]] = []

    def complete_structured(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        del name, schema
        self.calls.append(messages)
        if len(self.calls) == 1:
            payload = {
                "r": "Try an option that does not exist",
                "g": "wrong-goal",
                "s": "invented_skill",
                "p": {"allow_attack": True},
            }
        else:
            payload = {
                "r": "Use the allowed traversal option",
                "g": "wrong-goal",
                "s": "explore_forward",
                "p": {"allow_attack": True, "invented": "claim"},
                "o": "I will explore without attacking.",
            }
        return ModelResponse(
            text=json.dumps(payload),
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
    assert decision.game_chat is None


def test_operator_response_and_game_chat_are_distinct_schema_channels() -> None:
    decision = CognitionDecision(
        reasoning_summary="Private execution summary.",
        say="Visible only in the operator console.",
        game_chat="Visible to players inside Bedrock.",
    )

    assert decision.say == "Visible only in the operator console."
    assert decision.game_chat == "Visible to players inside Bedrock."


def test_high_level_uses_lossless_compact_structured_wire_schema() -> None:
    model = _WireCapturingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    decision = controller.decide(_board(), _context())

    assert set(model.schema["properties"]) == {
        "r",
        "g",
        "s",
        "p",
        "o",
        "c",
        "x",
        "q",
        "w",
        "d",
        "n",
    }
    assert "required" not in model.schema
    assert decision.skill_id == "explore_forward"
    assert decision.reasoning_summary == "Explore while gathering evidence"


def test_high_level_repairs_truncated_json_once_with_compact_bounded_context() -> None:
    model = _MalformedThenRepairingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    decision = controller.decide(_board(), _context())

    assert decision.skill_id == "explore_forward"
    assert len(model.calls) == 2
    assert model.calls[0][1] == "cognition_decision"
    repair_messages, repair_name = model.calls[1]
    assert repair_name == "cognition_decision_json_repair"
    assert len(repair_messages) == 2
    assert sum(len(message.content) for message in repair_messages) < 3_000
    assert "fresh_facts" not in "".join(message.content for message in repair_messages)
    repair_payload = json.loads(repair_messages[1].content)
    assert repair_payload["rejected_output"].endswith('"p":{')
    assert {item["s"] for item in repair_payload["authority_bounds"]["allowed_skills"]}
    assert controller.metrics.calls == 2
    assert controller.metrics.repairs == 1
    assert controller.metrics.json_repairs == 1
    assert controller.metrics.json_repair_failures == 0
    assert controller.metrics.failures == 0


def test_high_level_json_repair_is_single_attempt_and_fails_closed() -> None:
    model = _AlwaysMalformedModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    decision = controller.decide(_board(), _context())

    assert len(model.calls) == 2
    assert decision.skill_id is None
    assert decision.request_replan is True
    assert controller.metrics.calls == 2
    assert controller.metrics.json_repairs == 1
    assert controller.metrics.json_repair_failures == 1
    assert controller.metrics.failures == 1
    assert "after one bounded repair" in (controller.metrics.last_error or "")


def test_compact_option_repair_preserves_operator_authority_and_parameter_bounds() -> None:
    message = OperatorMessage(
        message_id="bounded-repair",
        created_ns=2,
        text="Explore the visible open ground, but do not attack.",
        status=OperatorMessageStatus.DELIVERED,
    )
    context = _context()
    context.operator_messages = (message,)
    model = _AuthorityRepairingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    decision = controller.decide(_board(), context)

    assert len(model.calls) == 2
    assert decision.chosen_goal_id == "operator:bounded-repair"
    assert decision.skill_id == "explore_forward"
    assert decision.skill_parameters == {"allow_attack": False}
    assert decision.say == "I will explore without attacking."
    repair_messages = model.calls[1]
    assert len(repair_messages) == 2
    assert sum(len(message.content) for message in repair_messages) < 3_000
    combined = "".join(item.content for item in repair_messages)
    assert "fresh_facts" not in combined
    assert "ACTIVE OPERATOR DIRECTIVE" not in combined
    repair_payload = json.loads(repair_messages[1].content)
    assert repair_payload["authority_bounds"]["authority_goal_id"] == ("operator:bounded-repair")
    assert repair_payload["authority_bounds"]["required_action_constraints"] == {
        "allow_attack": False
    }


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
    explore = next(skill for skill in payload["skills"] if skill["skill_id"] == "explore_forward")
    assert {"allow_attack", "allow_use", "allow_jump"}.issubset(explore["parameters"])
    assert "preconditions" not in explore
    assert "invariants" not in explore
    assert "success_evidence" in explore
    assert "establish_basic_shelter" not in {skill["skill_id"] for skill in payload["skills"]}
    assert (
        "must be addressed before any conflicting standing goal"
        in model.initial_messages[0].content
    )
    assert sum(len(message.content) for message in model.initial_messages) < 8_000
    assert decision.chosen_goal_id == "operator:correction"
    assert decision.say == "I am climbing the hill now."


def test_high_level_prompt_bounds_strategic_facts_and_omits_motor_fingerprints() -> None:
    now = time.monotonic_ns()
    facts = [
        PerceptionFact(
            key="frame.content_hash",
            value="f" * 1_000,
            confidence=1.0,
            observed_ns=now,
            source="capture:" + ("verbose-" * 100),
            expires_after_ms=10_000,
        ),
        PerceptionFact(
            key="perception.luma_grid",
            value="l" * 1_000,
            confidence=1.0,
            observed_ns=now,
            source="capture:test",
            expires_after_ms=10_000,
        ),
        PerceptionFact(
            key="scene.observation_dhash",
            value="a1b2c3d4e5f60718",
            confidence=1.0,
            observed_ns=now,
            source="vlm:test",
            expires_after_ms=10_000,
        ),
        PerceptionFact(
            key="environment.time_of_day",
            value="night" + ("-long" * 100),
            confidence=0.9874,
            observed_ns=now,
            source="vlm:" + ("verbose-" * 100),
            expires_after_ms=10_000,
        ),
        *(
            PerceptionFact(
                key=f"misc.fact_{index:02d}",
                value="detail-" * 100,
                confidence=0.9,
                observed_ns=now,
                source="test:" + ("verbose-" * 100),
                expires_after_ms=10_000,
            )
            for index in range(24)
        ),
    ]
    model = _CapturingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    controller.decide(_board(*facts), _context())

    payload = json.loads(model.initial_messages[1].content)
    strategic = payload["fresh_facts"]
    assert len(strategic) == 16
    assert "frame.content_hash" not in strategic
    assert "perception.luma_grid" not in strategic
    assert "scene.observation_dhash" not in strategic
    assert strategic["environment.time_of_day"] == [
        ("night" + ("-long" * 100))[:120],
        0.987,
    ]
    assert all(isinstance(value, list) and len(value) == 2 for value in strategic.values())
    combined = "".join(message.content for message in model.initial_messages)
    assert "capture:" not in combined
    assert "vlm:verbose" not in combined
    assert len(combined) < 9_000


@pytest.mark.parametrize(
    "status",
    (OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED),
)
@pytest.mark.parametrize(
    "text",
    (
        "Mine the marked dirt block.",
        "Mine the marked dirt block, then stop and reassess.",
        "Mine the marked dirt block until it breaks, then stop and reassess the opening.",
    ),
)
def test_marked_dirt_instruction_uses_immediate_operator_fast_path(
    status: OperatorMessageStatus,
    text: str,
) -> None:
    now = time.monotonic_ns()
    reference = PerceptionFact(
        key="target.reference_available",
        value=True,
        confidence=1.0,
        observed_ns=now,
        source="operator",
        expires_after_ms=10_000,
    )
    board = _board(reference)
    latest = board.latest()
    assert latest is not None
    board.upsert_semantic_track(
        instance_id=latest.instance_id,
        track=Track(
            track_id="operator:target",
            label="dirt",
            confidence=1.0,
            region=ScreenRegion(x=0.3, y=0.2, width=0.4, height=0.6),
            first_seen_ns=now,
            last_seen_ns=now,
            attributes={"source": "operator"},
        ),
    )
    message = OperatorMessage(
        message_id="mine-marked-dirt",
        created_ns=now,
        text=text,
        kind=OperatorMessageKind.CORRECTION,
        status=status,
    )
    context = _context()
    context.operator_messages = (message,)
    model = _GrammarCapturingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    decision = controller.decide(board, context)

    assert model.messages == ()
    assert controller.metrics.calls == 0
    assert decision.chosen_goal_id == "operator:mine-marked-dirt"
    assert decision.skill_id == "mine_visible_block"
    assert decision.skill_parameters == {"target": "dirt"}
    assert decision.instruction == message.text
    assert decision.say == "Starting that now."
    assert decision.request_replan is False


def test_direct_operator_action_uses_sampler_grammar_that_requires_a_skill() -> None:
    message = OperatorMessage(
        message_id="move-now",
        created_ns=2,
        text="Move forward through safe open terrain.",
        status=OperatorMessageStatus.DELIVERED,
    )
    context = _context()
    context.operator_messages = (message,)
    model = _GrammarCapturingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    decision = controller.decide(_board(), context)

    skill_rule = next(line for line in model.grammar.splitlines() if line.startswith("skill ::="))
    authority_rule = next(
        line for line in model.grammar.splitlines() if line.startswith("authority-goal ::=")
    )
    assert '"null"' not in skill_rule
    assert "explore_forward" in skill_rule
    assert "traverse_level_ground" in skill_rule
    assert "gather_nearby_wood" not in skill_rule
    assert "open_inventory" not in skill_rule
    assert '"null"' not in authority_rule
    assert "operator:move-now" in authority_rule
    assert decision.skill_id == "explore_forward"


def test_negated_question_and_unsupported_actions_keep_null_available() -> None:
    cases = (
        ("Do not move.", OperatorMessageKind.INSTRUCTION),
        ("Stop mining.", OperatorMessageKind.INSTRUCTION),
        ("Can you tell me how to craft a pickaxe?", OperatorMessageKind.INSTRUCTION),
        ("Move forward.", OperatorMessageKind.QUESTION),
        ("Eat food now.", OperatorMessageKind.INSTRUCTION),
    )
    for text, kind in cases:
        message = OperatorMessage(
            message_id="move-now",
            created_ns=2,
            text=text,
            kind=kind,
            status=OperatorMessageStatus.DELIVERED,
        )
        context = _context()
        context.operator_messages = (message,)
        model = _GrammarCapturingModel()
        controller = HighLevelController(model, build_bootstrap_skill_library())

        controller.decide(_board(), context)

        assert model.messages, text
        skill_rule = next(
            line for line in model.grammar.splitlines() if line.startswith("skill ::=")
        )
        assert '"null"' in skill_rule, text


def test_unavailable_requested_skill_allows_only_null() -> None:
    message = OperatorMessage(
        message_id="move-now",
        created_ns=2,
        text="Mine the stone in front of you.",
        status=OperatorMessageStatus.DELIVERED,
    )
    context = _context()
    context.operator_messages = (message,)
    model = _GrammarCapturingModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())

    decision = controller.decide(_board(), context)

    assert model.messages
    skill_rule = next(line for line in model.grammar.splitlines() if line.startswith("skill ::="))
    assert skill_rule == 'skill ::= "null"'
    assert decision.skill_id is None
    assert decision.request_replan is True


def test_danger_defers_operator_authority_without_acknowledging_completion() -> None:
    now = time.monotonic_ns()
    danger = PerceptionFact(
        key="danger.immediate",
        value=True,
        confidence=0.95,
        observed_ns=now,
        source="test",
        expires_after_ms=10_000,
    )
    message = OperatorMessage(
        message_id="danger-command",
        created_ns=2,
        text="Mine the marked dirt block.",
        status=OperatorMessageStatus.DELIVERED,
    )
    context = _context()
    context.operator_messages = (message,)
    model = _GrammarCapturingModel(
        skill_id="retreat_from_danger",
        chosen_goal_id="operator:danger-command",
    )
    controller = HighLevelController(model, build_bootstrap_skill_library())

    reference = PerceptionFact(
        key="target.reference_available",
        value=True,
        confidence=1.0,
        observed_ns=now,
        source="operator",
        expires_after_ms=10_000,
    )

    decision = controller.decide(_board(danger, reference), context)

    assert model.messages
    payload = json.loads(model.messages[1].content)
    assert payload["active_operator_message"] is None
    assert len(model.messages) == 2
    assert decision.skill_id == "retreat_from_danger"
    assert decision.chosen_goal_id is None
    assert decision.request_replan is True
    assert decision.say is None


def test_skill_shortlist_uses_unseen_prior_and_penalizes_repeated_failure() -> None:
    bootstrap = build_bootstrap_skill_library()
    library = SkillLibrary()
    library.register(bootstrap.get("explore_forward"))
    library.register(bootstrap.get("reacquire_target"))
    timeout = SkillRun(
        run_id="timeout",
        skill_id="explore_forward",
        started_ns=1,
        ended_ns=2,
        outcome=SkillOutcome.TIMED_OUT,
        failure_reason="skill-timeout",
    )
    for index in range(6):
        library.record(timeout.model_copy(update={"run_id": f"timeout-{index}"}))
    controller = HighLevelController(_GrammarCapturingModel(), library)

    payloads = controller._feasible_skill_payloads(_board())

    assert [payload["skill_id"] for payload in payloads] == [
        "reacquire_target",
        "explore_forward",
    ]
    assert payloads[0]["competence"] == 0.5
    assert payloads[1]["competence"] == 0.0


def test_skill_shortlist_reserves_matching_and_safety_options() -> None:
    bootstrap = build_bootstrap_skill_library()
    controller = HighLevelController(_GrammarCapturingModel(), bootstrap)

    movement = controller._feasible_skill_payloads(
        _board(),
        query_text="Move forward through open terrain.",
    )
    assert [payload["skill_id"] for payload in movement[:2]] == [
        "explore_forward",
        "traverse_level_ground",
    ]

    now = time.monotonic_ns()
    danger = PerceptionFact(
        key="danger.immediate",
        value=True,
        confidence=0.95,
        observed_ns=now,
        source="test",
        expires_after_ms=10_000,
    )
    safety = controller._feasible_skill_payloads(_board(danger))
    assert [payload["skill_id"] for payload in safety] == ["retreat_from_danger"]


def test_fresh_operator_directive_owns_idle_replan_decision() -> None:
    message = OperatorMessage(
        message_id="new-correction",
        created_ns=2,
        text="Use the selected open terrain gap. Do not attack.",
        status=OperatorMessageStatus.DELIVERED,
    )
    context = _context()
    context.operator_messages = (message,)
    controller = HighLevelController(
        _IdleCapturingModel(),
        build_bootstrap_skill_library(),
    )

    decision = controller.decide(_board(), context)

    assert decision.chosen_goal_id == "operator:new-correction"
    assert decision.skill_id is None
    assert decision.request_replan is True
    assert decision.skill_parameters["allow_attack"] is False


def test_operator_grounded_skill_binds_exact_active_track_label() -> None:
    message = OperatorMessage(
        message_id="grounded-correction",
        created_ns=2,
        text="Use the selected region with ROCKET.",
        status=OperatorMessageStatus.DELIVERED,
    )
    context = _context()
    context.operator_messages = (message,)
    board = _board()
    latest = board.latest()
    assert latest is not None
    board.upsert_semantic_track(
        instance_id=latest.instance_id,
        track=Track(
            track_id="operator:target",
            label="open terrain gap",
            confidence=1.0,
            region=ScreenRegion(x=0.1, y=0.2, width=0.2, height=0.3),
            first_seen_ns=1,
            last_seen_ns=2,
            attributes={"source": "operator"},
        ),
    )
    controller = HighLevelController(
        _TargetSelectingModel(),
        build_bootstrap_skill_library(),
    )

    decision = controller.decide(board, context)

    assert decision.chosen_goal_id == "operator:grounded-correction"
    assert decision.skill_id == "reacquire_target"
    assert decision.skill_parameters["target"] == "open terrain gap"


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


def test_literal_operator_action_prohibitions_are_enforced_on_model_output() -> None:
    message = OperatorMessage(
        message_id="constraints",
        created_ns=2,
        text=(
            "Continue exploring. Do not attack and do not use or interact. "
            "Jump over one-block rises."
        ),
        status=OperatorMessageStatus.ACKNOWLEDGED,
    )
    context = _context()
    context.operator_messages = (message,)
    controller = HighLevelController(_CapturingModel(), build_bootstrap_skill_library())

    decision = controller.decide(_board(), context)

    assert decision.skill_parameters["allow_attack"] is False
    assert decision.skill_parameters["allow_use"] is False
    assert "allow_jump" not in decision.skill_parameters


def test_repeated_timed_out_option_is_repaired_to_different_learned_option() -> None:
    library = build_bootstrap_skill_library()
    recent = SkillRun(
        run_id="timeout-2",
        skill_id="explore_forward",
        started_ns=1,
        ended_ns=2,
        outcome=SkillOutcome.TIMED_OUT,
        failure_reason="skill-timeout",
    )
    library.record(recent.model_copy(update={"run_id": "timeout-1"}))
    library.record(recent)
    context = _context()
    context.recent_skill_runs = (recent,)
    model = _RetryRepairingModel()
    controller = HighLevelController(model, library)

    decision = controller.decide(_board(), context)

    assert decision.skill_id == "reacquire_target"
    assert model.calls == 2
    assert controller.metrics.retry_repairs == 1


def test_repair_cannot_alternate_between_two_recently_failed_options() -> None:
    library = build_bootstrap_skill_library()
    explore = SkillRun(
        run_id="explore-2",
        skill_id="explore_forward",
        started_ns=1,
        ended_ns=2,
        outcome=SkillOutcome.TIMED_OUT,
        failure_reason="skill-timeout",
    )
    reacquire = explore.model_copy(
        update={
            "run_id": "reacquire-2",
            "skill_id": "reacquire_target",
            "outcome": SkillOutcome.FAILED,
            "failure_reason": "target-not-found",
        }
    )
    for run in (explore, explore, reacquire, reacquire):
        library.record(run.model_copy(update={"run_id": f"{run.run_id}-{time.time_ns()}"}))
    context = _context()
    context.recent_skill_runs = (reacquire, explore)
    model = _RetryRepairingModel()
    controller = HighLevelController(model, library)

    decision = controller.decide(_board(), context)

    assert decision.skill_id is None
    assert decision.request_replan is True
    assert model.calls == 2
    assert controller.metrics.retry_repairs == 1
    assert controller.metrics.last_error == "repeated-option-blocked:explore_forward"
