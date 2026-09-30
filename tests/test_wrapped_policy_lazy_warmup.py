"""Pure retained-worker admission tests; no models, IPC or game controls."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest

from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.config import PolicyConfig
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.policy_service import GroundedPolicyRouter, TemporalPolicyClient
from minecraft_ai.safety import MotorAction


class HandshakePolicy:
    def __init__(self, name: str, key: str = "w") -> None:
        self.policy_id = name
        self.key = key
        self.entered = threading.Event()
        self.allow_ready = threading.Event()
        self.loads = self.calls = self.resets = self.closed = 0
        self.ready = False

    def warmup(self) -> None:
        self.loads += 1
        self.entered.set()
        assert self.allow_ready.wait(2), "test did not release its private load gate"
        self.ready = True

    def act(self, board, intent, *, sequence) -> MotorAction:
        assert self.ready, "an unqualified worker received an action"
        self.calls += 1
        return MotorAction(sequence=sequence, keys_down=(self.key,))

    def reset(self) -> MotorAction:
        assert not self.entered.is_set() or self.ready, "reset raced the warmup owner"
        self.resets += 1
        return MotorAction(sequence=0, keys_up=(self.key,))

    def close(self) -> None:
        assert not self.entered.is_set() or self.ready, "close raced the warmup owner"
        self.closed += 1

    def status(self) -> dict[str, object]:
        return {"policy_id": self.policy_id}


def intent(level: ActionLevel, episode: str = "option-1") -> MotorIntent:
    return MotorIntent(
        skill_id="admitted-option", mode="inspect", episode_id=episode,
        action_level=level,
    )


def test_wrapped_startup_and_prospective_plan_hints_load_no_experts() -> None:
    policies = [HandshakePolicy(name) for name in ("steve", "raw", "gui", "rocket")]
    router = GroundedPolicyRouter(
        policies[0], raw_motion=policies[1], gui=policies[2], grounded=policies[3],
    )
    router.defer_option_warmup()
    router.defer_option_warmup()  # Same admission is idempotent.
    router.warmup()
    for level in ActionLevel:
        router.request_warm(level)
    assert [p.loads for p in policies] == [0, 0, 0, 0]
    assert router.status()["episode_id"] is None
    router.reset()
    router.close()
    assert [p.loads for p in policies] == [0, 0, 0, 0]
    assert [p.closed for p in policies] == [1, 1, 1, 1]


def test_unused_real_temporal_clients_never_capture_allocate_or_spawn(
    tmp_path, monkeypatch,
) -> None:
    for name in ("python", "model", "weights"):
        (tmp_path / name).touch()
    (tmp_path / "source").mkdir()
    config = PolicyConfig(
        enabled=True, python_path=str(tmp_path / "python"),
        source_path=str(tmp_path / "source"), model_path=str(tmp_path / "model"),
        weights_path=str(tmp_path / "weights"), model_sha256="a" * 64,
        weights_sha256="b" * 64, model_version="fixture-only",
        source_commit="c" * 40, license="MIT",
    )
    capture = Mock(side_effect=AssertionError("unused retained worker requested pixels"))
    spawn = Mock(side_effect=AssertionError("unused retained worker attempted a process"))
    allocate = Mock(side_effect=AssertionError("unused retained worker allocated shared pixels"))
    monkeypatch.setattr("minecraft_ai.policy_service.subprocess.Popen", spawn)
    monkeypatch.setattr("minecraft_ai.policy_service.shared_memory.SharedMemory", allocate)
    clients = [TemporalPolicyClient(config=config, frame_provider=capture) for _ in range(4)]
    router = GroundedPolicyRouter(
        clients[0], raw_motion=clients[1], gui=clients[2], grounded=clients[3],
    )
    router.defer_option_warmup()
    router.warmup()
    for level in ActionLevel:
        router.request_warm(level)
    router.reset()
    router.reset_world()
    router.notify_inputs_released()
    router.close()
    capture.assert_not_called()
    spawn.assert_not_called()
    allocate.assert_not_called()
    assert all(client._worker_start_attempts == 0 for client in clients)


@pytest.mark.parametrize(
    ("level", "selected", "loaded"),
    [
        (ActionLevel.LATENT, 0, (0,)),
        (ActionLevel.SKILL, 0, (0,)),
        (ActionLevel.RAW, 1, (1,)),
        (ActionLevel.MOTION, 1, (1,)),
        (ActionLevel.GUI, 2, (2,)),
        (ActionLevel.GROUNDED, 0, (0, 3)),
    ],
)
def test_first_actual_option_loads_only_its_bound_experts_and_waits_without_input(
    level: ActionLevel, selected: int, loaded: tuple[int, ...],
) -> None:
    policies = [HandshakePolicy(name) for name in ("steve", "raw", "gui", "rocket")]
    router = GroundedPolicyRouter(
        policies[0], raw_motion=policies[1], gui=policies[2], grounded=policies[3],
    )
    router.defer_option_warmup()
    router.warmup()
    request = intent(level)
    try:
        release = router.act(PerceptionBlackboard(), request, sequence=1)
        for index in loaded:
            assert policies[index].entered.wait(1)
        assert [p.loads for p in policies] == [int(i in loaded) for i in range(4)]
        assert release.keys_down == release.buttons_down == ()
        assert router.status()["episode_id"] is None
        assert not any(p.calls for p in policies)
        # Repeated motor ticks neither duplicate a load nor reset its sole reader.
        router.act(PerceptionBlackboard(), request, sequence=2)
        assert [p.loads for p in policies] == [int(i in loaded) for i in range(4)]
        assert all(p.resets == 0 for i, p in enumerate(policies) if i in loaded)
    finally:
        for p in policies:
            p.allow_ready.set()
        for index in loaded:
            router._ensure_warm(policies[index])
    action = router.act(PerceptionBlackboard(), request, sequence=3)
    assert action.keys_down == (policies[selected].key,)
    assert policies[selected].calls == 1
    assert router.status()["episode_id"] == "option-1"
    # A changed level inside the same option cannot load/rebind another body.
    router.act(PerceptionBlackboard(), intent(ActionLevel.RAW), sequence=4)
    assert policies[selected].calls == 2
    assert [p.loads for p in policies] == [int(i in loaded) for i in range(4)]
    router.close()


def test_cold_option_releases_previous_body_before_waiting_for_new_handshake() -> None:
    primary, raw = HandshakePolicy("steve"), HandshakePolicy("raw", "space")
    primary.allow_ready.set()
    router = GroundedPolicyRouter(primary, raw_motion=raw)
    router.defer_option_warmup()
    first = intent(ActionLevel.LATENT, "old")
    router.act(PerceptionBlackboard(), first, sequence=1)
    router._ensure_warm(primary)
    router.act(PerceptionBlackboard(), first, sequence=2)
    try:
        release = router.act(
            PerceptionBlackboard(), intent(ActionLevel.RAW, "new"), sequence=3,
        )
        assert raw.entered.wait(1)
        assert release.keys_down == release.buttons_down == ()
        assert release.keys_up == ("w",)
        assert raw.calls == raw.resets == 0
        assert router.status()["episode_id"] is None
    finally:
        raw.allow_ready.set()
        router._ensure_warm(raw)
    action = router.act(PerceptionBlackboard(), intent(ActionLevel.RAW, "new"), sequence=4)
    assert action.keys_down == ("space",)
    assert router.status()["active_route"] == "raw_motion"
    router.close()


def test_missing_optional_body_preserves_existing_primary_fallback() -> None:
    primary = HandshakePolicy("steve")
    router = GroundedPolicyRouter(primary)
    router.defer_option_warmup()
    try:
        action = router.act(PerceptionBlackboard(), intent(ActionLevel.RAW), sequence=1)
        assert primary.entered.wait(1)
        assert action.keys_down == action.buttons_down == ()
    finally:
        primary.allow_ready.set()
        router._ensure_warm(primary)
    router.act(PerceptionBlackboard(), intent(ActionLevel.RAW), sequence=2)
    assert router.status()["fallback_bindings"] == 1
    assert router.status()["active_route"] == "semantic"
    router.close()


def test_failed_first_handshake_cannot_bind_act_or_retry_another_worker(monkeypatch) -> None:
    errors = []
    monkeypatch.setattr(threading, "excepthook", errors.append)

    class RejectedPolicy(HandshakePolicy):
        def warmup(self) -> None:
            super().warmup()
            raise RuntimeError("checkpoint identity rejected")

    primary = RejectedPolicy("steve")
    raw = HandshakePolicy("raw")
    router = GroundedPolicyRouter(primary, raw_motion=raw)
    router.defer_option_warmup()
    try:
        router.act(PerceptionBlackboard(), intent(ActionLevel.LATENT), sequence=1)
        assert primary.entered.wait(1)
    finally:
        primary.allow_ready.set()
        thread = router._warming.get(id(primary))
        if thread is not None:
            thread.join(2)
    assert len(errors) == 1
    action = router.act(PerceptionBlackboard(), intent(ActionLevel.LATENT), sequence=2)
    assert action.keys_down == action.buttons_down == ()
    assert primary.loads == 1
    assert primary.calls == raw.loads == 0
    assert router.status()["episode_id"] is None
    assert router.status()["failed_warmup_policy_ids"] == ["steve"]
    router.close()


def test_retirement_fences_cold_load_without_racing_its_owner() -> None:
    primary = HandshakePolicy("steve")
    raw = HandshakePolicy("raw")
    router = GroundedPolicyRouter(primary, raw_motion=raw)
    router.defer_option_warmup()
    try:
        router.act(PerceptionBlackboard(), intent(ActionLevel.LATENT), sequence=1)
        assert primary.entered.wait(1)
        report = router.retire(deadline_ns=time.monotonic_ns(), inputs_released=True)
        assert report["complete"] is False
        assert primary.closed == 0
        router.act(PerceptionBlackboard(), intent(ActionLevel.RAW, "cancelled"), sequence=2)
        assert raw.loads == raw.calls == 0
    finally:
        primary.allow_ready.set()
        thread = router._warming.get(id(primary))
        if thread is not None:
            thread.join(2)
    assert id(primary) not in router._warmed
    router.close()


def test_deferral_refuses_a_preexisting_warm_or_bound_owner() -> None:
    primary = HandshakePolicy("steve")
    primary.allow_ready.set()
    router = GroundedPolicyRouter(primary)
    router.warmup()
    with pytest.raises(RuntimeError, match="unstarted router"):
        router.defer_option_warmup()
    assert primary.loads == 1
    router.close()


def test_actual_optional_erais_wrapper_retains_nonassociation_dispatch(tmp_path) -> None:
    module = pytest.importorskip(
        "erais.sensorimotor.minecraft_experiment",
        reason="the external ERAIS runtime factory is an optional integration",
    )
    primary, raw = HandshakePolicy("steve"), HandshakePolicy("raw")
    router = GroundedPolicyRouter(primary, raw_motion=raw)
    router.defer_option_warmup()
    # Fresh observations remain absent. This exercise must abstain, never
    # publish fabricated positive facts or run a learned model from fixtures.
    wrapper = module.AdaptiveInteractionPolicy(router, lambda: None, tmp_path / "body.npz")
    assert wrapper.base_policy is router
    wrapper.warmup()
    wrapper.request_warm(ActionLevel.LATENT)
    wrapper.act(
        PerceptionBlackboard(),
        MotorIntent(
            skill_id="explore_forward", mode="explore", episode_id="association-option",
            action_level=ActionLevel.LATENT,
        ),
        sequence=1,
    )
    assert primary.loads == raw.loads == primary.calls == raw.calls == 0
    request = intent(ActionLevel.LATENT, "retained-option")
    try:
        release = wrapper.act(PerceptionBlackboard(), request, sequence=2)
        assert primary.entered.wait(1)
        assert release.keys_down == release.buttons_down == ()
        assert wrapper.active is False
        assert router.status()["episode_id"] is None
    finally:
        primary.allow_ready.set()
        router._ensure_warm(primary)
    action = wrapper.act(PerceptionBlackboard(), request, sequence=3)
    assert action.keys_down == ("w",)
    assert primary.calls == 1
    assert raw.loads == raw.calls == 0
    assert router.status()["episode_id"] == "retained-option"
    router.close()
