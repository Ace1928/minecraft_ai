"""Pure optional-factory startup checks; never contact the live supervisor."""

from __future__ import annotations

import signal
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from minecraft_ai import agent_process
from minecraft_ai import runtime_factory as startup
from minecraft_ai.config import RuntimeConfig, RuntimeFactoryConfig


class FakeRuntime:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)
        self.stop = Mock()
        self.run_forever = Mock()
        self.close_before_run = Mock(return_value=True)
        self.close_constructed_runtime = Mock(return_value=True)


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    kwargs = dict(
        perception=SimpleNamespace(close=Mock()),
        blackboard=object(),
        executor=SimpleNamespace(policy=object(), close=Mock()),
        state_db=SimpleNamespace(close=Mock()),
        trajectory=SimpleNamespace(close=Mock()),
        lease_id="existing-lease",
        lease_renew_ms=100,
        high_level=object(),
    )
    commands: list[tuple[str, dict[str, object]]] = []
    handlers = {signal.SIGINT: signal.SIG_DFL, signal.SIGTERM: signal.SIG_DFL}
    factory = Mock(side_effect=lambda **call: FakeRuntime(**call["runtime_kwargs"]))
    importer = Mock(return_value=SimpleNamespace(create=factory))

    def install(sig: signal.Signals, handler: object) -> object:
        previous = handlers[sig]
        handlers[sig] = handler
        return previous

    def send(command: str, **payload: object) -> dict[str, object]:
        commands.append((command, payload))
        return {}

    monkeypatch.setattr(startup, "AgentRuntime", FakeRuntime)
    monkeypatch.setattr(startup.signal, "signal", install)
    monkeypatch.setattr(startup, "send_command", send)
    monkeypatch.setattr(startup, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(startup, "operator_pause_latched", lambda: False)
    monkeypatch.setattr(startup.importlib, "import_module", importer)
    return SimpleNamespace(
        kwargs=kwargs, commands=commands, handlers=handlers, factory=factory,
        importer=importer,
        config=RuntimeFactoryConfig(reference="custom_planner:create"),
    )


def test_default_config_does_not_select_or_import_factory(environment, monkeypatch) -> None:
    runtime = FakeRuntime(**environment.kwargs)
    constructor = Mock(return_value=runtime)
    monkeypatch.setattr(startup, "AgentRuntime", constructor)
    assert RuntimeConfig().runtime_factory is None
    startup.run_agent_runtime(environment.kwargs)
    constructor.assert_called_once_with(**environment.kwargs)
    environment.importer.assert_not_called()
    runtime.run_forever.assert_called_once()
    assert environment.commands == []


@pytest.mark.parametrize("reference", ["", "module", "a:b:c", "a/b:create", "a:create()"])
def test_factory_reference_is_explicit_module_callable(reference: str) -> None:
    with pytest.raises(ValidationError):
        RuntimeFactoryConfig(reference=reference)


def test_guard_handoff_preserves_all_public_owners(environment) -> None:
    runtime = FakeRuntime(**environment.kwargs)
    runtime.high_level = object()

    def run() -> None:
        assert not any(t.name == "minecraft-ai-construction-lease" for t in threading.enumerate())

    runtime.run_forever.side_effect = run
    environment.factory.side_effect = lambda **_: runtime
    startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    runtime.run_forever.assert_called_once()
    runtime.close_constructed_runtime.assert_not_called()
    assert environment.commands == [("renew", dict(
        lease_id="existing-lease", ttl_ms=5000, timeout_s=0.5,
    ))]
    assert all(h == signal.SIG_DFL for h in environment.handlers.values())


def test_initial_rejected_lease_skips_even_external_import(environment, monkeypatch) -> None:
    send = Mock(side_effect=RuntimeError("lease expired"))
    monkeypatch.setattr(startup, "send_command", send)
    with pytest.raises(RuntimeError, match="lease expired"):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    environment.importer.assert_not_called()
    environment.factory.assert_not_called()
    send.assert_called_once_with("renew", lease_id="existing-lease", ttl_ms=5000, timeout_s=0.5)
    environment.kwargs["perception"].close.assert_called_once()
    environment.kwargs["executor"].close.assert_called_once()
    assert 0 < environment.kwargs["trajectory"].close.call_args.kwargs["timeout_s"] <= 2
    environment.kwargs["state_db"].close.assert_not_called()


@pytest.mark.parametrize("latch", ["operator_pause_latched", "emergency_stop_latched"])
def test_existing_stop_latch_skips_renew_and_import(environment, monkeypatch, latch) -> None:
    monkeypatch.setattr(startup, latch, lambda: True)
    with pytest.raises(startup.RuntimeStartupCancelled):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    assert environment.commands == []
    environment.importer.assert_not_called()


def test_signal_during_import_skips_factory_and_stays_cancelled(environment) -> None:
    def import_module(_name: str) -> object:
        environment.handlers[signal.SIGTERM](signal.SIGTERM, None)
        return SimpleNamespace(create=environment.factory)

    environment.importer.side_effect = import_module
    with pytest.raises(startup.RuntimeStartupCancelled):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    environment.factory.assert_not_called()
    assert len(environment.commands) == 1


def test_signal_during_factory_uses_returned_cleanup_owner(environment) -> None:
    runtime = FakeRuntime(**environment.kwargs)

    def factory(*, runtime_kwargs, cancel_event):
        assert runtime_kwargs is not environment.kwargs
        environment.handlers[signal.SIGINT](signal.SIGINT, None)
        assert cancel_event.is_set()
        return runtime

    environment.factory.side_effect = factory
    with pytest.raises(startup.RuntimeStartupCancelled):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    runtime.run_forever.assert_not_called()
    runtime.stop.assert_called_once()
    runtime.close_constructed_runtime.assert_called_once()
    runtime.close_before_run.assert_not_called()  # Private hook delegates only when drained.
    environment.kwargs["perception"].close.assert_not_called()
    environment.kwargs["state_db"].close.assert_not_called()


def test_heartbeat_failure_cancels_slow_factory_and_never_hands_off(environment, monkeypatch):
    runtime = FakeRuntime(**environment.kwargs)
    calls = 0

    def renew(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("lease revoked by newer operator")
        return {}

    def factory(*, runtime_kwargs, cancel_event):
        assert cancel_event.wait(2)
        return runtime

    monkeypatch.setattr(startup, "send_command", renew)
    environment.factory.side_effect = factory
    with pytest.raises(startup.RuntimeStartupCancelled, match="lease failed"):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    assert calls == 2
    runtime.run_forever.assert_not_called()
    runtime.close_constructed_runtime.assert_called_once()


def test_absolute_deadline_rejects_returned_runtime(environment, monkeypatch) -> None:
    runtime = FakeRuntime(**environment.kwargs)
    clock = [100.0]
    monkeypatch.setattr(startup.time, "monotonic", lambda: clock[0])

    def factory(**_kwargs):
        clock[0] += 121
        return runtime

    environment.factory.side_effect = factory
    with pytest.raises(startup.RuntimeStartupCancelled, match="deadline"):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    runtime.run_forever.assert_not_called()
    runtime.close_constructed_runtime.assert_called_once()


@pytest.mark.parametrize("name", [
    "executor", "perception", "blackboard", "state_db", "lease_id", "trajectory",
])
def test_factory_cannot_replace_protected_runtime_fields(environment, name) -> None:
    runtime = FakeRuntime(**environment.kwargs)
    setattr(runtime, name, object())
    environment.factory.side_effect = lambda **_: runtime
    with pytest.raises(startup.RuntimeStartupCleanupIncomplete, match=f"protected field: {name}"):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    runtime.run_forever.assert_not_called()
    runtime.close_constructed_runtime.assert_not_called()
    for owned in ("perception", "executor", "trajectory", "state_db"):
        environment.kwargs[owned].close.assert_not_called()
    assert [command for command, _ in environment.commands] == ["renew"]


def test_deleted_protected_field_also_retains_ownership(environment) -> None:
    runtime = FakeRuntime(**environment.kwargs)
    del runtime.trajectory
    environment.factory.side_effect = lambda **_: runtime
    with pytest.raises(
        startup.RuntimeStartupCleanupIncomplete, match="protected field: trajectory",
    ):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    runtime.close_constructed_runtime.assert_not_called()
    environment.kwargs["trajectory"].close.assert_not_called()


def test_factory_cannot_swap_policy_inside_the_supplied_executor(environment) -> None:
    runtime = FakeRuntime(**environment.kwargs)

    def factory(**_kwargs):
        runtime.executor.policy = object()
        return runtime

    environment.factory.side_effect = factory
    with pytest.raises(startup.RuntimeStartupCleanupIncomplete, match="protected executor policy"):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    runtime.run_forever.assert_not_called()
    runtime.close_constructed_runtime.assert_not_called()
    environment.kwargs["executor"].close.assert_not_called()
    environment.kwargs["state_db"].close.assert_not_called()


@pytest.mark.parametrize("failure", [False, None, 1, RuntimeError("incomplete native drain")])
def test_incomplete_private_cleanup_forbids_public_fallback(environment, failure) -> None:
    runtime = FakeRuntime(**environment.kwargs)
    if isinstance(failure, Exception):
        runtime.close_constructed_runtime.side_effect = failure
    else:
        runtime.close_constructed_runtime.return_value = failure

    def factory(**_kwargs):
        environment.handlers[signal.SIGTERM](signal.SIGTERM, None)
        return runtime

    environment.factory.side_effect = factory
    with pytest.raises(startup.RuntimeStartupCleanupIncomplete):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    runtime.run_forever.assert_not_called()
    runtime.close_constructed_runtime.assert_called_once()
    environment.kwargs["perception"].close.assert_not_called()
    environment.kwargs["executor"].close.assert_not_called()
    environment.kwargs["state_db"].close.assert_not_called()
    assert [command for command, _ in environment.commands] == ["renew"]


def test_invalid_return_is_not_guessed_to_be_a_cleanup_owner(environment) -> None:
    invalid = SimpleNamespace(close_constructed_runtime=Mock())
    environment.factory.side_effect = lambda **_: invalid
    with pytest.raises(TypeError, match="must return AgentRuntime"):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    invalid.close_constructed_runtime.assert_not_called()
    environment.kwargs["perception"].close.assert_called_once()


def test_factory_exception_cleans_known_resources_without_supervisor_mutation(environment):
    environment.factory.side_effect = RuntimeError("factory owns partial private cleanup")
    with pytest.raises(RuntimeError, match="factory owns"):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    environment.kwargs["perception"].close.assert_called_once()
    environment.kwargs["executor"].close.assert_called_once()
    assert [command for command, _ in environment.commands] == ["renew"]


def test_partial_factory_can_explicitly_retain_borrowed_resources(environment) -> None:
    environment.factory.side_effect = startup.RuntimeStartupCleanupIncomplete("partial owner")
    with pytest.raises(startup.RuntimeStartupCleanupIncomplete, match="partial owner"):
        startup.run_agent_runtime(environment.kwargs, factory_config=environment.config)
    environment.kwargs["perception"].close.assert_not_called()
    environment.kwargs["executor"].close.assert_not_called()
    environment.kwargs["trajectory"].close.assert_not_called()
    environment.kwargs["state_db"].close.assert_not_called()


def test_unjoined_construction_worker_retains_resources_and_blocks_handoff(environment):
    guard = startup._ConstructionLease(
        "lease", threading.Event(), timeout_s=10, interval_s=0.5,
    )
    guard._thread = SimpleNamespace(join=Mock(), is_alive=Mock(return_value=True))
    with pytest.raises(startup.RuntimeStartupCleanupIncomplete, match="did not stop"):
        guard.close()
    guard._thread.join.assert_called_once_with(timeout=1.0)
    assert guard.cancel_event.is_set()


@pytest.mark.parametrize("incomplete", [False, True])
def test_main_database_cleanup_respects_pre_run_owner_failure(monkeypatch, incomplete):
    database = Mock()
    monkeypatch.setattr(agent_process, "StateDatabase", lambda *_: database)
    monkeypatch.setattr(agent_process, "load_config", lambda *_: RuntimeConfig())
    failure = (startup.RuntimeStartupCleanupIncomplete("still owned") if incomplete
               else RuntimeError("ordinary startup failure"))
    monkeypatch.setattr(agent_process, "build_bootstrap_skill_library", Mock(side_effect=failure))
    with pytest.raises(type(failure)):
        agent_process.main([
            "--lease-id", "lease", "--display", ":99", "--window-id", "42",
            "--instance-id", "bedrock:test:isolated",
        ])
    assert database.close.call_count == (0 if incomplete else 1)
