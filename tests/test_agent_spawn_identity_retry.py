"""The bounded spawn retry must not weaken process ownership checks."""
from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_ai.agent import lifecycle


@pytest.mark.parametrize("first_identity", ["missing", "pre_exec"])
def test_spawn_waits_for_exact_command_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_identity: str
) -> None:
    commands: list[tuple[str, ...]] = []
    reads: list[int] = []
    clock = [0.0]

    class Child:
        pid = 4242

        def poll(self) -> None:
            return None

    def popen(command: list[str], **kwargs: object) -> Child:
        commands.append(tuple(command))
        return Child()

    def identity(pid: int) -> tuple[int, tuple[str, ...]] | None:
        reads.append(pid)
        if len(reads) == 1:
            return None if first_identity == "missing" else (9876, ("pre-exec",))
        return 9876, commands[0]

    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(lifecycle, "AGENT_FILE", tmp_path / "agent-process.json")
    monkeypatch.setattr(lifecycle, "AGENT_LOG", tmp_path / "agent.log")
    monkeypatch.setattr(lifecycle.subprocess, "Popen", popen)
    monkeypatch.setattr(lifecycle, "_linux_process_identity", identity)
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        lifecycle.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        lifecycle, "_terminate_spawned_agent_group",
        lambda child: pytest.fail("a verified child must not be terminated"),
    )
    process = lifecycle.launch_agent_process(
        lease_id="test-lease", display=":2", window_id=123,
        instance_id="bedrock:test", role="explorer", capture_source="x11",
    )
    assert reads == [4242, 4242]
    assert process.proc_start_ticks == 9876
    assert process.command_sha256 == lifecycle._command_sha256(commands[0])
    assert lifecycle.AgentProcess.load() == process


@pytest.mark.parametrize("exited", [False, True])
def test_spawn_mismatch_cleans_up_without_publishing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exited: bool
) -> None:
    clock = [0.0]
    cleanups: list[int] = []

    class Child:
        pid = 4242

        def poll(self) -> int | None:
            return 1 if exited else None

    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(lifecycle, "AGENT_FILE", tmp_path / "agent-process.json")
    monkeypatch.setattr(lifecycle, "AGENT_LOG", tmp_path / "agent.log")
    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(lifecycle, "_linux_process_identity", lambda pid: (9876, ("other",)))
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        lifecycle.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        lifecycle, "_terminate_spawned_agent_group",
        lambda child: cleanups.append(child.pid) or True,
    )
    with pytest.raises(RuntimeError, match="could not establish agent process identity"):
        lifecycle.launch_agent_process(
            lease_id="test-lease", display=":2", window_id=123,
            instance_id="bedrock:test", role="explorer", capture_source="x11",
        )
    assert cleanups == [4242]
    assert not lifecycle.AGENT_FILE.exists()
    if exited:
        assert clock[0] == 0.0
    else:
        assert 2.0 <= clock[0] < 2.1
