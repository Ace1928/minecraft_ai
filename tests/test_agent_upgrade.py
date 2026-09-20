"""Offline handover tests. No game capture, real child, signal or live IPC."""

from __future__ import annotations

import json
import hashlib
import os
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from minecraft_ai import cli, supervisor
from minecraft_ai.agent import lifecycle, upgrade as u
from minecraft_ai.config import RuntimeConfig, RuntimeFactoryConfig


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    root.mkdir()
    module = root / "handover_fixture.py"
    module.write_text("raise AssertionError('preflight must never import candidate')\n")
    monkeypatch.syspath_prepend(str(root))
    previous = tmp_path / "previous.yaml"
    candidate = tmp_path / "candidate.yaml"
    previous.write_text("high_level:\n  api_key: do-not-publish-this\n")
    candidate.write_text(
        previous.read_text() + "runtime_factory:\n  reference: handover_fixture:build\n"
    )
    return previous, candidate, module


def test_offline_validation_does_not_import_candidate_or_contact_live(profiles, monkeypatch):
    previous, candidate, _ = profiles
    for name in ("_command", "_launcher_barrier", "activate_upgrade"):
        monkeypatch.setattr(u, name, lambda *a, **kw: pytest.fail("offline side effect"))
    result = CliRunner().invoke(cli.app, [
        "upgrade", "--dry-run", "--config", str(candidate), "--previous-config", str(previous),
    ])
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["phase"] == "validated-offline"
    assert receipt["prewarm"] == "unsupported-without-live-lease"
    assert len(receipt["plan_sha256"]) == 64
    assert "do-not-publish-this" not in result.output and str(previous.parent) not in result.output
    assert not (lifecycle.RUNTIME_DIR / "upgrade.json").exists()


@pytest.mark.parametrize("change", ["config", "source", "added-source"])
def test_plan_pins_reject_changes_after_offline_validation(profiles, monkeypatch, change):
    previous, candidate, module = profiles
    plan = u.prepare_upgrade(candidate, previous)
    if change == "config":
        candidate.write_text("cognition_hz: 1\n")
    elif change == "source":
        module.write_text("def build(): pass\n")
    else:
        # The fixture is a single-file module; also check package additions.
        package = module.parent / "package_fixture"
        package.mkdir()
        (package / "__init__.py").write_text("def build(): pass\n")
        candidate.write_text(previous.read_text() +
                             "runtime_factory:\n  reference: package_fixture:build\n")
        plan = u.prepare_upgrade(candidate, previous)
        (package / "new.py").write_text("VALUE = 1\n")
    with pytest.raises(u.UpgradeError, match="pinned-input-changed"):
        plan.check_unchanged()


@pytest.mark.parametrize("text", ["[1, 2]", "unknown: secret", "policy: [", "motor_hz: 0"])
def test_invalid_candidate_is_refused_without_echoing_configuration(profiles, text):
    previous, candidate, _ = profiles
    candidate.write_text(text)
    result = CliRunner().invoke(cli.app, [
        "upgrade", "--dry-run", "--config", str(candidate), "--previous-config", str(previous),
    ])
    assert result.exit_code == 2
    assert "secret" not in result.output and str(candidate) not in result.output


def test_no_model_promotion_through_handover(profiles):
    previous, candidate, _ = profiles
    candidate.write_text("policy:\n  weights_path: unqualified-checkpoint\n")
    with pytest.raises(u.UpgradeError, match="separate-qualification"):
        u.prepare_upgrade(candidate, previous)


def test_syntax_failure_is_found_without_import(profiles):
    previous, candidate, module = profiles
    module.write_text("def incomplete(\n")
    with pytest.raises(SyntaxError):
        u.prepare_upgrade(candidate, previous)


def test_editable_factory_mapping_is_resolved_without_running_finder(profiles, monkeypatch):
    previous, candidate, module = profiles
    name = "__editable___fixture_finder"
    finder_module = ModuleType(name)
    finder_module.MAPPING = {"editable_fixture": str(module)}

    class Finder:
        __module__ = name

        @classmethod
        def find_spec(cls, *_args, **_kwargs):
            pytest.fail("offline validation must not invoke an editable finder")

    monkeypatch.setitem(sys.modules, name, finder_module)
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, Finder])
    candidate.write_text(previous.read_text() +
                         "runtime_factory:\n  reference: editable_fixture:build\n")
    plan = u.prepare_upgrade(candidate, previous)
    assert module in plan.files


@pytest.fixture
def handover(tmp_path, monkeypatch):
    """Real supervisor/MotorGate; fake process groups, telemetry and Bedrock."""
    if os.name != "posix":
        pytest.skip("agent handover requires POSIX process ownership and locks")
    previous = tmp_path / "previous.json"
    previous.write_text("{}")
    config = RuntimeConfig()
    candidate = config.model_copy(update={
        "runtime_factory": RuntimeFactoryConfig(reference="candidate_fixture:build"),
    })
    plan = u.UpgradePlan(candidate, config, previous, {}, {}, {
        "plan_sha256": "a" * 64, "checkpoint": "unverified-not-promoted",
        "recorder": "unverified",
    })
    monkeypatch.setattr(u, "app_paths", lambda: SimpleNamespace(
        data_dir=tmp_path / "data", config_file=previous,
    ))
    old = lifecycle.AgentProcess(
        pid=10001, started_ns=time.monotonic_ns(), display=":test", window_id=42,
        instance_id="bedrock:test", role="generalist", capture_source="x11",
        proc_start_ticks=1234, command_sha256="b" * 64,
    )
    # Readiness deadlines are logical time, not a 100 ms host scheduling race.
    origin_ns, wall_ns = time.monotonic_ns(), time.time_ns()
    clock = [origin_ns]
    monkeypatch.setattr(u, "time", SimpleNamespace(
        monotonic=lambda: clock[0] / 1e9, monotonic_ns=lambda: clock[0],
        time_ns=lambda: wall_ns + clock[0] - origin_ns,
    ))

    class ClockEvent(threading.Event):
        def wait(self, timeout=None):
            clock[0] += int((timeout or 0) * 1e9)
            return self.is_set()

    old.persist()
    real = supervisor.Supervisor()
    real.start()
    lease = real.arm(old.instance_id)["lease_id"]
    real.activate()
    real.world_camera_pitch_units = 23
    real.world_camera_origin_calibrated = True
    real.world_camera_calibration_id = "fixture-calibration"
    endpoint = supervisor.ControlEndpoint(
        "127.0.0.1", 1234, "do-not-publish-token", 9, real.session_id,
    )
    real._endpoint = endpoint
    monkeypatch.setattr(supervisor.ControlEndpoint, "load", lambda *args: endpoint)
    groups = {old.pid}
    events = []
    h = SimpleNamespace(
        plan=plan, old=old, supervisor=real, endpoint=endpoint, old_lease=lease,
        groups=groups, events=events, launches=[], telemetry_mode="ready",
        contained=True, fail_candidate=False, after_stop=lambda: None,
        before_launch=lambda: None, before_command=lambda name: None,
        after_telemetry=lambda: None,
    )
    monkeypatch.setattr(u, "_launcher_barrier", lambda pid: "inactive")
    session = SimpleNamespace(display=old.display, find_window=lambda: old.window_id)
    monkeypatch.setattr(u.bedrock_session.BedrockSession, "load", lambda *args: session)
    monkeypatch.setattr(u.bedrock_session, "bedrock_session_alive", lambda *args: True)
    monkeypatch.setattr(u.bedrock_session, "require_autonomous_input_isolation", lambda *a: None)
    monkeypatch.setattr(lifecycle, "agent_alive", lambda process: process.pid in groups)
    monkeypatch.setattr(lifecycle, "_process_group_alive", lambda pid: pid in groups)
    monkeypatch.setattr(lifecycle, "_linux_process_identity", lambda pid: (1234, (
        "python", "-m", "minecraft_ai.agent_process", "--config", str(previous),
        "--lease-id", lease,
    )))

    def command(bound, name, **payload):
        assert bound is endpoint
        h.before_command(name)
        events.append(name)
        responses = []
        connection = SimpleNamespace(close=lambda: None)
        monkeypatch.setattr(supervisor, "_recv_json_line", lambda conn: {
            "token": endpoint.token, "command": name, **payload,
        })
        monkeypatch.setattr(
            supervisor, "_send_json_line", lambda conn, data: responses.append(data),
        )
        real._handle_connection(connection)
        if not responses[0]["ok"]:
            raise u.UpgradeError("supervisor-command-rejected")
        result = responses[0]["result"]
        if name in {"status", "disarm"}:
            result["input_window_id"] = 42
            result["live_capable"] = True
        return result

    monkeypatch.setattr(u, "_command", command)

    def stop(process, *, timeout_s, planned):
        assert timeout_s == 0.2
        assert real.motor.lease is None
        events.append(("stop", process.pid, planned))
        if h.contained:
            groups.discard(process.pid)
            lifecycle._remove_descriptor_if_owned(process)
        h.after_stop()
        return h.contained

    monkeypatch.setattr(lifecycle, "stop_agent_process", stop)

    def launch(**kwargs):
        h.before_launch()
        assert not groups
        assert real.state.value == "RUNNING"
        assert kwargs["config_file"].is_file()
        assert kwargs["display"] == old.display and kwargs["window_id"] == 42
        assert kwargs["instance_id"] == old.instance_id
        assert kwargs["lease_id"] != lease
        h.launches.append(kwargs)
        events.append("launch")
        if h.fail_candidate and len(h.launches) == 1:
            raise RuntimeError("do-not-publish-token /private/fixture-location")
        process = replace(old, pid=10001 + len(h.launches), started_ns=u.time.monotonic_ns(),
                          proc_start_ticks=1234 + len(h.launches))
        groups.add(process.pid)
        process.persist()
        return process

    monkeypatch.setattr(lifecycle, "launch_agent_process", launch)

    def telemetry(*args):
        h.after_telemetry()
        return {
            "lease_id": lease if h.telemetry_mode == "old" else h.launches[-1]["lease_id"],
            "updated_monotonic_ns": 1 if h.telemetry_mode == "stale" else u.time.monotonic_ns(),
            "state": "warming" if h.telemetry_mode == "warming" else "running", "frames": 1,
            "input_release_pending": False,
            "policy_warmup_error": "secret" if h.telemetry_mode == "policy-error" else None,
        }

    monkeypatch.setattr(u, "read_telemetry", telemetry)

    def apply(**kwargs):
        kwargs.setdefault("cancel", ClockEvent())
        return u.activate_upgrade(plan, expected_plan="a" * 64,
                                  drain_timeout_s=0.2, startup_timeout_s=0.1, **kwargs)

    h.apply = apply
    return h


def test_handover_revokes_drains_then_admits_one_owner(handover):
    h = handover
    camera = h.supervisor.status()["world_camera"]
    receipt = h.apply()
    assert receipt["phase"] == "upgraded", receipt
    assert receipt["old_group_contained"] is True
    assert receipt["checkpoint"] == "unverified-not-promoted"
    assert receipt["recorder"] == "unverified"
    assert receipt["before"]["agent_pid"] == h.old.pid
    assert receipt["after"]["agent_pid"] != h.old.pid
    before, after = receipt["before"], receipt["after"]
    assert before["supervisor_generation_sha256"] == after["supervisor_generation_sha256"]
    assert before["lease_generation_sha256"] != after["lease_generation_sha256"]
    assert h.supervisor.status()["world_camera"] == camera
    assert len(h.groups) == 1 and len(h.launches) == 1
    assert h.events.index("disarm") < h.events.index(("stop", h.old.pid, True))
    assert h.events.index(("stop", h.old.pid, True)) < h.events.index("resume-for-agent-reload")
    assert "stop" not in h.events and "resume" not in h.events
    assert not supervisor.operator_pause_latched()
    assert receipt["downtime_s"] >= 0
    encoded = json.dumps(receipt)
    assert h.old_lease not in encoded and h.endpoint.token not in encoded
    assert all(os.stat(p["config_file"]).st_mode & 0o777 == 0o600 for p in h.launches)


def test_candidate_start_failure_rolls_back_to_snapshotted_known_profile(handover):
    h = handover
    def startup_error():
        h.telemetry_mode = "policy-error" if len(h.launches) == 1 else "ready"
    h.after_telemetry = startup_error
    receipt = h.apply()
    assert receipt["phase"] == "rolled-back", receipt
    assert len(h.launches) == 2 and len(h.groups) == 1
    restored = json.loads(h.launches[-1]["config_file"].read_text())
    assert restored == h.plan.previous.model_dump(mode="json")
    assert "private" not in json.dumps(receipt) and "do-not-publish" not in json.dumps(receipt)


def test_spawn_exception_without_identity_never_assumes_cleanup_for_rollback(handover):
    h = handover
    h.fail_candidate = True
    receipt = h.apply()
    assert receipt["phase"] == "blocked"
    assert receipt["error"] == "failed-launch-cleanup-unconfirmed"
    assert len(h.launches) == 1 and h.supervisor.motor.lease is None
    assert "private" not in json.dumps(receipt) and "do-not-publish" not in json.dumps(receipt)


@pytest.mark.parametrize("mode", ["old", "stale", "warming", "policy-error"])
def test_late_old_replies_and_unready_candidate_never_count_as_ready(handover, mode):
    h = handover
    h.telemetry_mode = mode
    receipt = h.apply()
    assert receipt["phase"] == "blocked", receipt
    assert receipt["candidate_error"] in {
        "candidate-readiness-timeout", "candidate-runtime-startup-failed",
    }
    assert not h.groups and h.supervisor.motor.lease is None
    assert receipt["downtime_s"] is None


def test_drain_incomplete_blocks_candidate_and_rollback(handover):
    h = handover
    h.contained = False
    receipt = h.apply()
    assert receipt["phase"] == "blocked" and receipt["error"] == "agent-cleanup-incomplete"
    assert not h.launches and receipt["old_group_contained"] is False
    assert h.supervisor.motor.lease is None and lifecycle.AGENT_FILE.exists()


@pytest.mark.parametrize("when", ["before", "drain", "startup"])
@pytest.mark.parametrize("kind", ["pause", "emergency", "cancel"])
def test_operator_intent_wins_at_every_phase(handover, when, kind, monkeypatch):
    h = handover
    cancelled = threading.Event()
    def interrupt():
        if kind == "pause":
            supervisor.latch_operator_pause()
        elif kind == "emergency":
            monkeypatch.setattr(u, "emergency_stop_latched", lambda: True)
        else:
            cancelled.set()
    if when == "before":
        interrupt()
    elif when == "drain":
        h.after_stop = interrupt
    else:
        h.before_launch = interrupt
    receipt = h.apply(cancel=cancelled)
    assert receipt["phase"] in {"blocked", "cancelled"}
    assert len(h.launches) <= (1 if when == "startup" else 0)
    if when != "before":
        assert h.supervisor.motor.lease is None
    if kind == "pause":
        assert supervisor.operator_pause_latched()
    assert "resume" not in h.events


def test_guarded_resume_never_replaces_faulted_supervisor(handover):
    h = handover
    h.after_stop = lambda: h.supervisor.fail("test-fault")
    receipt = h.apply()
    assert receipt["phase"] == "blocked" and not h.launches
    assert h.supervisor.session_id == h.endpoint.session_id
    assert h.supervisor.state.value == "FAILSAFE"


def test_missing_reload_capability_fails_before_revocation(handover, monkeypatch):
    h = handover
    original = h.supervisor.status
    monkeypatch.setattr(h.supervisor, "status", lambda: {
        **original(), "agent_reload_resume_supported": False,
    })
    receipt = h.apply()
    assert receipt["phase"] == "blocked"
    assert "disarm" not in h.events and not h.launches


def test_plan_mismatch_is_not_permission_to_touch_live_owner(handover):
    h = handover
    with pytest.raises(u.UpgradeError, match="dry-run-plan-mismatch"):
        u.activate_upgrade(h.plan, expected_plan="bad")
    assert not h.events


def test_status_reads_historical_receipt_without_live_ipc(handover, monkeypatch):
    receipt = handover.apply()
    monkeypatch.setattr(u, "_command", lambda *a, **kw: pytest.fail("status must not mutate"))
    result = CliRunner().invoke(cli.app, ["upgrade", "--status"])
    assert result.exit_code == 0
    assert json.loads(result.output) == receipt


def test_pinned_ipc_refuses_replaced_supervisor_before_any_connection(monkeypatch):
    endpoint = supervisor.ControlEndpoint("127.0.0.1", 1, "secret", 1, "old")
    monkeypatch.setattr(
        supervisor.ControlEndpoint, "load", lambda: replace(endpoint, session_id="new"),
    )
    monkeypatch.setattr(u.socket, "create_connection", lambda *a, **kw: pytest.fail("wrong owner"))
    with pytest.raises(u.UpgradeError, match="generation-changed"):
        u._command(endpoint, "disarm")


def test_active_launcher_requires_parent_barrier(monkeypatch):
    monkeypatch.setattr(u, "persistent_agent_service_load_state", lambda: "loaded")
    monkeypatch.setattr(u, "persistent_agent_service_state", lambda: "active")
    with pytest.raises(u.UpgradeError, match="parent-held-launcher-barrier-required"):
        u._launcher_barrier(None)


def test_cli_sigterm_is_sticky_cancellation_not_an_automatic_resume(profiles, monkeypatch):
    previous, candidate, _ = profiles
    original = signal.getsignal(signal.SIGTERM)
    def activate(plan, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert kwargs["cancel"].is_set()
        return {"phase": "cancelled"}
    monkeypatch.setattr(u, "activate_upgrade", activate)
    result = CliRunner().invoke(cli.app, [
        "upgrade", "--config", str(candidate), "--previous-config", str(previous),
        "--expect-plan", "a",
    ])
    assert result.exit_code == 1 and json.loads(result.output)["phase"] == "cancelled"
    assert signal.getsignal(signal.SIGTERM) == original


def test_candidate_cleanup_incomplete_does_not_attempt_rollback_or_repeat_drain(handover):
    h = handover
    h.telemetry_mode = "policy-error"
    h.after_telemetry = lambda: setattr(h, "contained", False)
    receipt = h.apply()
    assert receipt["phase"] == "blocked"
    assert receipt["error"] == "agent-cleanup-incomplete"
    assert len(h.launches) == 1
    stops = [event for event in h.events if isinstance(event, tuple)]
    assert len(stops) == 2  # Old retirement and exactly one candidate cleanup.
    assert h.supervisor.motor.lease is None


def test_pause_during_readiness_reply_cannot_publish_success(handover):
    h = handover
    h.after_telemetry = supervisor.latch_operator_pause
    receipt = h.apply()
    assert receipt["phase"] == "blocked" and len(h.launches) == 1
    assert not h.groups and supervisor.operator_pause_latched()
    assert h.supervisor.motor.lease is None


def test_crashed_known_candidate_can_be_contained_before_rollback(handover):
    h = handover
    def die_once():
        if len(h.launches) == 1:
            h.groups.clear()
            lifecycle.AGENT_FILE.unlink()
            h.telemetry_mode = "policy-error"
        else:
            h.telemetry_mode = "ready"
    h.after_telemetry = die_once
    receipt = h.apply()
    assert receipt["phase"] == "rolled-back", receipt
    assert len(h.launches) == 2 and len(h.groups) == 1


def test_lost_supervisor_does_not_abandon_known_candidate(handover):
    h = handover
    def lose_endpoint():
        def reject(name):
            raise u.UpgradeError("supervisor-generation-changed")
        h.before_command = reject
        h.telemetry_mode = "policy-error"
        # Model a supervisor exit, which revokes independently of this client.
        h.supervisor.disarm()
    h.after_telemetry = lose_endpoint
    receipt = h.apply()
    assert receipt["phase"] == "blocked" and len(h.launches) == 1
    assert not h.groups
    assert ("stop", 10002, False) in h.events


def test_concurrent_handover_is_refused_before_any_live_command(handover):
    h = handover
    with u.bedrock_session.bedrock_lifecycle_lock():
        with pytest.raises(u.bedrock_session.IsolationError, match="already active"):
            h.apply()
    assert not h.events and not h.launches


@pytest.mark.parametrize("change", ["session", "window", "calibration", "agent"])
def test_generation_drift_after_drain_never_admits_candidate(handover, monkeypatch, change):
    h = handover
    def drift():
        if change == "session":
            h.supervisor.session_id = "replacement"
        elif change == "window":
            monkeypatch.setattr(u.bedrock_session.BedrockSession, "load", lambda: object())
        elif change == "calibration":
            h.supervisor.world_camera_calibration_id = "replacement"
        else:
            replace(h.old, pid=20001, proc_start_ticks=999).persist()
    h.after_stop = drift
    receipt = h.apply()
    assert receipt["phase"] == "blocked" and not h.launches
    assert h.supervisor.motor.lease is None


def test_changed_previous_profile_is_not_claimed_as_known_rollback(handover):
    h = handover
    h.plan.previous_path.write_text('{"cognition_hz": 1}')
    receipt = h.apply()
    assert receipt["phase"] == "blocked"
    assert receipt["error"] == "previous-profile-was-edited-after-launch"
    assert "disarm" not in h.events


def test_late_old_lease_cannot_actuate_after_new_admission(handover):
    h = handover
    receipt = h.apply()
    assert receipt["phase"] == "upgraded"
    before = h.supervisor.backend.release_count
    with pytest.raises(RuntimeError, match="invalid or expired"):
        h.supervisor.apply_motor_action(h.old_lease, {"sequence": 1, "keys_down": ["w"]})
    assert not h.supervisor.backend.held_keys
    # Existing protocol rejects stale capabilities by failing closed, not by
    # preserving availability. Do not claim uninterrupted new-owner operation.
    assert h.supervisor.state.value == "FAILSAFE"
    assert h.supervisor.backend.release_count > before


@pytest.mark.parametrize("stage", ["prepared", "starting-candidate"])
def test_audit_failure_does_not_abandon_revoked_owner_or_start_candidate(
    handover, monkeypatch, stage,
):
    h = handover
    write = u._write_json
    def fail(path, payload):
        if payload.get("phase") in {stage, "blocked"}:
            raise OSError("do-not-publish-this-location")
        write(path, payload)
    monkeypatch.setattr(u, "_write_json", fail)
    receipt = h.apply()
    assert receipt["phase"] == "blocked" and not h.launches
    assert receipt["audit_error"] == "OSError"
    if stage == "starting-candidate":
        assert not h.groups and receipt["old_group_contained"] is True
    else:
        assert "disarm" not in h.events
    assert "do-not-publish" not in json.dumps(receipt)


def test_uncertain_revocation_still_contains_old_group_without_rollback(handover, monkeypatch):
    h = handover
    command = u._command
    def lost_reply(endpoint, name, **kwargs):
        result = command(endpoint, name, **kwargs)
        if name == "disarm":
            raise TimeoutError("reply lost after revocation")
        return result
    monkeypatch.setattr(u, "_command", lost_reply)
    receipt = h.apply()
    assert receipt["phase"] == "blocked"
    assert receipt["old_group_contained"] is True
    assert not h.groups and not h.launches
    assert ("stop", h.old.pid, False) in h.events


@pytest.mark.parametrize("bad", [None, "commit", "weights"])
def test_policy_pins_are_checked_as_bytes_without_loading_models(tmp_path, monkeypatch, bad):
    source = tmp_path / "source"
    source.mkdir()
    (source / "worker.py").write_text("raise AssertionError('must not execute')\n")
    model, weights = tmp_path / "model", tmp_path / "weights"
    model.write_bytes(b"not a deserializable model")
    weights.write_bytes(b"not deserializable weights")
    policy = {
        "enabled": True, "source_path": str(source), "python_path": sys.executable,
        "model_path": str(model), "weights_path": str(weights),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "source_commit": "a" * 40, "license": "mit", "model_version": "fixture",
    }
    if bad == "weights":
        policy["weights_sha256"] = "0" * 64
    monkeypatch.setattr(u.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=("b" if bad == "commit" else "a") * 40,
    ))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"policy": policy}))
    if bad:
        with pytest.raises(u.UpgradeError, match="mismatch"):
            u.prepare_upgrade(profile, profile)
    else:
        plan = u.prepare_upgrade(profile, profile)
        assert model in plan.files and weights in plan.files
        weights.write_bytes(b"changed after validation")
        with pytest.raises(u.UpgradeError, match="pinned-input-changed"):
            plan.check_unchanged()


@pytest.mark.parametrize("fault", [None, "running", "child", "mainpid", "service-transition"])
@pytest.mark.skipif(os.name != "posix", reason="launcher barrier uses POSIX proc paths")
def test_parent_held_launcher_requires_exact_quiescent_shell(monkeypatch, fault):
    monkeypatch.setattr(u, "persistent_agent_service_load_state", lambda: "loaded")
    monkeypatch.setattr(u, "persistent_agent_service_state", lambda: (
        "unknown" if fault == "service-transition" else "active"
    ))
    monkeypatch.setattr(u.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout="2" if fault == "mainpid" else "10001",
    ))
    monkeypatch.setattr(lifecycle, "_linux_process_identity", lambda pid: (
        9, ("bash", "launcher.sh") if pid == 10001 else (
            "python" if fault == "child" else "sleep", "10",
        ),
    ))
    original_read, original_resolve = Path.read_text, Path.resolve
    def read(path, *args, **kwargs):
        fake = {
            "/proc/10001/stat": "10001 (bash) " + ("S" if fault == "running" else "T"),
            "/proc/10001/task/10001/children": "10002",
            "/proc/10002/task/10002/children": "",
        }
        return fake[str(path)] if str(path) in fake else original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(u.shutil, "which", lambda name: "/fixture/sleep")
    monkeypatch.setattr(Path, "resolve", lambda path, *args, **kwargs: (
        Path("/fixture/sleep") if str(path) == "/proc/10002/exe"
        else original_resolve(path, *args, **kwargs)
    ))
    monkeypatch.setattr(u.os, "kill", lambda *args: pytest.fail("barrier must not signal"))
    if fault:
        with pytest.raises(u.UpgradeError):
            u._launcher_barrier(10001)
    else:
        assert len(u._launcher_barrier(10001)) == 64
