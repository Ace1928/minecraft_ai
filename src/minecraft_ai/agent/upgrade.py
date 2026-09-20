"""Two-phase, cold agent-only handover using the existing process/lease protocol.

No factory import, inference or actuator ownership is available to preflight.
Rollback restores an explicitly supplied known profile on the *pinned current*
source tree, not old Python objects, learned state or a previous source release.
Keep both factory implementations available under distinct references when needed.
The deployment owner must keep sources/artifacts immutable during activation.

An active persistent launcher must already be held by its deployment owner:
only its stopped shell and idle sleep children are accepted. This module never
stops/resumes that shell, a service, Bedrock, a model server or the supervisor.
The owner releases the barrier only after separate gameplay qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from importlib.machinery import PathFinder
from importlib.util import spec_from_file_location
from pathlib import Path
from typing import Any

import typer
import yaml

from minecraft_ai import supervisor
from minecraft_ai.agent import lifecycle
from minecraft_ai.config import RuntimeConfig, app_paths
from minecraft_ai.emergency import emergency_stop_latched
from minecraft_ai.operator.service_control import (
    PERSISTENT_AGENT_SERVICE,
    persistent_agent_service_load_state,
    persistent_agent_service_state,
)
from minecraft_ai.operator.telemetry import read_telemetry
from minecraft_ai.platforms import bedrock_session
from minecraft_ai.roles import get_role


class UpgradeError(RuntimeError):
    """A public, path/credential-free refusal code."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _sources(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if (
        path.is_file() and path.suffix in {".py", ".so"}
        and not {".git", "__pycache__"}.intersection(path.relative_to(root).parts)
    )))


def _bundle_files(root: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    if not paths or any(not path.resolve().is_relative_to(root.resolve()) for path in paths):
        raise UpgradeError("invalid-artifact-bundle")
    return paths


@dataclass
class UpgradePlan:
    candidate: RuntimeConfig = field(repr=False)
    previous: RuntimeConfig = field(repr=False)
    previous_path: Path = field(repr=False)
    files: dict[Path, tuple[int, ...]] = field(repr=False)
    roots: dict[Path, tuple[Path, ...]] = field(repr=False)
    receipt: dict[str, Any]
    bundles: dict[Path, tuple[Path, ...]] = field(default_factory=dict, repr=False)

    def check_unchanged(self) -> None:
        if (any(_stamp(path) != stamp for path, stamp in self.files.items())
                or any(_sources(root) != paths for root, paths in self.roots.items())
                or any(_bundle_files(root) != paths for root, paths in self.bundles.items())):
            raise UpgradeError("pinned-input-changed")


def prepare_upgrade(candidate_path: Path, previous_path: Path) -> UpgradePlan:
    """Validate without importing candidate code, opening models or contacting IPC.

    Artifact bytes are streamed only for SHA-256 verification; no deserialization.
    Source identity covers Python/native files in the resolved factory package,
    public runtime package and configured policy source roots, not all dependencies.
    """
    files: dict[Path, tuple[int, ...]] = {}
    roots: dict[Path, tuple[Path, ...]] = {}
    identities: dict[str, str] = {}
    runtime_sources: dict[str, str] = {}
    source_declarations: list[dict[str, Any]] = []
    bundles: dict[Path, tuple[Path, ...]] = {}

    def pin(path: Path, *, syntax: bool = False) -> str:
        path = path.absolute()
        before = _stamp(path)
        if not path.is_file():
            raise UpgradeError("pin-is-not-a-file")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if syntax:
            compile(path.read_bytes(), "<upgrade-source>", "exec")
        if _stamp(path) != before:
            raise UpgradeError("pin-changed-during-validation")
        files[path] = before
        identities[str(path)] = digest
        return digest

    def config(path: Path) -> RuntimeConfig:
        pin(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise UpgradeError("configuration-must-be-a-mapping")
        result = RuntimeConfig.model_validate(raw)
        get_role(result.role)
        return result

    candidate_path = candidate_path.expanduser().absolute()
    previous_path = previous_path.expanduser().absolute()
    candidate, previous = config(candidate_path), config(previous_path)
    # Handover is not model promotion or camera requalification. Preserve the
    # entire body configuration, including external argv and calibration pins.
    for name in ("policy", "grounded_policy", "gui_policy", "raw_motion_policy",
                 "high_level", "vision_language", "role"):
        if getattr(candidate, name) != getattr(previous, name):
            raise UpgradeError("model-body-or-role-change-requires-separate-qualification")

    source_roots = {Path(__file__).resolve().parents[1]}
    for selected in (candidate, previous):
        factory = selected.runtime_factory
        if factory is not None:
            # Unlike util.find_spec('package.child'), this never imports parents.
            search = list(sys.path)
            for index, part in enumerate(factory.reference.split(":")[0].split(".")):
                spec = PathFinder.find_spec(part, search)
                if index == 0 and spec is None:
                    # PEP 660 setuptools installs keep import roots in an
                    # already-loaded finder module, not sys.path. Read its
                    # static mapping; never invoke a finder or import a factory.
                    locations = set()
                    for finder in sys.meta_path:
                        module_name = getattr(finder, "__module__", "")
                        module = sys.modules.get(module_name)
                        if not module_name.startswith("__editable___") or module is None:
                            continue
                        mapping = vars(module).get("MAPPING")
                        if type(mapping) is dict and type(mapping.get(part)) is str:
                            locations.add(Path(mapping[part]))
                    if len(locations) > 1:
                        raise UpgradeError("ambiguous-editable-factory-source")
                    if locations:
                        mapped_root = locations.pop()
                        source = (mapped_root / "__init__.py" if mapped_root.is_dir()
                                  else mapped_root.with_suffix(".py"))
                        spec = spec_from_file_location(part, source)
                if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
                    raise UpgradeError("factory-requires-resolvable-python-source")
                if index == 0:
                    root = Path(spec.origin).resolve()
                    source_roots.add(root.parent if spec.submodule_search_locations else root)
                search = list(spec.submodule_search_locations or ())
        for name in ("high_level", "vision_language"):
            model = getattr(selected, name)
            if model.enabled and not model.model_id:
                raise UpgradeError("enabled-model-requires-identity")
        if selected.mining_rule_snapshot:
            pin(Path(selected.mining_rule_snapshot).expanduser())

    from minecraft_ai.policy_service import _validate_policy_config

    for slot in ("policy", "grounded_policy", "gui_policy", "raw_motion_policy"):
        policy = getattr(candidate, slot)
        if policy is None or not policy.enabled:
            continue
        _validate_policy_config(policy)
        root = Path(policy.source_path).expanduser().resolve(strict=True)
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=5.0,
        )
        head = completed.stdout.strip()
        if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
            raise UpgradeError("invalid-runtime-source-commit")
        # source_commit predates handover and can describe historical provider
        # provenance. Preserve it verbatim in the private config snapshot, never
        # relabel it as current runtime or model-training provenance.
        source_declarations.append({
            "slot": slot, "declared_source_commit_sha256": _digest(policy.source_commit),
            "runtime_commit": head, "declaration_matches_runtime": head == policy.source_commit,
        })
        source_roots.add(root)
        pin(Path(policy.python_path))
        model_path, weights_path = Path(policy.model_path), Path(policy.weights_path)
        if model_path.is_dir():
            # External workers may bind an opaque bundle by its explicitly
            # configured descriptor. Hash every member; format/lineage validation
            # remains the worker's contract, not a public model-specific loader.
            if (policy.provider != "external" or not weights_path.is_file()
                    or not weights_path.resolve().is_relative_to(model_path.resolve())
                    or policy.model_sha256 != policy.weights_sha256):
                raise UpgradeError("unbound-artifact-bundle")
            bundles[model_path] = _bundle_files(model_path)
            for member in bundles[model_path]:
                pin(member)
        for location, expected in ((policy.model_path, policy.model_sha256),
                                   (policy.weights_path, policy.weights_sha256),
                                   (policy.scene_model_path, policy.scene_model_sha256)):
            if location == policy.model_path and model_path.is_dir():
                continue
            if location and pin(Path(location)) != expected:
                raise UpgradeError("artifact-digest-mismatch")

    for root in sorted(source_roots):
        paths = _sources(root) if root.is_dir() else (root,)
        if root.is_dir():
            roots[root] = paths
        for path in paths:
            runtime_sources[str(path)] = pin(path, syntax=path.suffix == ".py")
    pin(Path(sys.executable))
    receipt = {
        "schema_version": 1,
        "mode": "cold-agent-only",
        "candidate_config_sha256": _digest(candidate.model_dump(mode="json")),
        "previous_config_sha256": _digest(previous.model_dump(mode="json")),
        "source_and_artifacts_sha256": _digest(identities),
        "runtime_source_sha256": _digest({
            "files": runtime_sources, "declarations": source_declarations,
            "import_path": sys.path, "pythonpath": os.environ.get("PYTHONPATH", ""),
        }),
        "source_declarations": source_declarations,
        "source_review_required": any(
            not declaration["declaration_matches_runtime"] for declaration in source_declarations
        ),
        "artifact_bundles": len(bundles),
        "bundle_verification": "descriptor-pinned-members-plan-bound-worker-validates-format",
        "prewarm": "unsupported-without-live-lease",
        "rollback_scope": "previous-profile-on-pinned-current-source",
        "checkpoint": "unverified-not-promoted",
        "recorder": "unverified",
        "readiness": "fresh-runtime-telemetry-not-gameplay-qualification",
    }
    receipt["plan_sha256"] = _digest({
        **receipt, "interpreter": sys.executable, "prefix": sys.prefix,
        "import_path": sys.path, "pythonpath": os.environ.get("PYTHONPATH", ""),
    })
    plan = UpgradePlan(candidate, previous, previous_path, files, roots, receipt, bundles)
    plan.check_unchanged()
    return plan


def _command(endpoint: supervisor.ControlEndpoint, command: str, **payload: Any) -> dict[str, Any]:
    """Use the pinned authenticated endpoint, never rediscover one for mutation."""
    if (supervisor.ControlEndpoint.load() != endpoint
            or supervisor.control_endpoint_process_state(endpoint) != "verified-live"):
        raise UpgradeError("supervisor-generation-changed")
    with socket.create_connection((endpoint.host, endpoint.port), timeout=0.5) as sock:
        supervisor._send_json_line(sock, {
            "token": endpoint.token, "command": command, **payload,
        })
        response = supervisor._recv_json_line(sock)
    result = response.get("result")
    if response.get("ok") is not True or not isinstance(result, dict):
        raise UpgradeError("supervisor-command-rejected")
    if supervisor.ControlEndpoint.load() != endpoint:
        raise UpgradeError("supervisor-generation-changed")
    return result


def _launcher_barrier(pid: int | None) -> str:
    """Verify an externally held recovery shell; never send it a signal."""
    state = persistent_agent_service_load_state()
    active = persistent_agent_service_state() if state == "loaded" else "unknown"
    if state == "not-found" or (state == "loaded" and active == "inactive"):
        if pid is not None:
            raise UpgradeError("launcher-service-not-active")
        return "inactive"
    if state != "loaded" or active != "active" or pid is None:
        raise UpgradeError("parent-held-launcher-barrier-required")
    result = subprocess.run(
        ["systemctl", "--user", "show", "--property=MainPID", "--value",
         PERSISTENT_AGENT_SERVICE], check=True, capture_output=True, text=True, timeout=3.0,
    )
    identity = lifecycle._linux_process_identity(pid)
    if result.stdout.strip() != str(pid) or identity is None:
        raise UpgradeError("launcher-identity-unconfirmed")
    command = identity[1]
    stat = Path(f"/proc/{pid}/stat").read_text()
    if (Path(command[0]).name != "bash" or len(command) != 2
            or stat[stat.rfind(")") + 1:].split()[0] != "T"):
        raise UpgradeError("launcher-is-not-a-held-shell")
    children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    for child in children:
        child_command = lifecycle._linux_process_identity(int(child))
        if child_command is None:
            try:
                child_stat = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                continue
            # A stopped shell cannot reap its finished sleep child. An exact
            # dead child is not an active recovery operation or a reason to thaw.
            if child_stat[0] in {"Z", "X"} and int(child_stat[1]) == pid:
                continue
        sleep = shutil.which("sleep")
        if (child_command is None or Path(child_command[1][0]).name != "sleep"
                or sleep is None or Path(f"/proc/{child}/exe").resolve() != Path(sleep).resolve()
                or Path(f"/proc/{child}/task/{child}/children").read_text().strip()):
            raise UpgradeError("launcher-has-active-recovery-child")
    return _digest(identity)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        staged.replace(path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        staged.unlink(missing_ok=True)


def _summary(status: dict[str, Any], process: lifecycle.AgentProcess | None) -> dict[str, Any]:
    # Deliberate allowlist: no argv, endpoint token, lease capability, exception
    # text, profile contents, model paths or arbitrary adapter metadata.
    return {
        "supervisor_generation_sha256": _digest(status.get("session_id")),
        "state": (status.get("state") if status.get("state") in {
            "STARTING", "SAFE_IDLE", "ARMED", "RUNNING", "PAUSED", "FAILSAFE",
            "STOPPING", "STOPPED",
        } else "UNKNOWN"),
        "motor_lease_active": status.get("motor_lease_active") is True,
        "lease_generation_sha256": _digest(status.get("motor_lease_id")),
        "agent_pid": None if process is None else process.pid,
        "agent_start_ticks": None if process is None else process.proc_start_ticks,
        "camera_state_sha256": _digest(status.get("world_camera")),
    }


def activate_upgrade(
    plan: UpgradePlan, *, expected_plan: str, drain_timeout_s: float = 25.0,
    startup_timeout_s: float = 120.0, launcher_pid: int | None = None,
    cancel: threading.Event | None = None,
    reviewed_source_sha256: str = "",
) -> dict[str, Any]:
    """Bound explicit waits; fail closed if containment or intent is ambiguous.

    The operator lock is never held over drain/readiness waits. There is no
    generic FAILSAFE recovery: it cannot preserve this supervisor generation.
    Checkpoint/recorder success is not implied by process-group disappearance.
    """
    if expected_plan != plan.receipt["plan_sha256"]:
        raise UpgradeError("dry-run-plan-mismatch")
    if (plan.receipt.get("source_review_required") or reviewed_source_sha256) and (
        not reviewed_source_sha256
        or reviewed_source_sha256 != plan.receipt.get("runtime_source_sha256")
    ):
        raise UpgradeError("exact-reviewed-runtime-source-required")
    if not 0.1 <= drain_timeout_s <= 25.0 or not 0.1 <= startup_timeout_s <= 600.0:
        raise UpgradeError("invalid-wait-budget")
    cancel = threading.Event() if cancel is None else cancel
    receipt = {**plan.receipt, "transaction_id": uuid.uuid4().hex,
               "phase": "preflight", "events": [], "before": None, "after": None,
               "drain_timeout_s": drain_timeout_s, "startup_timeout_s": startup_timeout_s,
               "downtime_s": None, "old_group_contained": False}
    receipt["reviewed_source_sha256"] = reviewed_source_sha256 or None
    directory = app_paths().data_dir / "upgrades" / receipt["transaction_id"]
    latest = lifecycle.RUNTIME_DIR / "upgrade.json"
    endpoint: supervisor.ControlEndpoint | None = None
    before: dict[str, Any] = {}
    current: lifecycle.AgentProcess | None = None
    lease_id: str | None = None
    revoked_at: float | None = None
    admission_started = False
    old_retirement_attempted = False
    old_revoked = False

    def record(phase: str) -> None:
        receipt["phase"] = phase
        if revoked_at is not None:
            receipt["elapsed_since_revoke_s"] = round(time.monotonic() - revoked_at, 3)
        receipt["events"].append({"phase": phase, "wall_ns": time.time_ns()})
        _write_json(directory / "receipt.json", receipt)
        _write_json(latest, receipt)

    def permitted() -> None:
        if cancel.is_set() or emergency_stop_latched() or supervisor.operator_pause_latched():
            raise UpgradeError("operator-interrupted")

    def status() -> dict[str, Any]:
        assert endpoint is not None
        result = _command(endpoint, "status")
        if result.get("session_id") != endpoint.session_id:
            raise UpgradeError("supervisor-generation-changed")
        if before:
            if any(result.get(key) != before.get(key) for key in (
                "backend", "live_capable", "input_window_id",
            )):
                raise UpgradeError("supervisor-input-binding-changed")
            camera = result.get("world_camera")
            if not isinstance(camera, dict) or any(
                camera.get(key) != before["world_camera"].get(key)
                for key in ("origin_calibrated", "calibration_id", "pitch_counts_per_degree")
            ):
                raise UpgradeError("supervisor-calibration-changed")
        return result

    def environment() -> None:
        permitted()
        plan.check_unchanged()
        if _launcher_barrier(launcher_pid) != barrier:
            raise UpgradeError("launcher-barrier-changed")
        if (bedrock_session.BedrockSession.load() != session
                or not bedrock_session.bedrock_session_alive(session)
                or session.find_window() != old.window_id):
            raise UpgradeError("bedrock-generation-changed")

    def contain(process: lifecycle.AgentProcess, *, revoked: bool = True) -> None:
        lifecycle.stop_agent_process(
            process, timeout_s=drain_timeout_s, planned=revoked and not emergency_stop_latched(),
        )
        if (lifecycle._process_group_alive(process.pid) or lifecycle.AGENT_FILE.exists()
                or lifecycle.AGENT_FILE.is_symlink()):
            raise UpgradeError("agent-cleanup-incomplete")

    def start_profile(profile: RuntimeConfig, path: Path) -> None:
        nonlocal current, lease_id, admission_started
        deadline = time.monotonic() + startup_timeout_s
        environment()
        assert endpoint is not None
        observed = status()
        if observed.get("state") != "PAUSED" or observed.get("motor_lease_id") is not None:
            raise UpgradeError("rollback-or-admission-requires-cleanup-pause")
        # This IPC owns the operator lock itself. Do not deadlock it by holding
        # the same cross-process lock in the caller.
        _command(endpoint, "resume-for-agent-reload", session_id=endpoint.session_id)
        with supervisor.operator_intent_lock():
            environment()
            observed = status()
            if (observed.get("state") != "SAFE_IDLE"
                    or observed.get("motor_lease_id") is not None
                    or lifecycle.AGENT_FILE.exists() or lifecycle.AGENT_FILE.is_symlink()):
                raise UpgradeError("another-owner-won-admission")
            admission_started = True
            armed = _command(endpoint, "arm", target_instance=old.instance_id)
            lease_id = armed["lease"]["lease_id"]
            if not isinstance(lease_id, str) or not lease_id or lease_id == old_lease:
                raise UpgradeError("invalid-new-lease-generation")
            _command(endpoint, "activate")
            permitted()
            current = lifecycle.launch_agent_process(
                lease_id=lease_id, display=old.display, window_id=old.window_id,
                instance_id=old.instance_id, role=profile.role, config_file=path,
                allow_host_capture=False, capture_source=old.capture_source,
            )
        while time.monotonic() < deadline:
            environment()
            observed = status()
            if (lifecycle.AgentProcess.load() != current or not lifecycle.agent_alive(current)
                    or observed.get("state") != "RUNNING"
                    or observed.get("motor_lease_id") != lease_id
                    or observed.get("motor_lease_active") is not True):
                raise UpgradeError("candidate-owner-lost")
            telemetry = read_telemetry(lifecycle.RUNTIME_DIR / "telemetry.json") or {}
            updated = telemetry.get("updated_monotonic_ns")
            frames = telemetry.get("frames")
            if (telemetry.get("lease_id") == lease_id and type(updated) is int
                    and current.started_ns <= updated <= time.monotonic_ns()
                    and time.monotonic_ns() - updated < 2_000_000_000):
                if telemetry.get("policy_warmup_error") or telemetry.get("lease_heartbeat_error"):
                    raise UpgradeError("candidate-runtime-startup-failed")
                if (telemetry.get("state") == "running" and type(frames) is int and frames > 0
                        and telemetry.get("input_release_pending") is False):
                    permitted()
                    observed = status()
                    if (observed.get("state") != "RUNNING"
                            or observed.get("motor_lease_id") != lease_id
                            or observed.get("motor_lease_active") is not True
                            or lifecycle.AgentProcess.load() != current
                            or not lifecycle.agent_alive(current)):
                        raise UpgradeError("candidate-owner-lost")
                    if time.monotonic() >= deadline:
                        break
                    assert revoked_at is not None
                    receipt["after"] = _summary(observed, current)
                    receipt["downtime_s"] = round(time.monotonic() - revoked_at, 3)
                    return
            cancel.wait(0.05)
        raise UpgradeError("candidate-readiness-timeout")

    def cleanup_candidate() -> None:
        nonlocal current, admission_started
        assert endpoint is not None
        # Never repeat a full drain budget if this attempt cannot confirm cleanup.
        admission_started = False
        try:
            descriptor = lifecycle.AgentProcess.load()
        except FileNotFoundError:
            descriptor = None
        if descriptor is not None and descriptor != current:
            raise UpgradeError("candidate-descriptor-changed")
        revoked = False
        try:
            with supervisor.operator_intent_lock():
                observed = status()
                if observed.get("motor_lease_id") not in {None, lease_id}:
                    raise UpgradeError("candidate-lease-changed")
                released = _command(endpoint, "disarm")
                revoked = (released.get("motor_lease_id") is None
                           and released.get("motor_lease_active") is False
                           and released.get("held_keys") == []
                           and released.get("held_buttons") == [])
                if not revoked:
                    raise UpgradeError("candidate-release-unconfirmed")
        finally:
            # A dead supervisor cannot excuse abandoning a known child group.
            # Emergency/unconfirmed release selects group-first containment.
            if current is not None:
                contain(current, revoked=revoked)
        if current is None:
            # launch_agent_process does not return a typed cleanup receipt on
            # exception. Descriptor absence alone cannot certify an unknown PID.
            raise UpgradeError("failed-launch-cleanup-unconfirmed")
        current = None

    with bedrock_session.bedrock_lifecycle_lock():
        try:
            permitted()
            plan.check_unchanged()
            barrier = _launcher_barrier(launcher_pid)
            endpoint = supervisor.ControlEndpoint.load()
            before = status()
            old = lifecycle.AgentProcess.load()
            receipt["before"] = _summary(before, old)
            old_lease = before.get("motor_lease_id")
            camera = before.get("world_camera")
            if (before.get("agent_reload_resume_supported") is not True
                    or before.get("state") != "RUNNING" or not old_lease
                    or before.get("live_capable") is not True
                    or before.get("motor_lease_active") is not True
                    or before.get("motor_target_instance") != old.instance_id
                    or before.get("input_window_id") != old.window_id
                    or not lifecycle.agent_alive(old) or old.allow_host_capture
                    or not isinstance(camera, dict)
                    or camera.get("origin_calibrated") is not True
                    or not camera.get("calibration_id")):
                raise UpgradeError("existing-owner-not-qualified-for-reload")
            identity = lifecycle._linux_process_identity(old.pid)
            if identity is None:
                raise UpgradeError("old-agent-identity-unconfirmed")
            command = identity[1]
            selected = (Path(command[command.index("--config") + 1]) if "--config" in command
                        else app_paths().config_file)
            if (selected.resolve() != plan.previous_path.resolve()
                    or command[command.index("--lease-id") + 1] != old_lease
                    or old.role != plan.previous.role):
                raise UpgradeError("previous-profile-does-not-match-old-launch")
            # Existing agents do not report config hashes. Reject a profile
            # edited since launch instead of pretending it is a known rollback.
            launched_wall_ns = time.time_ns() - (time.monotonic_ns() - old.started_ns)
            profile_stat = plan.previous_path.stat()
            if max(profile_stat.st_mtime_ns, profile_stat.st_ctime_ns) > launched_wall_ns:
                raise UpgradeError("previous-profile-was-edited-after-launch")
            session = bedrock_session.BedrockSession.load()
            bedrock_session.require_autonomous_input_isolation(session)
            if session.display != old.display:
                raise UpgradeError("bedrock-binding-mismatch")
            environment()
            receipt["launcher_barrier"] = "inactive" if barrier == "inactive" else "parent-held"
            candidate_file = directory / "candidate.json"
            previous_file = directory / "previous.json"
            _write_json(candidate_file, plan.candidate.model_dump(mode="json"))
            _write_json(previous_file, plan.previous.model_dump(mode="json"))
            plan.files[candidate_file] = _stamp(candidate_file)
            plan.files[previous_file] = _stamp(previous_file)
            record("prepared")
            # Journal intent before revocation, not between disarm and TERM:
            # old ticks must see sticky stop promptly after losing their lease.
            record("draining")
            with supervisor.operator_intent_lock():
                environment()
                if (lifecycle.AgentProcess.load() != old
                        or status().get("motor_lease_id") != old_lease):
                    raise UpgradeError("old-generation-changed-before-revoke")
                revoked_at = time.monotonic()
                revoked = _command(endpoint, "disarm")
                if (revoked.get("motor_lease_active") is not False
                        or revoked.get("motor_lease_id") is not None
                        or revoked.get("held_keys") != [] or revoked.get("held_buttons") != []):
                    raise UpgradeError("actuator-revocation-unconfirmed")
                old_revoked = True
            old_retirement_attempted = True
            contain(old)
            receipt["old_group_contained"] = True
            record("starting-candidate")
            try:
                start_profile(plan.candidate, candidate_file)
            except Exception as exc:
                receipt["candidate_error"] = (
                    str(exc) if isinstance(exc, UpgradeError) else type(exc).__name__
                )
                if admission_started:
                    cleanup_candidate()
                environment()
                record("rolling-back")
                start_profile(plan.previous, previous_file)
                record("rolled-back")
            else:
                record("upgraded")
        except BaseException as exc:
            receipt["error"] = str(exc) if isinstance(exc, UpgradeError) else type(exc).__name__
            if revoked_at is not None and not old_retirement_attempted:
                # Audit/IPC failure after attempted revocation still owes the
                # verified old group a bounded containment attempt.
                try:
                    contain(old, revoked=old_revoked)
                    receipt["old_group_contained"] = True
                except Exception as cleanup_exc:
                    receipt["cleanup_error"] = type(cleanup_exc).__name__
            if admission_started:
                try:
                    cleanup_candidate()
                except Exception as cleanup_exc:
                    receipt["cleanup_error"] = (
                        str(cleanup_exc) if isinstance(cleanup_exc, UpgradeError)
                        else type(cleanup_exc).__name__
                    )
            if endpoint is not None:
                try:
                    try:
                        actual = lifecycle.AgentProcess.load()
                    except FileNotFoundError:
                        actual = None
                    receipt["after"] = _summary(status(), actual)
                except Exception:
                    pass
            try:
                record("cancelled" if cancel.is_set() else "blocked")
            except Exception as audit_exc:
                receipt["audit_error"] = type(audit_exc).__name__
    return receipt


def upgrade_command(
    config: Path | None = typer.Option(None, "--config", help="Candidate runtime profile."),
    previous_config: Path | None = typer.Option(
        None, "--previous-config", help="Unchanged, known profile used by the current agent.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Offline validation only; no live IPC."),
    status: bool = typer.Option(False, "--status", help="Read the historical upgrade receipt."),
    expect_plan: str = typer.Option(
        "", "--expect-plan", help="plan_sha256 from an offline dry run.",
    ),
    reviewed_source_sha256: str = typer.Option(
        "", "--reviewed-source-sha256",
        help="Explicitly admit this exact reviewed runtime_source_sha256; never repin provenance.",
    ),
    drain_timeout_s: float = typer.Option(25.0, "--drain-timeout", min=0.1, max=25.0),
    startup_timeout_s: float = typer.Option(120.0, "--startup-timeout", min=0.1, max=600.0),
    launcher_pid: int | None = typer.Option(
        None, "--launcher-pid", min=1, help="Parent-held recovery shell PID.",
    ),
) -> None:
    """Validate then hand over only the agent. Cold startup, not zero downtime.

    Apply requires --expect-plan. Never release a parent-held launcher here.
    Wait budgets are per drain/start attempt; rollback can add another of each.
    Startup includes cold loading. IPC/filesystem overhead is additional; failed
    containment leaves control unavailable rather than promising a recovery time.
    --status is historical, not proof the recorded generation is still alive.
    """
    try:
        if status:
            if config or previous_config or dry_run or expect_plan or reviewed_source_sha256:
                raise UpgradeError("status-cannot-be-combined-with-upgrade")
            path = lifecycle.RUNTIME_DIR / "upgrade.json"
            typer.echo(path.read_text() if path.exists() else '{"phase": "no-upgrade-receipt"}')
            return
        if config is None or previous_config is None:
            raise UpgradeError("config-and-previous-config-required")
        plan = prepare_upgrade(config, previous_config)
        if dry_run:
            needs_review = plan.receipt["source_review_required"] and not reviewed_source_sha256
            if (reviewed_source_sha256
                    and reviewed_source_sha256 != plan.receipt["runtime_source_sha256"]):
                raise UpgradeError("exact-reviewed-runtime-source-required")
            typer.echo(json.dumps({**plan.receipt, "phase": (
                "source-review-required" if needs_review else "validated-offline"
            )}, sort_keys=True))
            if needs_review:
                raise typer.Exit(2)
            return
        cancel = threading.Event()
        originals = {}
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                originals[sig] = signal.signal(sig, lambda *_args: cancel.set())
            receipt = activate_upgrade(
                plan, expected_plan=expect_plan, drain_timeout_s=drain_timeout_s,
                startup_timeout_s=startup_timeout_s, launcher_pid=launcher_pid, cancel=cancel,
                reviewed_source_sha256=reviewed_source_sha256,
            )
        finally:
            for sig, handler in originals.items():
                signal.signal(sig, handler)
        typer.echo(json.dumps(receipt, sort_keys=True))
        if receipt["phase"] != "upgraded":
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        code = str(exc) if isinstance(exc, UpgradeError) else type(exc).__name__
        typer.echo(json.dumps({"phase": "refused", "error": code}))
        raise typer.Exit(2) from None
