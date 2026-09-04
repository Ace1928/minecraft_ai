from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlsplit

from platformdirs import user_data_dir, user_runtime_dir

from .config import RuntimeConfig

if os.name == "nt":
    import msvcrt
else:
    import fcntl


_SERVICE_ID = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_MANIFEST_SCHEMA = 1
_IS_WINDOWS = os.name == "nt"
_IS_LINUX = sys.platform.startswith("linux")


def _required_int_attribute(module: object, name: str) -> int:
    value = getattr(module, name, None)
    if not isinstance(value, int):
        raise OSError(f"platform lock constant {name} is unavailable")
    return value


def _windows_lock_descriptor(fd: int, *, unlock: bool) -> None:
    locking = getattr(msvcrt, "locking", None)
    if not callable(locking):
        raise OSError("Windows file locking is unavailable")
    mode_name = "LK_UNLCK" if unlock else "LK_NBLCK"
    locking(fd, _required_int_attribute(msvcrt, mode_name), 1)


def _posix_lock_descriptor(fd: int, *, unlock: bool) -> None:
    flock = getattr(fcntl, "flock", None)
    if not callable(flock):
        raise OSError("POSIX file locking is unavailable")
    if unlock:
        operation = _required_int_attribute(fcntl, "LOCK_UN")
    else:
        operation = _required_int_attribute(fcntl, "LOCK_EX") | _required_int_attribute(
            fcntl, "LOCK_NB"
        )
    flock(fd, operation)


def _set_private_descriptor_mode(fd: int) -> None:
    fchmod = getattr(os, "fchmod", None)
    if callable(fchmod):
        fchmod(fd, 0o600)


class ServiceMode(StrEnum):
    """How a stack service establishes its durable runtime state."""

    DAEMON = "daemon"
    ONESHOT = "oneshot"
    EXTERNAL = "external"


class StackPhase(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True)
class ProcessProbe:
    """Require the exact daemon PID recorded by this stack to remain alive."""


@dataclass(frozen=True)
class FileProbe:
    path: Path


@dataclass(frozen=True)
class TcpProbe:
    host: str
    port: int
    timeout_s: float = 0.5

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("TCP health-probe port must be in [1, 65535]")
        if not 0.05 <= self.timeout_s <= 10.0:
            raise ValueError("TCP health-probe timeout must be in [0.05, 10] seconds")


@dataclass(frozen=True)
class HttpProbe:
    url: str
    expected_status: int = 200
    timeout_s: float = 1.0
    json_field: str | None = None
    json_value: object | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("HTTP health probe requires an absolute http(s) URL")
        if not 100 <= self.expected_status <= 599:
            raise ValueError("HTTP health-probe status must be in [100, 599]")
        if not 0.05 <= self.timeout_s <= 10.0:
            raise ValueError("HTTP health-probe timeout must be in [0.05, 10] seconds")


@dataclass(frozen=True)
class CommandProbe:
    command: tuple[str, ...]
    timeout_s: float = 3.0

    def __post_init__(self) -> None:
        _validate_command(self.command, label="health-probe")
        if not 0.05 <= self.timeout_s <= 30.0:
            raise ValueError("command health-probe timeout must be in [0.05, 30] seconds")


HealthProbe = ProcessProbe | FileProbe | TcpProbe | HttpProbe | CommandProbe


@dataclass(frozen=True)
class ServiceSpec:
    """One explicitly bounded member of a portable local runtime stack.

    Commands are argv tuples and are never evaluated through a shell. A daemon
    remains owned by the stack process group. An oneshot command may establish
    an out-of-process resource (for example a managed Bedrock compositor), so
    it must provide an inverse command for transaction rollback.
    """

    service_id: str
    mode: ServiceMode
    probes: tuple[HealthProbe, ...]
    command: tuple[str, ...] = ()
    stop_command: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    cwd: Path | None = None
    log_path: Path | None = None
    ready_timeout_s: float = 30.0
    stop_timeout_s: float = 10.0
    reuse_if_healthy: bool = True

    def __post_init__(self) -> None:
        if _SERVICE_ID.fullmatch(self.service_id) is None:
            raise ValueError(f"invalid service id: {self.service_id!r}")
        if not self.probes:
            raise ValueError(f"service {self.service_id!r} must have an explicit health probe")
        if self.mode == ServiceMode.EXTERNAL:
            if self.command or self.stop_command:
                raise ValueError("external services cannot declare start/stop commands")
            if any(isinstance(probe, ProcessProbe) for probe in self.probes):
                raise ValueError("external services cannot use an owned-process health probe")
        else:
            _validate_command(self.command, label=f"service {self.service_id!r}")
        if self.mode == ServiceMode.ONESHOT:
            _validate_command(self.stop_command, label=f"oneshot {self.service_id!r} stop")
        elif self.stop_command:
            _validate_command(self.stop_command, label=f"service {self.service_id!r} stop")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"service {self.service_id!r} has duplicate dependencies")
        for key, value in self.environment:
            if not key or "=" in key or "\x00" in key or "\x00" in value:
                raise ValueError(f"invalid environment override for service {self.service_id!r}")
        if len({key for key, _value in self.environment}) != len(self.environment):
            raise ValueError(f"service {self.service_id!r} has duplicate environment keys")
        if not 0.1 <= self.ready_timeout_s <= 1800.0:
            raise ValueError("service ready timeout must be in [0.1, 1800] seconds")
        if not 0.1 <= self.stop_timeout_s <= 120.0:
            raise ValueError("service stop timeout must be in [0.1, 120] seconds")


@dataclass(frozen=True)
class StackPlan:
    profile_id: str
    services: tuple[ServiceSpec, ...]
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if _SERVICE_ID.fullmatch(self.profile_id) is None:
            raise ValueError(f"invalid stack profile id: {self.profile_id!r}")
        if not self.services:
            raise ValueError("stack plan must contain at least one service")
        by_id = {service.service_id: service for service in self.services}
        if len(by_id) != len(self.services):
            raise ValueError("stack service ids must be unique")
        for service in self.services:
            missing = set(service.dependencies) - set(by_id)
            if missing:
                raise ValueError(
                    f"service {service.service_id!r} has unknown dependencies: {sorted(missing)}"
                )
            if service.service_id in service.dependencies:
                raise ValueError(f"service {service.service_id!r} depends on itself")
        self.ordered_services()
        if len({key for key, _value in self.environment}) != len(self.environment):
            raise ValueError("stack plan has duplicate environment keys")

    def ordered_services(self) -> tuple[ServiceSpec, ...]:
        """Return a stable topological order and reject dependency cycles."""

        remaining = {service.service_id: service for service in self.services}
        emitted: list[ServiceSpec] = []
        emitted_ids: set[str] = set()
        while remaining:
            ready = [
                service
                for service in self.services
                if service.service_id in remaining
                and set(service.dependencies).issubset(emitted_ids)
            ]
            if not ready:
                raise ValueError(f"stack plan contains a dependency cycle: {sorted(remaining)}")
            for service in ready:
                emitted.append(service)
                emitted_ids.add(service.service_id)
                remaining.pop(service.service_id)
        return tuple(emitted)

    @property
    def digest(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "environment": list(self.environment),
            "services": [_service_digest_payload(service) for service in self.ordered_services()],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ServiceRecord:
    service_id: str
    mode: ServiceMode
    owned: bool
    pid: int | None
    started_wall_ns: int
    command_digest: str | None
    # Linux field 22 from /proc/<pid>/stat.  This is stable for the lifetime of
    # a process and prevents a persisted manifest from confusing a reused PID
    # with the daemon it originally launched.
    proc_start_ticks: int | None = None


@dataclass(frozen=True)
class StackManifest:
    schema_version: int
    transaction_id: str
    profile_id: str
    plan_digest: str
    phase: StackPhase
    created_wall_ns: int
    updated_wall_ns: int
    services: tuple[ServiceRecord, ...]
    error: str | None = None

    @classmethod
    def load(cls, path: Path) -> StackManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or int(raw.get("schema_version", -1)) != _MANIFEST_SCHEMA:
            raise ValueError("unsupported stack manifest")
        raw_services = raw.get("services")
        if not isinstance(raw_services, list):
            raise TypeError("stack manifest services must be a list")
        services = tuple(
            ServiceRecord(
                service_id=str(item["service_id"]),
                mode=ServiceMode(str(item["mode"])),
                owned=bool(item["owned"]),
                pid=None if item.get("pid") is None else int(item["pid"]),
                started_wall_ns=int(item["started_wall_ns"]),
                command_digest=(
                    None if item.get("command_digest") is None else str(item["command_digest"])
                ),
                proc_start_ticks=(
                    None
                    if item.get("proc_start_ticks") is None
                    else int(item["proc_start_ticks"])
                ),
            )
            for item in raw_services
            if isinstance(item, dict)
        )
        if len(services) != len(raw_services):
            raise TypeError("stack manifest contains a malformed service record")
        return cls(
            schema_version=_MANIFEST_SCHEMA,
            transaction_id=str(raw["transaction_id"]),
            profile_id=str(raw["profile_id"]),
            plan_digest=str(raw["plan_digest"]),
            phase=StackPhase(str(raw["phase"])),
            created_wall_ns=int(raw["created_wall_ns"]),
            updated_wall_ns=int(raw["updated_wall_ns"]),
            services=services,
            error=None if raw.get("error") is None else str(raw["error"]),
        )

    def persist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = asdict(self)
        payload["phase"] = self.phase.value
        for item in payload["services"]:
            item["mode"] = item["mode"].value
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        temporary.replace(path)


class StackStartError(RuntimeError):
    def __init__(self, message: str, *, rollback_errors: tuple[str, ...] = ()) -> None:
        self.rollback_errors = rollback_errors
        suffix = "" if not rollback_errors else f"; rollback errors: {'; '.join(rollback_errors)}"
        super().__init__(message + suffix)


@dataclass(frozen=True)
class ManagedModelServer:
    """Explicit local model-server command selected by a hardware profile."""

    command: tuple[str, ...]
    health_url: str | None = None
    ready_timeout_s: float = 300.0
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _validate_command(self.command, label="managed model server")


class PortableStackLauncher:
    """Transactional, persistent process orchestration without systemd coupling."""

    def __init__(
        self,
        plan: StackPlan,
        *,
        runtime_dir: Path | None = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        self.plan = plan
        self.runtime_dir = (
            Path(user_runtime_dir("minecraft-ai")) / "stack" if runtime_dir is None else runtime_dir
        )
        self.manifest_path = self.runtime_dir / "manifest.json"
        self.lock_path = self.runtime_dir / "lifecycle.lock"
        self.poll_interval_s = poll_interval_s
        self._children: dict[int, subprocess.Popen[bytes]] = {}

    def start(self) -> StackManifest:
        with self._lock():
            previous = self._load_manifest()
            if previous is not None:
                self._require_matching_plan(previous)
                if previous.phase == StackPhase.RUNNING and self._manifest_healthy(previous):
                    return previous
                previous = replace(
                    previous,
                    phase=StackPhase.ROLLING_BACK,
                    updated_wall_ns=time.time_ns(),
                )
                previous.persist(self.manifest_path)
                prior_rollback_errors = self._rollback(previous.services)
                if prior_rollback_errors:
                    previous = replace(
                        previous,
                        phase=StackPhase.FAILED,
                        error="; ".join(prior_rollback_errors),
                        updated_wall_ns=time.time_ns(),
                    )
                    previous.persist(self.manifest_path)
                    raise StackStartError(
                        "refusing to start over an incompletely rolled-back stack",
                        rollback_errors=prior_rollback_errors,
                    )
                previous = replace(
                    previous,
                    phase=StackPhase.STOPPED,
                    services=_release_records(previous.services),
                    error=None,
                    updated_wall_ns=time.time_ns(),
                )
                previous.persist(self.manifest_path)

            now = time.time_ns()
            manifest = StackManifest(
                schema_version=_MANIFEST_SCHEMA,
                transaction_id=uuid.uuid4().hex,
                profile_id=self.plan.profile_id,
                plan_digest=self.plan.digest,
                phase=StackPhase.STARTING,
                created_wall_ns=now,
                updated_wall_ns=now,
                services=(),
            )
            manifest.persist(self.manifest_path)
            records: list[ServiceRecord] = []

            def record_started(record: ServiceRecord) -> None:
                nonlocal manifest
                records.append(record)
                manifest = replace(
                    manifest,
                    services=tuple(records),
                    updated_wall_ns=time.time_ns(),
                )
                manifest.persist(self.manifest_path)

            try:
                for service in self.plan.ordered_services():
                    self._start_service(service, on_started=record_started)
            except Exception as exc:
                manifest = replace(
                    manifest,
                    phase=StackPhase.ROLLING_BACK,
                    services=tuple(records),
                    error=f"{type(exc).__name__}: {exc}",
                    updated_wall_ns=time.time_ns(),
                )
                persistence_errors: list[str] = []
                try:
                    manifest.persist(self.manifest_path)
                except Exception as persist_exc:
                    persistence_errors.append(
                        "could not persist rolling-back state: "
                        f"{type(persist_exc).__name__}: {persist_exc}"
                    )
                rollback_errors = self._rollback(tuple(records))
                manifest = replace(
                    manifest,
                    phase=StackPhase.FAILED,
                    services=(
                        tuple(records) if rollback_errors else _release_records(tuple(records))
                    ),
                    updated_wall_ns=time.time_ns(),
                )
                try:
                    manifest.persist(self.manifest_path)
                except Exception as persist_exc:
                    persistence_errors.append(
                        "could not persist failed state: "
                        f"{type(persist_exc).__name__}: {persist_exc}"
                    )
                reported_errors = [*rollback_errors, *persistence_errors]
                raise StackStartError(
                    f"stack start failed: {type(exc).__name__}: {exc}",
                    rollback_errors=tuple(reported_errors),
                ) from exc

            manifest = replace(
                manifest,
                phase=StackPhase.RUNNING,
                services=tuple(records),
                updated_wall_ns=time.time_ns(),
            )
            manifest.persist(self.manifest_path)
            return manifest

    def stop(self) -> StackManifest | None:
        with self._lock():
            manifest = self._load_manifest()
            if manifest is None:
                return None
            self._require_matching_plan(manifest)
            manifest = replace(
                manifest,
                phase=StackPhase.STOPPING,
                updated_wall_ns=time.time_ns(),
            )
            manifest.persist(self.manifest_path)
            rollback_errors = self._rollback(manifest.services)
            manifest = replace(
                manifest,
                phase=StackPhase.STOPPED if not rollback_errors else StackPhase.FAILED,
                services=(
                    manifest.services if rollback_errors else _release_records(manifest.services)
                ),
                error=None if not rollback_errors else "; ".join(rollback_errors),
                updated_wall_ns=time.time_ns(),
            )
            manifest.persist(self.manifest_path)
            if rollback_errors:
                raise StackStartError("stack stop was incomplete", rollback_errors=rollback_errors)
            return manifest

    def status(self) -> tuple[StackManifest | None, dict[str, bool]]:
        manifest = self._load_manifest()
        if manifest is None:
            return None, {}
        if manifest.plan_digest != self.plan.digest:
            return manifest, {"plan_matches": False}
        by_id = {service.service_id: service for service in self.plan.services}
        health = {
            record.service_id: self._record_healthy(by_id[record.service_id], record)
            for record in manifest.services
            if record.service_id in by_id
        }
        return manifest, health

    def _start_service(
        self,
        service: ServiceSpec,
        *,
        on_started: Callable[[ServiceRecord], None] | None = None,
    ) -> ServiceRecord:
        def announce(record: ServiceRecord) -> ServiceRecord:
            if on_started is not None:
                on_started(record)
            return record

        if service.reuse_if_healthy and self._healthy(service, pid=None, reuse_check=True):
            return announce(
                ServiceRecord(
                    service_id=service.service_id,
                    mode=service.mode,
                    owned=False,
                    pid=None,
                    started_wall_ns=time.time_ns(),
                    command_digest=None,
                    proc_start_ticks=None,
                )
            )
        if service.mode == ServiceMode.EXTERNAL:
            deadline = time.monotonic() + service.ready_timeout_s
            while time.monotonic() < deadline:
                if self._healthy(service, pid=None):
                    return announce(
                        ServiceRecord(
                            service_id=service.service_id,
                            mode=service.mode,
                            owned=False,
                            pid=None,
                            started_wall_ns=time.time_ns(),
                            command_digest=None,
                            proc_start_ticks=None,
                        )
                    )
                time.sleep(self.poll_interval_s)
            raise TimeoutError(
                f"required external service {service.service_id!r} did not become healthy "
                f"within {service.ready_timeout_s:.1f}s"
            )

        pid: int | None = None
        command_digest = _command_digest(service.command)
        proc_start_ticks: int | None = None
        record_command_digest = command_digest
        identity: tuple[int, tuple[str, ...]] | None = None
        if service.mode == ServiceMode.DAEMON:
            process = self._spawn(service, service.command)
            pid = process.pid
            self._children[pid] = process
            if _IS_LINUX:
                observed_start_ticks = _linux_process_start_ticks(pid)
                identity_deadline = time.monotonic() + min(0.5, service.ready_timeout_s)
                while True:
                    identity = _linux_process_identity(pid)
                    if identity is not None:
                        observed_start_ticks = identity[0]
                        record_command_digest = _command_digest(identity[1])
                        if record_command_digest == command_digest:
                            break
                    else:
                        observed_start_ticks = (
                            _linux_process_start_ticks(pid) or observed_start_ticks
                        )
                    if process.poll() is not None or time.monotonic() >= identity_deadline:
                        break
                    time.sleep(max(0.001, min(self.poll_interval_s, 0.01)))
                proc_start_ticks = observed_start_ticks
            record = announce(
                ServiceRecord(
                    service_id=service.service_id,
                    mode=service.mode,
                    owned=True,
                    pid=pid,
                    started_wall_ns=time.time_ns(),
                    command_digest=record_command_digest,
                    proc_start_ticks=proc_start_ticks,
                )
            )
        else:
            record = ServiceRecord(
                service_id=service.service_id,
                mode=service.mode,
                owned=True,
                pid=None,
                started_wall_ns=time.time_ns(),
                command_digest=command_digest,
                proc_start_ticks=None,
            )

            def announce_oneshot_start() -> None:
                # Spawning the start helper is the ownership boundary. Persist
                # it before waiting so rollback can run the inverse command if
                # the helper fails after creating an out-of-process resource.
                announce(record)

            self._run_oneshot(
                service,
                service.command,
                on_spawned=announce_oneshot_start,
            )
        if _IS_LINUX and service.mode == ServiceMode.DAEMON and (
            identity is None or record_command_digest != command_digest
        ):
            raise RuntimeError(
                f"could not establish process identity for service {service.service_id!r}"
            )

        deadline = time.monotonic() + service.ready_timeout_s
        while time.monotonic() < deadline:
            if self._healthy(service, pid=pid):
                return record
            if pid is not None:
                child = self._children.get(pid)
                if child is not None and child.poll() is not None:
                    raise RuntimeError(
                        f"service {service.service_id!r} exited with code {child.returncode}"
                    )
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            f"service {service.service_id!r} did not become healthy within "
            f"{service.ready_timeout_s:.1f}s; inspect {self._log_path(service)}"
        )

    def _manifest_healthy(self, manifest: StackManifest) -> bool:
        records = {record.service_id: record for record in manifest.services}
        if set(records) != {service.service_id for service in self.plan.services}:
            return False
        return all(
            self._record_healthy(service, records[service.service_id])
            for service in self.plan.services
        )

    def _record_healthy(self, service: ServiceSpec, record: ServiceRecord) -> bool:
        """Evaluate probes without accepting a reconstructed, reused PID."""

        if record.owned and record.pid is not None:
            if record.command_digest != _command_digest(service.command):
                return False
            child = self._children.get(record.pid)
            if child is not None:
                if child.poll() is not None:
                    return False
            elif not _persisted_pid_matches(
                record.pid,
                command_digest=record.command_digest,
                proc_start_ticks=record.proc_start_ticks,
            ):
                return False
        return self._healthy(service, pid=record.pid)

    def _healthy(
        self,
        service: ServiceSpec,
        *,
        pid: int | None,
        reuse_check: bool = False,
    ) -> bool:
        probes = tuple(
            probe
            for probe in service.probes
            if not (reuse_check and isinstance(probe, ProcessProbe))
        )
        if reuse_check and not probes:
            return False
        environment = self._environment(service)
        return all(_probe_healthy(probe, pid=pid, environment=environment) for probe in probes)

    def _rollback(self, records: tuple[ServiceRecord, ...]) -> tuple[str, ...]:
        by_id = {service.service_id: service for service in self.plan.services}
        errors: list[str] = []
        for record in reversed(records):
            if not record.owned:
                continue
            service = by_id.get(record.service_id)
            if service is None:
                errors.append(f"missing service spec for owned {record.service_id!r}")
                continue
            if service.stop_command:
                try:
                    self._run_oneshot(service, service.stop_command)
                except Exception as exc:
                    errors.append(f"{record.service_id} stop command: {type(exc).__name__}: {exc}")
            if record.pid is not None:
                try:
                    self._terminate_owned_pid(
                        record.pid,
                        service.stop_timeout_s,
                        record.command_digest,
                        record.proc_start_ticks,
                    )
                except Exception as exc:
                    errors.append(f"{record.service_id} process: {type(exc).__name__}: {exc}")
        return tuple(errors)

    def _spawn(self, service: ServiceSpec, command: tuple[str, ...]) -> subprocess.Popen[bytes]:
        log_path = self._log_path(service)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log:
            if _IS_WINDOWS:
                return subprocess.Popen(
                    command,
                    cwd=None if service.cwd is None else str(service.cwd),
                    env=self._environment(service),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
            return subprocess.Popen(
                command,
                cwd=None if service.cwd is None else str(service.cwd),
                env=self._environment(service),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )

    def _run_oneshot(
        self,
        service: ServiceSpec,
        command: tuple[str, ...],
        *,
        on_spawned: Callable[[], None] | None = None,
    ) -> None:
        process = self._spawn(service, command)
        self._children[process.pid] = process
        proc_start_ticks: int | None = None
        if _IS_LINUX:
            identity_deadline = time.monotonic() + min(0.5, service.ready_timeout_s)
            while proc_start_ticks is None and process.poll() is None:
                proc_start_ticks = _linux_process_start_ticks(process.pid)
                if proc_start_ticks is not None or time.monotonic() >= identity_deadline:
                    break
                time.sleep(max(0.001, min(self.poll_interval_s, 0.01)))
        try:
            if on_spawned is not None:
                try:
                    on_spawned()
                except Exception as exc:
                    try:
                        self._terminate_owned_pid(
                            process.pid,
                            service.stop_timeout_s,
                            _command_digest(command),
                            proc_start_ticks,
                        )
                    except Exception as cleanup_exc:
                        exc.add_note(
                            "failed to contain spawned oneshot command: "
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
                    raise
            try:
                return_code = process.wait(timeout=service.ready_timeout_s)
            except subprocess.TimeoutExpired as exc:
                self._terminate_owned_pid(
                    process.pid,
                    service.stop_timeout_s,
                    _command_digest(command),
                    proc_start_ticks,
                )
                raise TimeoutError(
                    f"command for {service.service_id!r} exceeded {service.ready_timeout_s:.1f}s"
                ) from exc
        finally:
            self._children.pop(process.pid, None)
        if return_code != 0:
            raise RuntimeError(
                f"command for {service.service_id!r} exited with code {return_code}; "
                f"inspect {self._log_path(service)}"
            )

    def _terminate_owned_pid(
        self,
        pid: int,
        timeout_s: float,
        command_digest: str | None,
        proc_start_ticks: int | None = None,
    ) -> None:
        child = self._children.get(pid)
        if _IS_WINDOWS:
            if child is not None and child.poll() is not None:
                self._children.pop(pid, None)
                return
            if not _pid_alive(pid):
                self._children.pop(pid, None)
                return
            if child is None:
                raise RuntimeError(
                    f"refusing to signal reconstructed Windows PID {pid}: "
                    "process identity is unverifiable"
                )
            # taskkill /T is required even when the direct child has a retained
            # Popen handle: Popen.terminate() only stops that process and can
            # orphan the service tree.  Judge success from the retained handle
            # (or a PID probe after manifest reconstruction), not taskkill's
            # runner-dependent return code.
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=timeout_s,
            )
            if child is not None:
                try:
                    child.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired as tree_timeout:
                    # Best-effort direct-child cleanup does not prove that the
                    # tree was stopped, so still surface the failed tree kill.
                    child.kill()
                    try:
                        child.wait(timeout=1.0)
                    except subprocess.TimeoutExpired as exc:
                        raise RuntimeError(f"process {pid} did not terminate") from exc
                    detail = completed.stderr.decode(errors="replace").strip()
                    suffix = "" if not detail else f": {detail}"
                    raise RuntimeError(
                        f"taskkill did not terminate process tree {pid} "
                        f"(code {completed.returncode}){suffix}"
                    ) from tree_timeout
                finally:
                    self._children.pop(pid, None)
                return
            if not _pid_alive(pid):
                return
            detail = completed.stderr.decode(errors="replace").strip()
            suffix = "" if not detail else f": {detail}"
            raise RuntimeError(f"taskkill failed with code {completed.returncode}{suffix}")

        kill_group = getattr(os, "killpg", None)
        if not callable(kill_group):
            raise RuntimeError("process-group termination is unavailable on this platform")
        group_alive = _process_group_alive(pid)
        child_running = child is not None and child.poll() is None
        if not group_alive:
            if child is not None:
                child.poll()
            self._children.pop(pid, None)
            return
        if child is None:
            if not _persisted_process_or_orphan_group_matches(
                pid,
                command_digest=command_digest,
                proc_start_ticks=proc_start_ticks,
            ):
                raise RuntimeError(
                    f"refusing to signal process group {pid}: persisted process identity "
                    "changed or is unverifiable"
                )
        elif not child_running:
            # Once the retained leader exits, its handle no longer proves that
            # the numeric process group still belongs to this launcher. Treat
            # the surviving group like reconstructed state and revalidate the
            # persisted Linux identity before signaling it.
            self._children.pop(pid, None)
            child = None

        # Close the PID-reuse race between the initial check and killpg for a
        # launcher reconstructed from disk.  A live child handle does not need
        # procfs identity because it is direct ownership evidence.
        if child is None and not _persisted_process_or_orphan_group_matches(
            pid,
            command_digest=command_digest,
            proc_start_ticks=proc_start_ticks,
        ):
            raise RuntimeError(
                f"refusing to signal process group {pid}: persisted process identity "
                "changed or is unverifiable"
            )
        try:
            kill_group(pid, signal.SIGTERM)
        except ProcessLookupError:
            if child is not None:
                child.poll()
            self._children.pop(pid, None)
            return

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if child is not None:
                child.poll()
            if not _process_group_alive(pid):
                if child is not None:
                    child.poll()
                self._children.pop(pid, None)
                return
            time.sleep(self.poll_interval_s)

        identity_required = child is None or child.poll() is not None
        if identity_required and not _persisted_process_or_orphan_group_matches(
            pid,
            command_digest=command_digest,
            proc_start_ticks=proc_start_ticks,
        ):
            raise RuntimeError(
                f"refusing to escalate process group {pid}: persisted process identity "
                "changed or is unverifiable"
            )
        try:
            kill_group(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            if child is not None:
                child.poll()
            self._children.pop(pid, None)
            return
        kill_deadline = time.monotonic() + 1.0
        while time.monotonic() < kill_deadline:
            if child is not None:
                child.poll()
            if not _process_group_alive(pid):
                if child is not None:
                    child.poll()
                self._children.pop(pid, None)
                return
            time.sleep(self.poll_interval_s)
        if child is not None:
            child.poll()
        self._children.pop(pid, None)
        raise RuntimeError(f"process group {pid} did not terminate")

    def _environment(self, service: ServiceSpec) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(dict(self.plan.environment))
        environment.update(dict(service.environment))
        environment["MINECRAFT_AI_STACK_PROFILE"] = self.plan.profile_id
        return environment

    def _log_path(self, service: ServiceSpec) -> Path:
        if service.log_path is not None:
            return service.log_path
        return Path(user_data_dir("minecraft-ai")) / "logs" / "stack" / f"{service.service_id}.log"

    def _load_manifest(self) -> StackManifest | None:
        try:
            return StackManifest.load(self.manifest_path)
        except FileNotFoundError:
            return None

    def _require_matching_plan(self, manifest: StackManifest) -> None:
        if manifest.plan_digest != self.plan.digest:
            raise RuntimeError(
                "persisted stack belongs to a different launch plan; stop it with its original "
                "profile before replacing it"
            )

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        handle = self.lock_path.open("a+b")
        acquired = False
        try:
            try:
                if os.name == "nt":
                    if self.lock_path.stat().st_size == 0:
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    _windows_lock_descriptor(handle.fileno(), unlock=False)
                else:
                    _posix_lock_descriptor(handle.fileno(), unlock=False)
                acquired = True
            except (BlockingIOError, OSError) as exc:
                owner = _read_lock_owner(self.lock_path)
                suffix = "" if owner is None else f" owned by PID {owner}"
                raise RuntimeError(f"another stack lifecycle operation is active{suffix}") from exc
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "token": token}).encode())
            handle.flush()
            try:
                _set_private_descriptor_mode(handle.fileno())
            except OSError:
                pass
            yield
        finally:
            if acquired:
                try:
                    if os.name == "nt":
                        handle.seek(0)
                        _windows_lock_descriptor(handle.fileno(), unlock=True)
                    else:
                        _posix_lock_descriptor(handle.fileno(), unlock=True)
                except OSError:
                    pass
            handle.close()


def build_bedrock_stack_plan(
    config: RuntimeConfig,
    *,
    role: str = "generalist",
    python_executable: str = sys.executable,
    width: int = 1920,
    height: int = 1080,
    dashboard_port: int = 8765,
    high_level_server: ManagedModelServer | None = None,
    vision_server: ManagedModelServer | None = None,
    log_dir: Path | None = None,
) -> StackPlan:
    """Build the full isolated Bedrock startup transaction.

    Model endpoints configured as enabled are mandatory health gates. A caller
    may provide a hardware-qualified managed launch command; otherwise the
    endpoint is treated as externally managed and must already be healthy.
    This prevents a one-click path from silently substituting an unreviewed
    checkpoint, license, quantization, or accelerator backend.
    """

    if not 320 <= width <= 7680 or not 240 <= height <= 4320:
        raise ValueError("Bedrock dimensions are outside supported bounds")
    if not 1 <= dashboard_port <= 65535:
        raise ValueError("dashboard port must be in [1, 65535]")
    logs = Path(user_data_dir("minecraft-ai")) / "logs" / "stack" if log_dir is None else log_dir
    health_module = (python_executable, "-m", "minecraft_ai.stack_health")
    services: list[ServiceSpec] = [
        ServiceSpec(
            service_id="safety-gate",
            mode=ServiceMode.EXTERNAL,
            probes=(CommandProbe((*health_module, "start-permitted"), timeout_s=3.0),),
            ready_timeout_s=0.1,
            reuse_if_healthy=True,
        ),
        ServiceSpec(
            service_id="dashboard",
            mode=ServiceMode.DAEMON,
            command=(
                python_executable,
                "-m",
                "minecraft_ai.cli",
                "dashboard",
                "--host",
                "127.0.0.1",
                "--port",
                str(dashboard_port),
            ),
            probes=(
                ProcessProbe(),
                HttpProbe(
                    f"http://127.0.0.1:{dashboard_port}/healthz",
                    json_field="status",
                    json_value="ok",
                ),
            ),
            dependencies=("safety-gate",),
            log_path=logs / "dashboard.log",
        ),
        ServiceSpec(
            service_id="bedrock",
            mode=ServiceMode.ONESHOT,
            command=(
                python_executable,
                "-m",
                "minecraft_ai.cli",
                "bedrock",
                "launch",
                "--width",
                str(width),
                "--height",
                str(height),
                "--fullscreen",
            ),
            stop_command=(
                python_executable,
                "-m",
                "minecraft_ai.cli",
                "bedrock",
                "stop",
                "--transient",
            ),
            probes=(CommandProbe((*health_module, "bedrock"), timeout_s=5.0),),
            dependencies=("dashboard",),
            log_path=logs / "bedrock.log",
            ready_timeout_s=120.0,
            stop_timeout_s=20.0,
        ),
    ]

    model_dependencies: list[str] = []
    if config.high_level.enabled:
        services.append(
            _model_service(
                "high-level-model",
                config.high_level.base_url,
                high_level_server,
                dependencies=("bedrock",),
                log_path=logs / "high-level-model.log",
            )
        )
        model_dependencies.append("high-level-model")
    if config.vision_language.enabled:
        services.append(
            _model_service(
                "vision-model",
                config.vision_language.base_url,
                vision_server,
                dependencies=("bedrock",),
                log_path=logs / "vision-model.log",
            )
        )
        model_dependencies.append("vision-model")

    services.extend(
        (
            ServiceSpec(
                service_id="supervisor",
                mode=ServiceMode.DAEMON,
                command=(
                    python_executable,
                    "-m",
                    "minecraft_ai.supervisor",
                    "--role",
                    role,
                ),
                probes=(
                    ProcessProbe(),
                    CommandProbe((*health_module, "supervisor"), timeout_s=3.0),
                ),
                dependencies=("bedrock", *model_dependencies),
                log_path=logs / "supervisor.log",
            ),
            ServiceSpec(
                service_id="world-ready",
                mode=ServiceMode.EXTERNAL,
                probes=(CommandProbe((*health_module, "playable-bedrock"), timeout_s=5.0),),
                dependencies=("bedrock", *model_dependencies, "supervisor"),
                ready_timeout_s=900.0,
            ),
            ServiceSpec(
                service_id="live-agent",
                mode=ServiceMode.ONESHOT,
                command=(
                    python_executable,
                    "-m",
                    "minecraft_ai.cli",
                    "run",
                    "--role",
                    role,
                    "--live",
                ),
                stop_command=(*health_module, "stop-agent"),
                probes=(CommandProbe((*health_module, "live-agent"), timeout_s=5.0),),
                dependencies=(
                    "dashboard",
                    "bedrock",
                    *model_dependencies,
                    "supervisor",
                    "world-ready",
                ),
                log_path=logs / "live-agent.log",
                ready_timeout_s=max(180.0, config.policy.startup_timeout_s + 60.0),
            ),
        )
    )
    return StackPlan(profile_id="bedrock-live", services=tuple(services))


def _model_service(
    service_id: str,
    base_url: str,
    managed: ManagedModelServer | None,
    *,
    dependencies: tuple[str, ...],
    log_path: Path,
) -> ServiceSpec:
    health_url = _model_health_url(base_url) if managed is None else managed.health_url
    if health_url is None:
        health_url = _model_health_url(base_url)
    probe = HttpProbe(health_url)
    if managed is None:
        return ServiceSpec(
            service_id=service_id,
            mode=ServiceMode.EXTERNAL,
            probes=(probe,),
            dependencies=dependencies,
            log_path=log_path,
            ready_timeout_s=5.0,
        )
    return ServiceSpec(
        service_id=service_id,
        mode=ServiceMode.DAEMON,
        command=managed.command,
        probes=(ProcessProbe(), probe),
        dependencies=dependencies,
        environment=managed.environment,
        log_path=log_path,
        ready_timeout_s=managed.ready_timeout_s,
    )


def _model_health_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid local model base URL: {base_url!r}")
    netloc = parsed.netloc
    return f"{parsed.scheme}://{netloc}/health"


def _probe_healthy(
    probe: HealthProbe,
    *,
    pid: int | None,
    environment: Mapping[str, str],
) -> bool:
    if isinstance(probe, ProcessProbe):
        return pid is not None and _pid_alive(pid)
    if isinstance(probe, FileProbe):
        return probe.path.is_file()
    if isinstance(probe, TcpProbe):
        try:
            with socket.create_connection((probe.host, probe.port), timeout=probe.timeout_s):
                return True
        except OSError:
            return False
    if isinstance(probe, HttpProbe):
        try:
            with urllib.request.urlopen(probe.url, timeout=probe.timeout_s) as response:
                if response.status != probe.expected_status:
                    return False
                if probe.json_field is None:
                    return True
                raw = json.loads(response.read().decode("utf-8"))
                return isinstance(raw, dict) and raw.get(probe.json_field) == probe.json_value
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError):
            return False
    try:
        completed = subprocess.run(
            probe.command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(environment),
            timeout=probe.timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _service_digest_payload(service: ServiceSpec) -> dict[str, object]:
    probes: list[dict[str, object]] = []
    for probe in service.probes:
        payload = asdict(probe)
        payload["kind"] = type(probe).__name__
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        probes.append(payload)
    return {
        "service_id": service.service_id,
        "mode": service.mode.value,
        "command": list(service.command),
        "stop_command": list(service.stop_command),
        "dependencies": list(service.dependencies),
        "environment": list(service.environment),
        "cwd": None if service.cwd is None else str(service.cwd),
        "log_path": None if service.log_path is None else str(service.log_path),
        "ready_timeout_s": service.ready_timeout_s,
        "stop_timeout_s": service.stop_timeout_s,
        "reuse_if_healthy": service.reuse_if_healthy,
        "probes": probes,
    }


def _validate_command(command: tuple[str, ...], *, label: str) -> None:
    if not command or any(not value or "\x00" in value for value in command):
        raise ValueError(f"{label} requires a non-empty argv command")


def _command_digest(command: tuple[str, ...]) -> str:
    return hashlib.sha256(b"\x00".join(os.fsencode(argument) for argument in command)).hexdigest()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _IS_WINDOWS:
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_pid_alive(pid: int) -> bool:
    """Query a Windows process without using the destructive os.kill shim."""
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return False
    kernel32: Any = win_dll("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    open_process.restype = ctypes.c_void_p
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    wait_for_single_object.restype = ctypes.c_ulong
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int

    synchronize = 0x00100000
    wait_timeout = 0x00000102
    handle = open_process(synchronize, False, pid)
    if not handle:
        return False
    try:
        wait_result = int(wait_for_single_object(handle, 0))
        return wait_result == wait_timeout
    finally:
        close_handle(handle)


def _pid_matches(pid: int, expected_digest: str) -> bool:
    command_file = Path(f"/proc/{pid}/cmdline")
    try:
        command = tuple(
            item.decode(errors="surrogateescape")
            for item in command_file.read_bytes().split(b"\x00")
            if item
        )
    except OSError:
        return False
    return _command_digest(command) == expected_digest


def _linux_process_start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    close_paren = stat.rfind(")")
    if close_paren < 0:
        return None
    # Text after comm begins at proc(5) field 3 (state), making index 19
    # field 22: starttime in clock ticks since system boot.
    fields = stat[close_paren + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        start_ticks = int(fields[19])
    except ValueError:
        return None
    return start_ticks if start_ticks > 0 else None


def _linux_process_identity(pid: int) -> tuple[int, tuple[str, ...]] | None:
    """Return Linux start ticks plus exact argv for PID-reuse-safe ownership."""

    start_ticks = _linux_process_start_ticks(pid)
    try:
        command_raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    command = tuple(
        os.fsdecode(argument)
        for argument in command_raw.rstrip(b"\0").split(b"\0")
        if argument
    )
    if start_ticks is None or not command:
        return None
    return start_ticks, command


def _persisted_pid_matches(
    pid: int,
    *,
    command_digest: str | None,
    proc_start_ticks: int | None,
) -> bool:
    """Verify a process reconstructed from a persisted service record."""

    if _IS_WINDOWS:
        # Retained Popen handles are safe within one launcher process, but the
        # v1 manifest does not contain a Windows creation-time token. Never
        # reconstruct ownership from a numeric PID alone.
        return False
    if (
        not _IS_LINUX
        or command_digest is None
        or proc_start_ticks is None
        or proc_start_ticks <= 0
        or not _pid_alive(pid)
    ):
        return False
    identity = _linux_process_identity(pid)
    if identity is None:
        return False
    start_ticks, command = identity
    return start_ticks == proc_start_ticks and _command_digest(command) == command_digest


def _persisted_process_or_orphan_group_matches(
    pid: int,
    *,
    command_digest: str | None,
    proc_start_ticks: int | None,
) -> bool:
    if _persisted_pid_matches(
        pid,
        command_digest=command_digest,
        proc_start_ticks=proc_start_ticks,
    ):
        return True
    if (
        not _IS_LINUX
        or command_digest is None
        or len(command_digest) != 64
        or any(character not in "0123456789abcdef" for character in command_digest)
        or proc_start_ticks is None
        or proc_start_ticks <= 0
        or _pid_alive(pid)
    ):
        return False
    members = _linux_process_group_members(pid)
    if members is None or any(member_pid == pid for member_pid, _state in members):
        return False
    # Linux cannot reuse a numeric PGID while that group remains populated.
    # A missing leader PID plus visible live members therefore identifies the
    # orphaned group retained from this modern, identity-bearing record.  This
    # deliberately does not extend to legacy records or opaque POSIX systems.
    return any(state not in {"Z", "X"} for _member_pid, state in members)


def _linux_process_group_members(
    process_group_id: int,
) -> tuple[tuple[int, str], ...] | None:
    try:
        process_entries = tuple(Path("/proc").iterdir())
    except OSError:
        return None
    members: list[tuple[int, str]] = []
    for entry in process_entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        close_paren = stat.rfind(")")
        if close_paren < 0:
            continue
        fields = stat[close_paren + 1 :].split()
        if len(fields) <= 2:
            continue
        try:
            group_id = int(fields[2])
        except ValueError:
            continue
        if group_id == process_group_id:
            members.append((int(entry.name), fields[0]))
    return tuple(members)


def _process_group_alive(process_group_id: int) -> bool:
    """Probe a POSIX process group, including descendants after leader exit."""

    if process_group_id <= 0 or _IS_WINDOWS:
        return False
    try:
        os.kill(-process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    if not _IS_LINUX:
        return True

    # kill(-pgid, 0) also succeeds for groups containing only zombies.  Such a
    # group has no executable members left and cannot respond to another
    # signal, so do not turn harmless reaping latency into a failed shutdown.
    members = _linux_process_group_members(process_group_id)
    if members is None:
        return True
    if not members:
        # procfs may hide other users' processes; preserve kill(0)'s result.
        return True
    return any(state not in {"Z", "X"} for _pid, state in members)


def _read_lock_owner(path: Path) -> int | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        pid = int(raw["pid"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return pid if pid > 0 else None


def _release_records(records: tuple[ServiceRecord, ...]) -> tuple[ServiceRecord, ...]:
    """Mark resources as released so a later start cannot stop unrelated replacements."""

    return tuple(
        replace(
            record,
            owned=False,
            pid=None,
            command_digest=None,
            proc_start_ticks=None,
        )
        for record in records
    )
