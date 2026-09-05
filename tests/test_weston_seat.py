from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import minecraft_ai.platforms.weston_seat as seat
from minecraft_ai.platforms.bedrock_x11 import IsolationError


_posix_native_build = pytest.mark.skipif(
    os.name != "posix", reason="native artifact publication requires POSIX flock and permissions"
)


def _artifact(tmp_path: Path) -> seat.HeadlessSeatArtifact:
    module = tmp_path / "seat.so"
    module.write_bytes(b"test-module")
    return seat.HeadlessSeatArtifact(
        str(module),
        seat._sha256(module),
        seat._sha256(seat.SEAT_SOURCE),
    )


def test_module_provenance_rejects_binary_source_and_version_drift(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    seat.require_headless_seat_artifact(artifact)
    for changed in (
        replace(artifact, module_sha256="0" * 64),
        replace(artifact, source_sha256="0" * 64),
        replace(artifact, weston_version="14.0.0"),
        replace(artifact, module_path="relative.so"),
    ):
        with pytest.raises(IsolationError, match="provenance changed"):
            seat.require_headless_seat_artifact(changed)


def test_windows_native_calls_fail_closed_before_creating_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    monkeypatch.setattr(seat, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(IsolationError, match="requires Linux procfs"):
        seat.require_loaded_headless_seat(123, artifact)
    output = tmp_path / "must-not-be-created"
    with pytest.raises(IsolationError, match="requires POSIX file locking"):
        seat.build_headless_seat_module(root=output)
    assert not output.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux procfs device/inode mapping contract")
@pytest.mark.parametrize("mapping", ["valid", "inode-changed", "not-executable", "deleted"])
def test_loaded_module_requires_executable_mapping_of_exact_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mapping: str,
) -> None:
    artifact = _artifact(tmp_path)
    stat = Path(artifact.module_path).stat()
    inode = stat.st_ino + (1 if mapping == "inode-changed" else 0)
    permissions = "r--p" if mapping == "not-executable" else "r-xp"
    suffix = " (deleted)" if mapping == "deleted" else ""
    maps = (
        f"1000-2000 {permissions} 00000000 "
        f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x} "
        f"{inode} {artifact.module_path}{suffix}\n"
    )
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            maps if path == Path("/proc/123/maps") else original(path, *args, **kwargs)
        ),
    )
    if mapping == "valid":
        seat.require_loaded_headless_seat(123, artifact)
    else:
        with pytest.raises(IsolationError, match="not loaded"):
            seat.require_loaded_headless_seat(123, artifact)


@_posix_native_build
def test_build_uses_explicit_headers_and_reuses_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(seat.shutil, "which", lambda _name: "/usr/bin/cc")

    def compile_module(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"compiled-test-module")

    monkeypatch.setattr(seat.subprocess, "run", compile_module)
    artifact = seat.build_headless_seat_module(root=tmp_path, include_root=tmp_path / "headers")
    assert seat.build_headless_seat_module(root=tmp_path) == artifact
    assert len(commands) == 1
    assert f"-I{tmp_path / 'headers/usr/include/libweston-13'}" in commands[0]
    assert "-Wl,-z,defs" in commands[0]
    assert Path(artifact.module_path).parent.stat().st_mode & 0o777 == 0o700
    assert Path(artifact.module_path).stat().st_mode & 0o777 == 0o500
    manifest = Path(artifact.module_path).with_name("manifest.json")
    assert json.loads(manifest.read_text()) == {
        "module_path": artifact.module_path,
        "module_sha256": artifact.module_sha256,
        "source_sha256": artifact.source_sha256,
        "weston_version": "13.0.0",
    }


@_posix_native_build
def test_bad_installed_abi_is_rejected_before_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(seat.shutil, "which", lambda _name: "/usr/bin/cc")
    monkeypatch.setattr(seat.subprocess, "check_output", lambda *_args, **_kwargs: "14.0.0\n")
    monkeypatch.setattr(seat.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("compiled"))
    with pytest.raises(IsolationError, match="requires Weston 13.0.0"):
        seat.build_headless_seat_module(root=tmp_path)


@_posix_native_build
def test_compiler_failure_does_not_publish_module_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(seat.shutil, "which", lambda _name: "/usr/bin/cc")

    def failed_build(command: list[str], **_kwargs: object) -> None:
        Path(command[command.index("-o") + 1]).write_bytes(b"incomplete")
        raise subprocess.CalledProcessError(1, command, stderr="bad ABI")

    monkeypatch.setattr(seat.subprocess, "run", failed_build)
    with pytest.raises(IsolationError, match="bad ABI"):
        seat.build_headless_seat_module(root=tmp_path, include_root=tmp_path)
    assert not list(tmp_path.rglob("*.so"))
    assert not list(tmp_path.rglob("manifest.json"))


@_posix_native_build
def test_concurrent_builders_share_one_binary_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    compiled: list[str] = []
    monkeypatch.setattr(seat.shutil, "which", lambda _name: "/usr/bin/cc")

    def compile_module(command: list[str], **_kwargs: object) -> None:
        compiled.append(command[0])
        entered.set()
        assert release.wait(timeout=2)
        Path(command[command.index("-o") + 1]).write_bytes(b"same-binary")

    monkeypatch.setattr(seat.subprocess, "run", compile_module)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            seat.build_headless_seat_module, root=tmp_path, include_root=tmp_path
        )
        assert entered.wait(timeout=2)
        second = executor.submit(
            seat.build_headless_seat_module, root=tmp_path, include_root=tmp_path
        )
        release.set()
        artifact = first.result(timeout=2)
        inode = Path(artifact.module_path).stat().st_ino
        assert second.result(timeout=2) == artifact
    assert compiled == ["/usr/bin/cc"]
    assert Path(artifact.module_path).stat().st_ino == inode


@_posix_native_build
def test_incomplete_or_tampered_artifact_is_never_recompiled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = seat._artifact_directory(tmp_path)
    directory.mkdir(parents=True)
    output = directory / "headless-seat.so"
    output.write_bytes(b"unverified-binary")
    inode = output.stat().st_ino
    monkeypatch.setattr(seat.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("compiled"))
    with pytest.raises(IsolationError, match="refusing binary replacement"):
        seat.build_headless_seat_module(root=tmp_path)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "module_path": str(output),
                "module_sha256": "0" * 64,
                "source_sha256": seat._sha256(seat.SEAT_SOURCE),
            }
        )
    )
    with pytest.raises(IsolationError, match="provenance changed"):
        seat.build_headless_seat_module(root=tmp_path)
    assert output.stat().st_ino == inode
