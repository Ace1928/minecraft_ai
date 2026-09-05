"""Build and verify the pinned Weston module that supplies a virtual-only seat."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_data_dir

from .bedrock_x11 import IsolationError


WESTON_SEAT_VERSION = "13.0.0"
SEAT_SOURCE = Path(__file__).with_name("native") / "headless_seat.c"
SEAT_ARTIFACT_ROOT = Path(user_data_dir("minecraft-ai")) / "native"


@dataclass(frozen=True)
class HeadlessSeatArtifact:
    module_path: str
    module_sha256: str
    source_sha256: str
    weston_version: str = WESTON_SEAT_VERSION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_headless_seat_artifact(artifact: HeadlessSeatArtifact) -> None:
    """Reject stale source, an altered binary, or an unsupported Weston ABI."""
    path = Path(artifact.module_path)
    try:
        valid = (
            path.is_absolute()
            and artifact.weston_version == WESTON_SEAT_VERSION
            and artifact.source_sha256 == _sha256(SEAT_SOURCE)
            and artifact.module_sha256 == _sha256(path)
        )
    except OSError as exc:
        raise IsolationError("headless virtual-seat artifact is unavailable") from exc
    if not valid:
        raise IsolationError("headless virtual-seat source or binary provenance changed")


def require_loaded_headless_seat(pid: int, artifact: HeadlessSeatArtifact) -> None:
    """Bind the current binary to an executable mapping in the live compositor."""
    if sys.platform != "linux":
        raise IsolationError("live headless virtual-seat verification requires Linux procfs")
    require_headless_seat_artifact(artifact)
    module = Path(artifact.module_path)
    try:
        stat = module.stat()
        lines = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise IsolationError("loaded headless virtual-seat module is unverifiable") from exc
    expected_device = (os.major(stat.st_dev), os.minor(stat.st_dev))
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1]:
            continue
        # procfs escapes whitespace in file names with octal sequences.
        mapped_path = fields[5].replace("\\040", " ").replace("\\011", "\t")
        if mapped_path != str(module):
            continue
        try:
            device = tuple(int(part, 16) for part in fields[3].split(":"))
            inode = int(fields[4])
        except ValueError:
            continue
        if device == expected_device and inode == stat.st_ino:
            return
    raise IsolationError("verified virtual-seat binary is not loaded in the live compositor")


def _artifact_directory(root: Path | None = None) -> Path:
    return (SEAT_ARTIFACT_ROOT if root is None else root) / (
        f"weston-{WESTON_SEAT_VERSION}-{_sha256(SEAT_SOURCE)}"
    )


def load_headless_seat_artifact(*, root: Path | None = None) -> HeadlessSeatArtifact:
    try:
        manifest = _artifact_directory(root) / "manifest.json"
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        artifact = HeadlessSeatArtifact(**raw)
    except (OSError, ValueError, TypeError) as exc:
        raise IsolationError(
            "headless virtual-seat module is not built; run "
            "python -m minecraft_ai.platforms.weston_seat after installing "
            "libweston-13-dev and a C compiler"
        ) from exc
    require_headless_seat_artifact(artifact)
    return artifact


def build_headless_seat_module(
    *,
    root: Path | None = None,
    include_root: Path | None = None,
) -> HeadlessSeatArtifact:
    """Build once for this exact source; optional headers support extracted packages.

    The helper does not install packages, change a compositor, or launch a game.
    An existing valid artifact is reused so a rebuild cannot replace a live inode.
    """
    if sys.platform == "win32":
        raise IsolationError("headless virtual-seat publication requires POSIX file locking")
    import fcntl

    selected_root = (SEAT_ARTIFACT_ROOT if root is None else root).resolve()
    selected_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(selected_root, 0o700)
    with (selected_root / ".build.lock").open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _build_headless_seat_module_locked(root=selected_root, include_root=include_root)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _build_headless_seat_module_locked(
    *,
    root: Path,
    include_root: Path | None,
) -> HeadlessSeatArtifact:
    directory = _artifact_directory(root).resolve()
    if (directory / "manifest.json").exists():
        return load_headless_seat_artifact(root=root)
    if (directory / "headless-seat.so").exists():
        raise IsolationError(
            "incomplete headless virtual-seat publication; refusing binary replacement"
        )
    source_sha256 = _sha256(SEAT_SOURCE)
    compiler = shutil.which("cc")
    if compiler is None:
        raise IsolationError("a C compiler is required to build the headless virtual seat")
    if include_root is None:
        try:
            version = subprocess.check_output(
                ["pkg-config", "--modversion", "libweston-13"],
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
            if version != WESTON_SEAT_VERSION:
                raise IsolationError(f"headless virtual seat requires Weston {WESTON_SEAT_VERSION}")
            flags = shlex.split(
                subprocess.check_output(
                    ["pkg-config", "--cflags", "libweston-13", "wayland-server"],
                    text=True,
                    stderr=subprocess.PIPE,
                )
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise IsolationError(
                "headless virtual-seat build requires libweston-13-dev, "
                "libwayland-dev, libxkbcommon-dev and libpixman-1-dev"
            ) from exc
    else:
        headers = include_root.resolve() / "usr" / "include"
        flags = [f"-I{headers}", f"-I{headers / 'libweston-13'}", f"-I{headers / 'pixman-1'}"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    output = directory / "headless-seat.so"
    fd, name = tempfile.mkstemp(prefix=".headless-seat-", suffix=".so", dir=directory)
    os.close(fd)
    staged = Path(name)
    staged_manifest: Path | None = None
    try:
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-O2",
                "-fPIC",
                "-shared",
                *flags,
                str(SEAT_SOURCE),
                "-o",
                str(staged),
                "-Wl,-z,defs",
                "-Wl,-l:libweston-13.so.0",
                "-Wl,-l:libwayland-server.so.0",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if _sha256(SEAT_SOURCE) != source_sha256:
            raise IsolationError("headless virtual-seat source changed during compilation")
        os.chmod(staged, 0o500)
        # Link is atomic and cannot replace an inode used by an existing process.
        os.link(staged, output)
        artifact = HeadlessSeatArtifact(str(output), _sha256(output), source_sha256)
        manifest = directory / "manifest.json"
        fd, name = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=directory)
        os.close(fd)
        staged_manifest = Path(name)
        staged_manifest.write_text(json.dumps(asdict(artifact), indent=2) + "\n", encoding="utf-8")
        os.chmod(staged_manifest, 0o600)
        os.link(staged_manifest, manifest)
        require_headless_seat_artifact(artifact)
        return artifact
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise IsolationError(f"headless virtual-seat build failed: {detail}") from exc
    finally:
        staged.unlink(missing_ok=True)
        if staged_manifest is not None:
            staged_manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Minecraft AI's private virtual-seat module")
    parser.add_argument("--include-root", type=Path, help="Root containing extracted usr/include")
    parser.add_argument("--output-root", type=Path, help="Override the module artifact directory")
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(
                build_headless_seat_module(
                    root=args.output_root,
                    include_root=args.include_root,
                )
            ),
            indent=2,
        )
    )
