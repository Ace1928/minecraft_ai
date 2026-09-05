"""Keep local adapters and model artifacts out of the public tracked tree."""

import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOTS = {
    ".local", "private", "private_adapters", "local_adapters",
    "checkpoints", "trajectories", "runtime",
}
LOCAL_SUBTREES = {
    ("adapters", "private"), ("adapters", "local"),
    ("models", "artifacts"), ("models", "checkpoints"),
    ("models", "local"), ("models", "private"),
}
ARTIFACT_SUFFIXES = (
    ".safetensors", ".safetensors.index.json", ".gguf", ".ggml", ".weights",
    ".ckpt", ".pt", ".pth", ".onnx", ".model", ".bin", ".bin.index.json",
    ".pkl", ".pickle", ".npy", ".npz", ".sqlite3", ".sqlite3-shm",
    ".sqlite3-wal", ".log", ".local.yaml", ".local.yml", ".local.json",
)


def forbidden_public_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        path.parts[0] in LOCAL_ROOTS
        or path.parts[:2] in LOCAL_SUBTREES
        or path.name.endswith(ARTIFACT_SUFFIXES)
        or (path.name == ".env" or path.name.startswith(".env."))
        and path.name != ".env.example"
    )


def git_output(*args: str, input_text: str | None = None) -> str:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("tracked-tree checks require a Git checkout")
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_tracked_tree_excludes_private_paths_and_model_artifacts() -> None:
    paths = git_output("ls-files", "--cached", "-z").split("\0")
    forbidden = [path for path in paths if path and forbidden_public_path(path)]
    # Report only the count, not potentially private filenames or contents.
    assert not forbidden, f"{len(forbidden)} tracked private/runtime/model artifact path(s)"


def test_ignore_rules_protect_local_artifacts_without_hiding_generic_code() -> None:
    protected = (
        ".env", ".env.production", "nested/.env.development",
        "config.local.yaml", "settings.local.yml", "config.local.json",
        ".local/config.yaml", "private/adapter.py", "private_adapters/worker.py",
        "local_adapters/worker.py", "adapters/private/worker.py", "adapters/local/worker.py",
        "checkpoints/config.json", "trajectories/run/step.jsonl", "runtime/status.json",
        "models/artifacts/config.json", "models/checkpoints/config.json",
        "models/local/config.json", "models/private/config.json",
        *(f"nested/payload{suffix}" for suffix in ARTIFACT_SUFFIXES),
    )
    public = (
        ".env.example", "nested/.env.example", "src/minecraft_ai/config.py",
        "src/minecraft_ai/policy_service.py", "tests/test_external_temporal_worker.py",
        "models/registry/example.yaml", "adapters/example_worker.py",
        "docs/ARCHITECTURE.md",
    )
    assert all(forbidden_public_path(path) for path in protected)
    assert not any(forbidden_public_path(path) for path in public)
    ignored = git_output(
        "check-ignore", "--no-index", "--stdin", "-z",
        input_text="\0".join((*protected, *public)) + "\0",
    )
    assert set(filter(None, ignored.split("\0"))) == set(protected)
