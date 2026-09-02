from pathlib import Path

import pytest

from minecraft_ai.config import PolicyConfig
from minecraft_ai.policy_service import LearnedPolicyOutput, _validate_policy_config


def test_learned_policy_output_rejects_unknown_action_fields() -> None:
    with pytest.raises(ValueError):
        LearnedPolicyOutput.model_validate(
            {
                "keys": [],
                "buttons": [],
                "mouse_dx": 0,
                "mouse_dy": 0,
                "inference_ns": 1,
                "model_version": "v1",
                "scripted_recovery": True,
            }
        )


def test_policy_config_requires_hashes_provenance_and_paths(tmp_path: Path) -> None:
    executable = tmp_path / "python"
    source = tmp_path / "source"
    model = tmp_path / "model"
    weights = tmp_path / "weights"
    executable.touch()
    source.mkdir()
    model.touch()
    weights.touch()
    config = PolicyConfig(
        enabled=True,
        python_path=str(executable),
        source_path=str(source),
        model_path=str(model),
        weights_path=str(weights),
        model_sha256="a" * 64,
        weights_sha256="b" * 64,
        model_version="official-v1",
        source_commit="c" * 40,
        license="MIT",
    )
    _validate_policy_config(config)

    with pytest.raises(ValueError, match="license"):
        _validate_policy_config(config.model_copy(update={"license": "unknown"}))
