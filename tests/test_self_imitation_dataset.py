from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from minecraft_ai.datasets.self_imitation import write_self_imitation_dataset
from test_hindsight_dataset import _record_fixture
from minecraft_ai.datasets.hindsight import write_hindsight_dataset


def _hindsight(tmp_path: Path, *, training_allowed: bool = True):
    trajectory, database = _record_fixture(tmp_path, training_allowed=training_allowed)
    return write_hindsight_dataset(
        (trajectory,),
        state_db_path=database,
        output_dir=tmp_path / "hindsight",
        history_steps=4,
        partial_window_steps=3,
    )


def test_self_imitation_deduplicates_actions_and_keeps_verified_success(tmp_path: Path) -> None:
    hindsight = _hindsight(tmp_path)

    artifact = write_self_imitation_dataset(
        (hindsight.manifest_path,),
        output_dir=tmp_path / "self-imitation",
        beta=0.25,
        maximum_awr_weight=4.0,
        elite_fraction=0.5,
    )

    assert artifact.manifest.source_examples == 7
    assert artifact.manifest.eligible_prediction_examples == 7
    assert artifact.manifest.discarded_non_prediction_examples == 0
    assert artifact.manifest.unique_actions == 5
    assert artifact.manifest.examples == 4
    assert artifact.manifest.successful_examples == 2
    assert artifact.manifest.partial_progress_examples == 2
    assert artifact.manifest.automatic_promotion is False
    examples = [json.loads(line) for line in artifact.data_path.read_text().splitlines()]
    action_ids = {
        (example["trajectory_id"], example["step_index"], example["skill_run_id"])
        for example in examples
    }
    assert len(action_ids) == len(examples)
    assert all(example["training_weight"] > 0 for example in examples)
    assert all(example["source_example_id"] == example["example_id"] for example in examples)
    assert all(example["policy_action_kind"] == "prediction" for example in examples)
    assert all(example["held_keys"] == ["w"] for example in examples)


def test_self_imitation_artifact_is_reproducible(tmp_path: Path) -> None:
    hindsight = _hindsight(tmp_path)
    output = tmp_path / "self-imitation"

    first = write_self_imitation_dataset((hindsight.manifest_path,), output_dir=output)
    second = write_self_imitation_dataset((hindsight.manifest_path,), output_dir=output)

    assert first.data_sha256 == second.data_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest.training_allowed is True
    assert first.manifest.redistribution_allowed is True


def test_self_imitation_refuses_unlicensed_training_source(tmp_path: Path) -> None:
    hindsight = _hindsight(tmp_path)
    rejected = hindsight.manifest.model_copy(update={"training_allowed": False})
    raw = (
        json.dumps(
            rejected.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    rejected_path = hindsight.manifest_path.parent / (
        hashlib.sha256(raw).hexdigest() + ".manifest.json"
    )
    rejected_path.write_bytes(raw)

    with pytest.raises(ValueError, match="not licensed for training"):
        write_self_imitation_dataset(
            (rejected_path,),
            output_dir=tmp_path / "self-imitation",
        )


def test_self_imitation_rejects_tampered_hindsight_data(tmp_path: Path) -> None:
    hindsight = _hindsight(tmp_path)
    with hindsight.data_path.open("ab") as handle:
        handle.write(b"{}\n")

    with pytest.raises(ValueError, match="integrity mismatch"):
        write_self_imitation_dataset(
            (hindsight.manifest_path,),
            output_dir=tmp_path / "self-imitation",
        )


def test_self_imitation_refuses_hold_only_legacy_targets(tmp_path: Path) -> None:
    hindsight = _hindsight(tmp_path)
    records = [json.loads(line) for line in hindsight.data_path.read_text().splitlines()]
    for record in records:
        record["policy_action_kind"] = "prediction_hold"
    data = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for record in records
    )
    data_sha256 = hashlib.sha256(data).hexdigest()
    data_path = hindsight.data_path.parent / f"{data_sha256}.jsonl"
    data_path.write_bytes(data)
    manifest = hindsight.manifest.model_copy(
        update={"data_filename": data_path.name, "data_sha256": data_sha256}
    )
    manifest_raw = (
        json.dumps(
            manifest.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    manifest_path = hindsight.manifest_path.parent / (
        hashlib.sha256(manifest_raw).hexdigest() + ".manifest.json"
    )
    manifest_path.write_bytes(manifest_raw)

    with pytest.raises(ValueError, match="no fresh policy predictions"):
        write_self_imitation_dataset(
            (manifest_path,),
            output_dir=tmp_path / "self-imitation-holds",
        )
