#!/usr/bin/env python3
"""Train and evidence-gate a tiny Bedrock scene classifier from trajectory shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import tarfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


LABELS = ("world", "inventory")
IMAGE_SIZE = (160, 96)


class FastSceneNet(nn.Module):
    """Small fully convolutional model suitable for every policy observation."""

    def __init__(self, class_count: int = len(LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(12, 96),
            nn.SiLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            # Bedrock GUI panels occupy stable screen regions. Retaining a
            # coarse spatial grid prevents the classifier from explaining the
            # label with biome or brightness statistics alone.
            nn.AdaptiveAvgPool2d((3, 5)),
        )
        self.classifier = nn.Linear(128 * 3 * 5, class_count)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image).flatten(1))


@dataclass(frozen=True)
class Sample:
    image: torch.Tensor
    label: int
    trajectory_id: str
    step_index: int
    split: str


class TrajectoryReader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._manifests: dict[str, dict[str, Any]] = {}
        self._archives: dict[Path, tarfile.TarFile] = {}

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def frame(self, trajectory_id: str, step_index: int) -> np.ndarray:
        directory = self.root / trajectory_id
        manifest = self._manifests.get(trajectory_id)
        if manifest is None:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if not manifest.get("shards"):
                manifest["shards"] = self._index_legacy_shards(directory, manifest)
            self._manifests[trajectory_id] = manifest
        shard = next(
            (
                item
                for item in manifest.get("shards", ())
                if int(item["first_step_index"]) <= step_index <= int(item["last_step_index"])
            ),
            None,
        )
        if shard is None:
            raise ValueError(f"step {step_index} is absent from completed {trajectory_id}")
        path = directory / str(shard["filename"])
        archive = self._archives.get(path)
        if archive is None:
            archive = tarfile.open(path, mode="r")
            self._archives[path] = archive
        key = f"{step_index:012d}"
        header_member = archive.extractfile(f"{key}.frame.json")
        frame_member = archive.extractfile(f"{key}.frame.bgra.zlib")
        if header_member is None or frame_member is None:
            raise ValueError(f"incomplete frame sample {trajectory_id}:{step_index}")
        header = json.loads(header_member.read())
        if header.get("codec") != "zlib" or header.get("pixel_format") != "BGRA":
            raise ValueError(f"unsupported frame encoding at {trajectory_id}:{step_index}")
        raw = zlib.decompress(frame_member.read())
        expected = int(header["width"]) * int(header["height"]) * 4
        if len(raw) != expected or int(header["raw_bytes"]) != expected:
            raise ValueError(f"corrupt frame payload at {trajectory_id}:{step_index}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            int(header["height"]),
            int(header["width"]),
            4,
        )

    def _index_legacy_shards(
        self,
        directory: Path,
        manifest: dict[str, Any],
    ) -> list[dict[str, int | str]]:
        """Recover ranges for v1 manifests that only persisted shard IDs."""
        indexed: list[dict[str, int | str]] = []
        shard_ids = tuple(manifest.get("shard_ids", ()))
        if not shard_ids:
            shard_ids = tuple(
                path.stem for path in sorted(directory.glob("*-shard-*.tar"))
            )
        for shard_id in shard_ids:
            path = directory / f"{shard_id}.tar"
            archive = self._archives.get(path)
            if archive is None:
                archive = tarfile.open(path, mode="r")
                self._archives[path] = archive
            step_indices = [
                int(member.name.removesuffix(".step.json"))
                for member in archive.getmembers()
                if member.name.endswith(".step.json")
            ]
            if not step_indices:
                raise ValueError(f"trajectory shard contains no steps: {path}")
            indexed.append(
                {
                    "filename": path.name,
                    "first_step_index": min(step_indices),
                    "last_step_index": max(step_indices),
                }
            )
        return indexed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_samples(annotation_path: Path, trajectory_root: Path) -> list[Sample]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if tuple(annotation["labels"]) != LABELS:
        raise ValueError(f"labels must be exactly {LABELS}")
    offsets = annotation["sampling"]
    before_offsets = tuple(int(value) for value in offsets["before_offsets"])
    after_offsets = tuple(int(value) for value in offsets["after_offsets"])
    split_trajectories: dict[str, set[str]] = {}
    samples: list[Sample] = []
    reader = TrajectoryReader(trajectory_root)

    def add_sample(
        trajectory_id: str,
        step_index: int,
        label: str,
        split: str,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split {split!r}")
        if label not in LABELS:
            raise ValueError(f"invalid label {label!r}")
        split_trajectories.setdefault(split, set()).add(trajectory_id)
        bgra = reader.frame(trajectory_id, step_index)
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        resized = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(
            resized.transpose(2, 0, 1).copy()
        ).float().div_(255.0)
        samples.append(
            Sample(
                image=image,
                label=LABELS.index(label),
                trajectory_id=trajectory_id,
                step_index=step_index,
                split=split,
            )
        )

    try:
        for transition in annotation["transitions"]:
            trajectory_id = str(transition["trajectory_id"])
            split = str(transition["split"])
            direction = str(transition["transition"])
            if direction not in {"open", "close"}:
                raise ValueError(f"invalid transition {direction!r}")
            center = int(transition["step_index"])
            label_pairs = (
                ((before_offsets, "world"), (after_offsets, "inventory"))
                if direction == "open"
                else ((before_offsets, "inventory"), (after_offsets, "world"))
            )
            for selected_offsets, label in label_pairs:
                for offset in selected_offsets:
                    add_sample(
                        trajectory_id,
                        center + offset,
                        label,
                        split,
                    )
        for example in annotation.get("examples", ()):
            trajectory_id = str(example["trajectory_id"])
            label = str(example["label"])
            split = str(example["split"])
            for step_index in example["step_indices"]:
                add_sample(trajectory_id, int(step_index), label, split)
    finally:
        reader.close()
    split_names = tuple(split_trajectories)
    for index, split in enumerate(split_names):
        for other in split_names[index + 1 :]:
            overlap = split_trajectories[split] & split_trajectories[other]
            if overlap:
                raise ValueError(f"trajectory leakage between {split} and {other}: {overlap}")
    unique: dict[tuple[str, int], Sample] = {}
    for item in samples:
        identity = (item.trajectory_id, item.step_index)
        existing = unique.get(identity)
        if existing is not None and (
            existing.label != item.label or existing.split != item.split
        ):
            raise ValueError(f"conflicting annotation for frame sample {identity}")
        unique[identity] = item
    return list(unique.values())


def _augment(images: torch.Tensor) -> torch.Tensor:
    batch = images.clone()
    flip = torch.rand(batch.shape[0], device=batch.device) < 0.5
    batch[flip] = torch.flip(batch[flip], dims=(3,))
    brightness = torch.empty(batch.shape[0], 1, 1, 1, device=batch.device).uniform_(0.75, 1.25)
    contrast = torch.empty(batch.shape[0], 1, 1, 1, device=batch.device).uniform_(0.8, 1.2)
    mean = batch.mean(dim=(2, 3), keepdim=True)
    batch = (batch - mean) * contrast + mean
    batch = batch * brightness
    batch = batch + torch.randn_like(batch) * 0.015
    return batch.clamp_(0.0, 1.0)


def _evaluate(
    model: nn.Module,
    samples: list[Sample],
    *,
    confidence_threshold: float,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("evaluation split is empty")
    images = torch.stack([item.image for item in samples])
    labels = torch.tensor([item.label for item in samples], dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values
        loss = nn.functional.cross_entropy(logits, labels).item()
    confusion = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.int64)
    for actual, predicted in zip(labels, predictions, strict=True):
        confusion[int(actual), int(predicted)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(LABELS):
        actual_count = int((labels == index).sum())
        correct = int(confusion[index, index])
        confident_correct_mask = (
            (labels == index)
            & (predictions == index)
            & (confidence >= confidence_threshold)
        )
        confident_correct = int(confident_correct_mask.sum())
        per_class[label] = {
            "samples": actual_count,
            "recall": correct / max(1, actual_count),
            "confident_recall": confident_correct / max(1, actual_count),
        }
    world_index = LABELS.index("world")
    inventory_index = LABELS.index("inventory")
    world_count = int((labels == world_index).sum())
    false_inventory = int(
        (
            (labels == world_index)
            & (predictions == inventory_index)
            & (confidence >= confidence_threshold)
        ).sum()
    )
    return {
        "samples": len(samples),
        "loss": loss,
        "accuracy": float((predictions == labels).float().mean()),
        "coverage": float((confidence >= confidence_threshold).float().mean()),
        "false_inventory_rate": false_inventory / max(1, world_count),
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def _train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    samples = _load_samples(args.annotations, args.trajectory_root)
    by_split = {
        split: [item for item in samples if item.split == split]
        for split in ("train", "validation", "test")
    }
    model = FastSceneNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_images = torch.stack([item.image for item in by_split["train"]])
    train_labels = torch.tensor([item.label for item in by_split["train"]], dtype=torch.long)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = (-math.inf, -math.inf, -math.inf)
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_labels))
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            logits = model(_augment(train_images[indices]))
            loss = nn.functional.cross_entropy(logits, train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % args.evaluate_every == 0 or epoch == args.epochs:
            validation = _evaluate(
                model,
                by_split["validation"],
                confidence_threshold=args.confidence_threshold,
            )
            train_evaluation = _evaluate(
                model,
                by_split["train"],
                confidence_threshold=args.confidence_threshold,
            )
            score = (
                float(validation["accuracy"]),
                float(train_evaluation["accuracy"]),
                -float(validation["loss"]),
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("training did not produce a candidate state")
    model.load_state_dict(best_state)
    split_metrics = {
        split: _evaluate(
            model,
            split_samples,
            confidence_threshold=args.confidence_threshold,
        )
        for split, split_samples in by_split.items()
    }
    train_metrics = split_metrics["train"]
    test_metrics = split_metrics["test"]
    validation_metrics = split_metrics["validation"]
    inventory_metrics = test_metrics["per_class"]["inventory"]
    validation_inventory_metrics = validation_metrics["per_class"]["inventory"]
    promotion_eligible = bool(
        train_metrics["accuracy"] >= args.minimum_train_accuracy
        and train_metrics["false_inventory_rate"]
        <= args.maximum_false_inventory_rate
        and validation_metrics["accuracy"] >= args.minimum_validation_accuracy
        and validation_metrics["false_inventory_rate"]
        <= args.maximum_false_inventory_rate
        and validation_inventory_metrics["confident_recall"]
        >= args.minimum_confident_inventory_recall
        and test_metrics["accuracy"] >= args.minimum_test_accuracy
        and test_metrics["false_inventory_rate"] <= args.maximum_false_inventory_rate
        and inventory_metrics["confident_recall"] >= args.minimum_confident_inventory_recall
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "bedrock-fast-scene-v1.pt"
    if artifact.exists() and not args.overwrite:
        raise FileExistsError(f"artifact exists; pass --overwrite to replace {artifact}")
    staged = artifact.with_name(f".{artifact.name}.{os.getpid()}.tmp")
    model.eval()
    traced = torch.jit.trace(model, torch.zeros(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]))
    torch.jit.save(traced, staged)
    staged.replace(artifact)
    artifact_sha256 = _sha256(artifact)
    report = {
        "schema_version": 1,
        "model_version": "bedrock-fast-scene-v1",
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256,
        "git_commit": _git_commit(args.repo),
        "dataset": str(args.annotations),
        "dataset_sha256": _sha256(args.annotations),
        "trajectory_root": str(args.trajectory_root),
        "labels": list(LABELS),
        "input": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1], "format": "RGB-f32-0-1"},
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "confidence_threshold": args.confidence_threshold,
        "samples": {split: len(items) for split, items in by_split.items()},
        "metrics": split_metrics,
        "promotion_gate": {
            "eligible": promotion_eligible,
            "minimum_train_accuracy": args.minimum_train_accuracy,
            "minimum_validation_accuracy": args.minimum_validation_accuracy,
            "minimum_test_accuracy": args.minimum_test_accuracy,
            "maximum_false_inventory_rate": args.maximum_false_inventory_rate,
            "minimum_confident_inventory_recall": (
                args.minimum_confident_inventory_recall
            ),
        },
        "trained_ns": time.time_ns(),
    }
    manifest = artifact.with_suffix(".manifest.json")
    manifest_staged = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    manifest_staged.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_staged.replace(manifest)
    report["manifest"] = str(manifest)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=repo / "datasets/scene/bedrock_inventory_transitions_v1.json",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=Path.home() / ".local/share/minecraft-ai/trajectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / ".local/share/minecraft-ai/models/fast-scene",
    )
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--evaluate-every", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--minimum-train-accuracy", type=float, default=0.98)
    parser.add_argument("--minimum-validation-accuracy", type=float, default=0.98)
    parser.add_argument("--minimum-test-accuracy", type=float, default=0.98)
    parser.add_argument("--maximum-false-inventory-rate", type=float, default=0.0)
    parser.add_argument("--minimum-confident-inventory-recall", type=float, default=0.95)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1928)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = _train(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["promotion_gate"]["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
