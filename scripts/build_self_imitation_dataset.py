#!/usr/bin/env python3
"""Build elite, advantage-weighted motor examples from verified hindsight data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minecraft_ai.config import app_paths
from minecraft_ai.datasets.self_imitation import write_self_imitation_dataset


def _parser() -> argparse.ArgumentParser:
    paths = app_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hindsight-manifest",
        type=Path,
        action="append",
        required=True,
        help="Verified hindsight manifest; repeat to combine independent recordings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.data_dir / "datasets" / "self-imitation",
    )
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--maximum-awr-weight", type=float, default=4.0)
    parser.add_argument("--elite-fraction", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = write_self_imitation_dataset(
        args.hindsight_manifest,
        output_dir=args.output_dir,
        beta=args.beta,
        maximum_awr_weight=args.maximum_awr_weight,
        elite_fraction=args.elite_fraction,
    )
    print(
        json.dumps(
            {
                "data_path": str(artifact.data_path),
                "data_sha256": artifact.data_sha256,
                "manifest_path": str(artifact.manifest_path),
                "manifest_sha256": artifact.manifest_sha256,
                "examples": artifact.manifest.examples,
                "source_examples": artifact.manifest.source_examples,
                "eligible_prediction_examples": (
                    artifact.manifest.eligible_prediction_examples
                ),
                "discarded_non_prediction_examples": (
                    artifact.manifest.discarded_non_prediction_examples
                ),
                "unique_actions": artifact.manifest.unique_actions,
                "successful_examples": artifact.manifest.successful_examples,
                "partial_progress_examples": artifact.manifest.partial_progress_examples,
                "training_allowed": artifact.manifest.training_allowed,
                "automatic_promotion": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
