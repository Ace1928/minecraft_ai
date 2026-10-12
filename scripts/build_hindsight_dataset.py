#!/usr/bin/env python3
"""Build verified goal-action training examples from recorded Bedrock experience."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minecraft_ai.config import app_paths
from minecraft_ai.datasets.hindsight import (
    discover_trajectory_paths,
    write_hindsight_dataset,
)


def _parser() -> argparse.ArgumentParser:
    paths = app_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        default=[],
        help="Sealed trajectory directory or manifest; repeat to combine trajectories.",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=paths.data_dir / "trajectories",
        help="Used to discover sealed trajectories when --trajectory is omitted.",
    )
    parser.add_argument(
        "--maximum-trajectories",
        type=int,
        default=None,
        help="Limit automatic discovery to the newest N sealed, non-empty trajectories.",
    )
    parser.add_argument("--state-db", type=Path, default=paths.state_db)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.data_dir / "datasets" / "hindsight",
    )
    parser.add_argument("--history-steps", type=int, default=128)
    parser.add_argument("--partial-window-steps", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    trajectories = tuple(args.trajectory) or discover_trajectory_paths(
        args.trajectory_root,
        maximum=args.maximum_trajectories,
    )
    if not trajectories:
        raise SystemExit("no sealed, non-empty Bedrock trajectories were selected")
    artifact = write_hindsight_dataset(
        trajectories,
        state_db_path=args.state_db,
        output_dir=args.output_dir,
        history_steps=args.history_steps,
        partial_window_steps=args.partial_window_steps,
    )
    output = {
        "data_path": str(artifact.data_path),
        "data_sha256": artifact.data_sha256,
        "manifest_path": str(artifact.manifest_path),
        "manifest_sha256": artifact.manifest_sha256,
        "examples": artifact.manifest.examples,
        "successful_examples": artifact.manifest.successful_examples,
        "partial_progress_examples": artifact.manifest.partial_progress_examples,
        "source_trajectories": len(artifact.manifest.sources),
        "training_allowed": artifact.manifest.training_allowed,
        "automatic_promotion": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
