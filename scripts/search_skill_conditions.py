#!/usr/bin/env python3
"""Evaluate replay-bound STEVE instruction/CFG candidates without promoting them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minecraft_ai.skill_condition_search import (
    content_address,
    evaluate_skill_conditions,
    load_search_spec,
    write_content_addressed_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON search spec binding candidates and initial-pose seeds to trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / ".local/share/minecraft-ai/evaluations/skill-conditions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and evaluate all artifacts, print the result, but write nothing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = evaluate_skill_conditions(load_search_spec(args.spec))
    digest = content_address(manifest)
    path: str | None = None
    if not args.dry_run:
        artifact = write_content_addressed_manifest(manifest, args.output_dir)
        digest = artifact.sha256
        path = str(artifact.path)
    output = {
        "dry_run": bool(args.dry_run),
        "manifest_path": path,
        "manifest_sha256": digest,
        "comparison_ready": manifest.comparison.comparison_ready,
        "leading_candidate_id": manifest.comparison.leading_candidate_id,
        "automatic_promotion": False,
        "manifest": manifest.model_dump(mode="json"),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
