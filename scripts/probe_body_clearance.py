#!/usr/bin/env python3
"""Survey a retained game image without capturing, moving or publishing facts.

The configured local VLM must already be running with a single inference slot.
Refuse a busy server to avoid adding diagnostic work behind live cognition.
This is an offline visual qualification tool, not a gameplay success test.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import httpx
from PIL import Image

from minecraft_ai.body_clearance import BodyClearanceSurveyor, BodyClearanceValidationError
from minecraft_ai.config import load_config
from minecraft_ai.models import OpenAICompatibleLocalModel
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--feature", choices=("underside", "riser", "side_face"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists; choose a new evidence record")
    config = load_config(args.config).vision_language
    if not config.enabled:
        raise SystemExit("configure an enabled local vision model first")
    model = OpenAICompatibleLocalModel(
        model_id=config.model_id, base_url=config.base_url, api_key=config.api_key,
        timeout_s=min(config.timeout_s, 90), max_tokens=96,
        thinking_budget_tokens=config.thinking_budget_tokens,
        reasoning_format=config.reasoning_format,
    )
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(
                config.base_url.rstrip("/").removesuffix("/v1") + "/slots",
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
            response.raise_for_status()
            slots = response.json()
    except (httpx.HTTPError, ValueError):
        raise SystemExit("model availability unconfirmed; no diagnostic submitted") from None
    # A single server slot serializes even a racing live request. The preflight
    # is advisory, not a cross-process admission lock: do not run probe batches.
    if (
        not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict)
        or slots[0].get("is_processing") is not False
    ):
        raise SystemExit("live model busy or not single-slot; no diagnostic submitted")
    image_bytes = args.image.read_bytes()
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGBA")
    frame = CapturedFrame(
        # Offline identities explicitly do not impersonate a current capture.
        frame_id=0, captured_ns=1, width=image.width, height=image.height,
        bgra=image.tobytes("raw", "BGRA"),
    )
    try:
        result = BodyClearanceSurveyor(model).inspect(frame, requested_feature=args.feature)
    except BodyClearanceValidationError as exc:
        result = exc.inspection
    report = {
        "mode": "offline_observation_only",
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "model": config.model_id,
        "requested_feature": args.feature,
        "status": "parsed" if result.validation_error is None else "validation_error",
        "validation_error": result.validation_error,
        "model_input_png_sha256": result.model_input_sha256,
        "model_input_size": result.model_input_size,
        "evidence": result.evidence.model_dump(mode="json"),
        "candidate": None if result.candidate is None else asdict(result.candidate),
        "latency_ms": result.latency_ms,
        "raw_response": result.raw_response,
        "collision_verdict": "unknown",
        "metric_depth": None,
        "action_authorized": False,
        "independent_visual_review": "required",
    }
    encoded = json.dumps(report, indent=2)
    with args.output.open("x") as stream:
        stream.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
