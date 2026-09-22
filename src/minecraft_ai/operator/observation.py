"""Opt-in, model-neutral observation export. No capture, inference or game control.

A private producer may write a source-stamped packet using publish_observation.
Nothing in this module infers consumed pixels or activations from runtime counters.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image
from platformdirs import user_runtime_dir

SCHEMA = "minecraft.observation.v1"
OBSERVATION_FILE = Path(user_runtime_dir("minecraft-ai")) / "public-observation.json"
MAX_BYTES = 240_000
FRESH_NS = 30_000_000_000
SOURCES = {"native-policy", "association-brain"}
HASH = re.compile(r"[a-f0-9]{64}\Z")
STREAM = re.compile(r"[a-f0-9]{32}\Z")
BUTTONS = {"forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use"}


def _integer(value: Any, maximum: int = 2**53 - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid observation count")
    return value


def _number(value: Any, maximum: float = 1.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > maximum:
        raise ValueError("invalid observation magnitude")
    return float(value)


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError("invalid observation digest")
    return value


def _image(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    width, height = _integer(raw["width"], 1024), _integer(raw["height"], 1024)
    digest = _hash(raw["sha256"])
    encoded = raw["png_base64"]
    if not isinstance(encoded, str) or len(encoded) > 180_000 or not width or not height:
        raise ValueError("invalid consumed image")
    data = base64.b64decode(encoded, validate=True)
    if base64.b64encode(data).decode("ascii") != encoded:
        raise ValueError("noncanonical consumed image")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("consumed image digest mismatch")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (width, height) or image.mode != "RGB":
            raise ValueError("consumed image format mismatch")
        image.load()
        # PNG metadata can contain arbitrary private text. Re-encode and require
        # the producer to supply this metadata-free representation of its pixels.
        clean = io.BytesIO()
        image.save(clean, format="PNG")
        if clean.getvalue() != data:
            raise ValueError("consumed image must be metadata-free canonical PNG")
    return {"width": width, "height": height, "sha256": digest, "png_base64": encoded}


def _model(raw: Any, component_id: int) -> dict[str, Any]:
    if type(raw["available"]) is not bool:
        raise ValueError("invalid model availability")
    available = raw["available"]
    calls = _integer(raw["calls"], 1_000_000)
    count = _integer(raw["source_unit_count"])
    binding = _hash(raw["binding_sha256"]) if available else None
    population = raw.get("population")
    out: dict[str, Any] = dict(available=available, calls=calls, source_unit_count=count,
                               binding_sha256=binding, population=None)
    basis = raw.get("activity_basis", "observed-activation")
    if basis not in {"observed-activation", "readout-input"}:
        raise ValueError("invalid activity basis")
    reused = raw.get("activity_reused", False)
    if type(reused) is not bool:
        raise ValueError("invalid activity reuse")
    kcs = _integer(raw.get("source_kc_count", 0), count)
    active = _integer(raw.get("source_active_kcs", 0), kcs)
    out.update(activity_basis=basis, activity_reused=reused,
               source_kc_count=kcs, source_active_kcs=active)
    if not available and (calls or count or population is not None):
        raise ValueError("unavailable model cannot claim activity")
    if population is None:
        return out
    if not available or not calls:
        raise ValueError("activity requires a bound, executed model")
    n = _integer(population["unit_count"], 1024)
    layers = [_integer(v, 1024) for v in population["layer_sizes"]]
    positions, edges, activity = (population[k] for k in ("positions", "edges", "activity"))
    if (not n or n > count or not 1 <= len(layers) <= 32 or not all(layers)
            or sum(layers) != n or len(positions) != n or len(edges) > 2048
            or len(activity) > n):
        raise ValueError("modeled population budget exceeded")
    points = []
    for p in positions:
        if len(p) != 4:
            raise ValueError("invalid modeled position")
        points.append([*[_number(v, 10) for v in p[:3]], _integer(p[3], len(layers) - 1)])
    links = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("invalid modeled edge")
        links.append([_integer(v, n - 1) for v in edge])
    values, seen = [], set()
    for pair in activity:
        if len(pair) != 2:
            raise ValueError("invalid modeled activation")
        index, value = _integer(pair[0], n - 1), _number(pair[1], 1e15)
        if index in seen:
            raise ValueError("duplicate modeled activation")
        seen.add(index)
        values.append([index, value])
    total = _integer(population["total_edges"])
    if total < len(links):
        raise ValueError("invalid modeled edge coverage")
    out["population"] = {
        "schema": "erais.neural-population.v1", "identity": _hash(population["identity"]),
        "implementation_sha256": _hash(population["implementation_sha256"]),
        "component_id": component_id, "unit_count": n, "layer_sizes": layers,
        "positions": points, "edges": links, "activity": values, "total_edges": total,
        "edges_complete": total == len(links),
    }
    if "source_indices" in population:
        indices = [_integer(i, count - 1) for i in population["source_indices"]]
        if len(indices) != n or len(set(indices)) != n:
            raise ValueError("invalid sampled source indices")
        out["population"]["source_indices"] = indices
    return out


def project_observation(
    raw: Any, *, now_ns: int | None = None,
    preferred_source: str = "native-policy", stream_id: str = "",
) -> dict[str, Any]:
    """Allowlist a producer packet. Capture time, not file mtime, owns freshness."""
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if preferred_source not in SOURCES or (stream_id and not STREAM.fullmatch(stream_id)):
        raise ValueError("invalid observation preference")
    safety = raw.get("safety") if isinstance(raw, dict) else None
    if (not isinstance(raw, dict) or raw.get("schema") != SCHEMA
            or raw.get("online", True) is not True
            or raw.get("source_id") != preferred_source
            or not isinstance(raw.get("stream_id"), str)
            or not STREAM.fullmatch(raw["stream_id"])
            or (stream_id and stream_id != raw["stream_id"])
            or not isinstance(safety, dict) or set(safety) != {"game_only", "chat_free"}
            or safety["game_only"] is not True or type(safety["chat_free"]) is not bool):
        raise ValueError("unverified observation source")
    captured = raw.get("source_captured_ns")
    if not isinstance(captured, str) or not re.fullmatch(r"[1-9][0-9]{0,19}", captured):
        raise ValueError("invalid capture stamp")
    age_ns = now_ns - int(captured)
    if not 0 <= age_ns < FRESH_NS:
        raise ValueError("observation expired")
    frame_id = _integer(raw["source_frame_id"])
    result = {
        "schema": SCHEMA, "online": True, "source_id": preferred_source,
        "safety": {"game_only": True, "chat_free": safety["chat_free"]},
        "stream_id": raw["stream_id"], "sequence": _integer(raw["sequence"], 10**12),
        "source_frame_id": frame_id, "source_captured_ns": captured,
        "frame_age_ms": age_ns / 1_000_000,
        "input": _image(raw.get("input")), "reconstruction": None,
    }
    if not safety["chat_free"] and (raw.get("input") is not None
                                   or raw.get("reconstruction") is not None):
        raise ValueError("pixels withheld without chat-free evidence")
    if "consumed_rgb_sha256" in raw:
        result["consumed_rgb_sha256"] = _hash(raw["consumed_rgb_sha256"])
    if raw.get("reconstruction") is not None:
        decoded = raw["reconstruction"]
        image = _image(decoded)
        assert image is not None
        for key in ("supported_fraction", "reprojection_rmse"):
            image[key] = _number(decoded[key])
            if image[key] < 0:
                raise ValueError("invalid reconstruction coverage")
        if result["input"] is None:
            raise ValueError("reconstruction without consumed input")
        result["reconstruction"] = image
    fields = raw["receptive_fields"]
    count = _integer(fields["source_count"])
    samples = fields["samples"]
    if not isinstance(samples, list) or len(samples) > min(128, count):
        raise ValueError("receptive field budget exceeded")
    projected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for sample in samples:
        index = _integer(sample["index"], count - 1)
        box = [_number(v) for v in sample["box"]]
        if (len(box) != 4 or min(box) < 0 or not box[2] or not box[3]
                or box[0] + box[2] > 1 or box[1] + box[3] > 1 or index in seen):
            raise ValueError("invalid receptive field")
        seen.add(index)
        projected.append(dict(index=index, box=box, value=_number(sample["value"])))
    scope = fields.get("scope", "consumed-rgb")
    if scope not in {"consumed-rgb", "sparse-luminance-8x8"}:
        raise ValueError("invalid receptive scope")
    if scope == "sparse-luminance-8x8":
        if count != 64 or len(projected) != 64 or any(
            p["value"] < 0 or not .19 <= p["box"][0] <= .81
            or not .17 <= p["box"][1] <= .59
            or p["box"][2] > 1 / 160 or p["box"][3] > 1 / 90 for p in projected
        ):
            raise ValueError("invalid sparse consumed luminance scope")
    elif projected and result["input"] is None:
        raise ValueError("RGB features without consumed pixels")
    result["receptive_fields"] = dict(source_count=count, scope=scope, samples=projected)
    result["models"] = {key: _model(raw["models"][key], i) for i, key in enumerate(
        ("native_policy", "association_brain"))}
    result["action"] = None
    action = raw.get("action")
    if action is not None:
        if (_integer(action["source_frame_id"]) != frame_id
                or action["source_captured_ns"] != captured
                or action["kind"] not in {
                    "prediction", "prediction_hold", "synthetic", "reset", "release"}
                or type(action["accepted"]) is not bool
                or action["outcome"] not in {"pending", "succeeded", "failed", "unavailable"}
                or not isinstance(action["buttons"], list)
                or len(action["buttons"]) > len(BUTTONS)
                or any(b not in BUTTONS for b in action["buttons"])
                or len(set(action["buttons"])) != len(action["buttons"])
                or len(action["camera"]) != 2):
            raise ValueError("unbound action or outcome")
        result["action"] = {"kind": action["kind"], "accepted": action["accepted"],
                            "outcome": action["outcome"], "buttons": action["buttons"],
                            "camera": [_number(v, 180) for v in action["camera"]],
                            "source_frame_id": frame_id, "source_captured_ns": captured}
        result["action"]["outcome_evidence_sha256"] = (
            _hash(action.get("outcome_evidence_sha256"))
            if action["outcome"] in {"succeeded", "failed"} else None
        )
        if action["outcome"] in {"succeeded", "failed"} and not action["accepted"]:
            raise ValueError("outcome requires accepted action evidence")
    if len(json.dumps(result, separators=(",", ":")).encode()) > MAX_BYTES:
        raise ValueError("observation budget exceeded")
    return result


def publish_observation(raw: dict[str, Any], *, path: Path = OBSERVATION_FILE) -> None:
    """Optional producer hook; write only validated, public-safe data atomically.

    Call outside the public runtime's motor loop. The producer must already hold
    the actual consumed input and source-owned activation/recording stamps.
    """
    public = project_observation(raw, preferred_source=raw["source_id"])
    # The disk contract retains only the two safety attestations and action stamp.
    if public["action"] is not None:
        public["action"].update(source_frame_id=public["source_frame_id"],
                                source_captured_ns=public["source_captured_ns"])
    data = json.dumps(public, separators=(",", ":")).encode()
    if len(data) > MAX_BYTES:
        raise ValueError("observation budget exceeded")
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staged.write_bytes(data)
    staged.replace(path)


def read_observation(*, preferred_source: str = "association-brain", stream_id: str = "",
                     path: Path = OBSERVATION_FILE) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("observation budget exceeded")
        return project_observation(json.loads(data), preferred_source=preferred_source,
                                   stream_id=stream_id)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, AssertionError, RecursionError,
            Image.DecompressionBombError):
        return {"schema": SCHEMA, "online": False, "reason": "source_unavailable"}
