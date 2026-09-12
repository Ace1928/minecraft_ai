"""Version-scoped mining evidence. Absence of a pickup is not a tool verdict.

This module does no game IO and emits no input. Optional, immutable rule snapshots
must come from an operator-owned game/pack adapter. Pixel-only deployments learn
observed attempts; they never present model guesses as authoritative game rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from minecraft_ai.memory.store import MemoryKind, MemoryRecord, MemoryStore


class MiningKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    ruleset: str = Field(min_length=1, max_length=512)
    block: str = Field(min_length=1, max_length=256)
    tool: str = Field(min_length=1, max_length=256)

    @field_validator("block", "tool")
    @classmethod
    def canonical_identity(cls, value: str) -> str:
        value = value.strip().casefold().removeprefix("minecraft:")
        if not value:
            raise ValueError("empty canonical identity")
        return value

    @property
    def token(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class MiningRule(BaseModel):
    """Exact resolved rule for one block state and equipped-tool state.

    A missing field means unknown, not false. The adapter resolves tags,
    enchantments, effects, adventure permissions, and pack/version semantics.
    No expression from a pack is evaluated by this library.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    block: str = Field(min_length=1, max_length=256)
    tool: str = Field(min_length=1, max_length=256)

    @field_validator("block", "tool")
    @classmethod
    def canonical_identity(cls, value: str) -> str:
        value = value.strip().casefold().removeprefix("minecraft:")
        if not value:
            raise ValueError("empty canonical identity")
        return value

    can_break: bool | None = None
    can_harvest: bool | None = None
    expected_ms: float | None = Field(default=None, gt=0, le=3_600_000, allow_inf_nan=False)


class MiningRuleSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1] = 1
    ruleset_id: str = Field(min_length=1, max_length=512)
    provenance: str = Field(min_length=1, max_length=1024)
    rules: tuple[MiningRule, ...] = Field(default=(), max_length=100_000)

    @model_validator(mode="after")
    def unique_rules(self) -> MiningRuleSnapshot:
        keys = {(rule.block, rule.tool) for rule in self.rules}
        if len(keys) != len(self.rules):
            raise ValueError("duplicate block/tool rule")
        return self

    @classmethod
    def load(cls, path: Path) -> MiningRuleSnapshot:
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("mining rule snapshot exceeds 16 MiB")
        return cls.model_validate_json(path.read_bytes())

    @property
    def scope(self) -> str:
        # Changing the actual rule data cannot silently reuse an old model,
        # even when a human forgets to update a descriptive ruleset label.
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class MiningTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    attempt_id: str = Field(min_length=1, max_length=256)
    key: MiningKey
    parent_attempt_id: str | None = Field(default=None, min_length=1, max_length=256)
    broke: bool | None = None
    harvested: bool | None = None
    picked_up: bool | None = None
    elapsed_ms: float = Field(gt=0, le=3_600_000, allow_inf_nan=False)
    evidence: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def consistent(self) -> MiningTrial:
        if self.parent_attempt_id is not None and self.broke is not None:
            raise ValueError("collection evidence cannot recount a block break")
        if self.harvested is True and self.broke is not True and self.parent_attempt_id is None:
            raise ValueError("verified harvest requires a verified break")
        if self.picked_up is True and self.harvested is not True:
            raise ValueError("a causally joined pickup requires verified harvest")
        return self

    @property
    def memory_id(self) -> str:
        identity = json.dumps([self.key.ruleset, self.attempt_id], separators=(",", ":"))
        return "mining-trial:" + hashlib.sha256(identity.encode()).hexdigest()


@dataclass
class MiningBelief:
    breaks: int = 0
    nonbreaks: int = 0
    censored: int = 0
    harvests: int = 0
    nonharvests: int = 0
    pickups: int = 0
    uncollected: int = 0
    log_mean: float = 0.0
    log_m2: float = 0.0

    def observe(self, trial: MiningTrial) -> None:
        if trial.broke is True:
            self.breaks += 1
            value = math.log(trial.elapsed_ms)
            delta = value - self.log_mean
            self.log_mean += delta / self.breaks
            self.log_m2 += delta * (value - self.log_mean)
        elif trial.broke is False:
            self.nonbreaks += 1
        elif trial.parent_attempt_id is None:
            self.censored += 1
        self.harvests += int(trial.harvested is True)
        self.nonharvests += int(trial.harvested is False)
        self.pickups += int(trial.picked_up is True)
        self.uncollected += int(trial.picked_up is False)

    def budget_ms(self, fallback_ms: int, *, cap_ms: int) -> int:
        """Upper log-duration prediction from completed breaks, not timeouts."""
        if self.breaks < 3:
            return min(fallback_ms, cap_ms)
        variance = max(0.0, self.log_m2 / (self.breaks - 1))
        upper = math.exp(min(math.log(cap_ms), self.log_mean + 2 * math.sqrt(variance)))
        return min(cap_ms, max(fallback_ms, math.ceil(upper * 1.25)))


@dataclass
class MiningKnowledge:
    memories: MemoryStore
    scope: str
    snapshot: MiningRuleSnapshot | None = None
    _beliefs: dict[MiningKey, MiningBelief] = field(default_factory=dict, init=False)
    _rules: dict[tuple[str, str], MiningRule] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.snapshot is not None:
            self.scope = self.snapshot.scope
            self._rules = {(r.block, r.tool): r for r in self.snapshot.rules}
        # Validate the scope even when no attempts have been recorded.
        MiningKey(ruleset=self.scope, block="unknown", tool="unknown")
        trials: dict[str, MiningTrial] = {}
        for record in self.memories.records.values():
            if record.source != "runtime:mining-trial-v1":
                continue
            payload = record.metadata.get("trial")
            if not isinstance(payload, str):
                continue
            try:
                trial = MiningTrial.model_validate_json(payload)
            except ValueError:
                continue
            if trial.key.ruleset == self.scope and record.memory_id == trial.memory_id:
                trials[trial.attempt_id] = trial
        for trial in trials.values():
            if trial.parent_attempt_id is not None:
                parent = trials.get(trial.parent_attempt_id)
                if (
                    parent is None
                    or parent.key != trial.key
                    or parent.broke is not True
                    or parent.parent_attempt_id is not None
                ):
                    continue
            self._beliefs.setdefault(trial.key, MiningBelief()).observe(trial)

    def key(self, block: str, tool: str) -> MiningKey:
        return MiningKey(ruleset=self.scope, block=block, tool=tool)

    def rule(self, block: str, tool: str) -> MiningRule | None:
        key = self.key(block, tool)
        return self._rules.get((key.block, key.tool))

    def belief(self, block: str, tool: str) -> MiningBelief:
        # Return a copy so callers cannot fabricate learned evidence in place.
        from dataclasses import replace

        return replace(self._beliefs.get(self.key(block, tool), MiningBelief()))

    def trial(self, attempt_id: str) -> MiningTrial | None:
        identity = json.dumps([self.scope, attempt_id], separators=(",", ":"))
        memory_id = "mining-trial:" + hashlib.sha256(identity.encode()).hexdigest()
        record = self.memories.records.get(memory_id)
        if record is None or record.source != "runtime:mining-trial-v1":
            return None
        payload = record.metadata.get("trial")
        if not isinstance(payload, str):
            return None
        try:
            trial = MiningTrial.model_validate_json(payload)
            if trial.memory_id == record.memory_id and trial.key.ruleset == self.scope:
                return trial
            return None
        except ValueError:
            return None

    def record(self, trial: MiningTrial) -> MemoryRecord | None:
        if trial.key.ruleset != self.scope:
            raise ValueError("trial belongs to another ruleset")
        if trial.parent_attempt_id is not None:
            parent = self.trial(trial.parent_attempt_id)
            if (
                parent is None
                or parent.key != trial.key
                or parent.broke is not True
                or parent.parent_attempt_id is not None
            ):
                raise ValueError("collection evidence lacks its exact verified parent break")
        previous = self.memories.records.get(trial.memory_id)
        if previous is not None:
            if previous.metadata.get("trial") != trial.model_dump_json():
                raise ValueError("conflicting duplicate mining trial")
            return None
        now = time.time_ns()
        record = MemoryRecord(
            memory_id=trial.memory_id,
            kind=MemoryKind.PROCEDURAL,
            text=(
                f"Mining {trial.key.block} with {trial.key.tool}: "
                f"break={trial.broke}, harvest={trial.harvested}, pickup={trial.picked_up}. "
                "Unknown or absent pickup does not establish an unsuitable tool."
            ),
            created_ns=now,
            updated_ns=now,
            importance=0.4,
            source="runtime:mining-trial-v1",
            entity_tags=(trial.key.block, trial.key.tool),
            metadata={"trial": trial.model_dump_json(), "ruleset": self.scope},
        )
        self.memories.upsert(record)
        self._beliefs.setdefault(trial.key, MiningBelief()).observe(trial)
        return record

    def budget_ms(self, block: str, tool: str, fallback_ms: int, *, cap_ms: int) -> int:
        if any(type(v) is not int or v < 1 for v in (fallback_ms, cap_ms)):
            raise ValueError("duration budgets must be positive")
        rule = self.rule(block, tool)
        if rule is not None and rule.expected_ms is not None:
            fallback_ms = math.ceil(rule.expected_ms * 1.25)
        return self.belief(block, tool).budget_ms(fallback_ms, cap_ms=cap_ms)
