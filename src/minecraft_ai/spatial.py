from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PlaceKind(StrEnum):
    BASE = "base"
    SHELTER = "shelter"
    VILLAGE = "village"
    ORE_VEIN = "ore_vein"
    LANDMARK = "landmark"
    PORTAL = "portal"
    CHEST = "chest"
    FARM = "farm"
    MINE = "mine"
    DANGER_ZONE = "danger_zone"
    DEATH_SPOT = "death_spot"
    CRAFTING_SITE = "crafting_site"
    WAYPOINT = "waypoint"


class PlaceRecord(BaseModel):
    """Spatial and episodic record of a specific location in the Minecraft world (PEM inspired)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    place_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    kind: PlaceKind
    x: float
    y: float
    z: float
    dimension: str = "overworld"
    biome: str | None = None
    discovered_ns: int
    last_visited_ns: int
    visit_count: int = Field(default=1, ge=1)
    notes: str = ""
    resource_types: tuple[str, ...] = ()
    safety_rating: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    def distance_to(self, x: float, y: float, z: float) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2 + (self.z - z) ** 2)

    def horizontal_distance_to(self, x: float, z: float) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.z - z) ** 2)


@dataclass
class SpatialPlaceMemory:
    """Fast spatial index and recommendation engine for places in the world."""

    places: dict[str, PlaceRecord] = field(default_factory=dict)

    def upsert(self, place: PlaceRecord) -> PlaceRecord:
        existing = self.places.get(place.place_id)
        if existing is not None:
            updated_visits = max(existing.visit_count + 1, place.visit_count)
            updated_last = max(existing.last_visited_ns, place.last_visited_ns)
            place = place.model_copy(
                update={"visit_count": updated_visits, "last_visited_ns": updated_last}
            )
        self.places[place.place_id] = place
        return place

    def remove(self, place_id: str) -> None:
        self.places.pop(place_id, None)

    def get(self, place_id: str) -> PlaceRecord | None:
        return self.places.get(place_id)

    def find_nearest(
        self,
        x: float,
        y: float,
        z: float,
        *,
        kind: PlaceKind | None = None,
        resource: str | None = None,
        dimension: str = "overworld",
        max_distance: float | None = None,
    ) -> PlaceRecord | None:
        best: PlaceRecord | None = None
        best_dist = float("inf")

        for record in self.places.values():
            if record.dimension != dimension:
                continue
            if kind is not None and record.kind != kind:
                continue
            if resource is not None and resource not in record.resource_types:
                continue

            dist = record.distance_to(x, y, z)
            if max_distance is not None and dist > max_distance:
                continue

            if dist < best_dist:
                best_dist = dist
                best = record

        return best

    def find_in_radius(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        *,
        dimension: str = "overworld",
        kind: PlaceKind | None = None,
    ) -> list[PlaceRecord]:
        results: list[tuple[float, PlaceRecord]] = []
        for record in self.places.values():
            if record.dimension != dimension:
                continue
            if kind is not None and record.kind != kind:
                continue
            dist = record.distance_to(x, y, z)
            if dist <= radius:
                results.append((dist, record))
        results.sort(key=lambda item: item[0])
        return [record for _, record in results]

    def query_by_resource(self, resource: str, dimension: str = "overworld") -> list[PlaceRecord]:
        matches = [
            record
            for record in self.places.values()
            if record.dimension == dimension and resource in record.resource_types
        ]
        matches.sort(key=lambda record: record.importance, reverse=True)
        return matches

    def recommend_places(
        self,
        current_x: float,
        current_y: float,
        current_z: float,
        *,
        intent: str = "explore",
        dimension: str = "overworld",
        limit: int = 5,
        now_ns: int | None = None,
    ) -> list[tuple[float, PlaceRecord]]:
        """SOTA Spatial Recommendation Engine.

        Calculates utility based on distance penalty, safety rating, importance score,
        and intent matching bonus (e.g. mining -> ORE_VEIN / MINE, shelter -> BASE / SHELTER).
        """
        if limit < 1 or not self.places:
            return []

        now = time.time_ns() if now_ns is None else now_ns
        intent_lower = intent.lower()

        intent_kind_boosts: dict[str, set[PlaceKind]] = {
            "mine": {PlaceKind.ORE_VEIN, PlaceKind.MINE},
            "mining": {PlaceKind.ORE_VEIN, PlaceKind.MINE},
            "shelter": {PlaceKind.BASE, PlaceKind.SHELTER, PlaceKind.CRAFTING_SITE},
            "craft": {PlaceKind.BASE, PlaceKind.CRAFTING_SITE},
            "farm": {PlaceKind.FARM, PlaceKind.VILLAGE},
            "trade": {PlaceKind.VILLAGE},
            "explore": {PlaceKind.LANDMARK, PlaceKind.PORTAL, PlaceKind.WAYPOINT, PlaceKind.VILLAGE},
            "danger": {PlaceKind.DANGER_ZONE, PlaceKind.DEATH_SPOT},
        }

        boosted_kinds = set()
        for key, kinds in intent_kind_boosts.items():
            if key in intent_lower:
                boosted_kinds.update(kinds)

        scored: list[tuple[float, PlaceRecord]] = []
        for record in self.places.values():
            if record.dimension != dimension:
                continue

            dist = record.distance_to(current_x, current_y, current_z)
            distance_score = 1.0 / (1.0 + (dist / 100.0))  # Distance decay

            age_s = max(0.0, (now - record.last_visited_ns) / 1e9)
            recency_score = math.exp(-age_s / (7 * 86_400.0))

            kind_bonus = 2.5 if record.kind in boosted_kinds else 0.0
            safety_score = record.safety_rating

            utility = (
                record.importance * 2.0
                + distance_score * 1.5
                + safety_score * 1.0
                + kind_bonus
                + recency_score * 0.5
            )
            scored.append((utility, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:limit]
