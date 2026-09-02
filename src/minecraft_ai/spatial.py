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


@dataclass(frozen=True)
class WaypointEdge:
    source_id: str
    target_id: str
    distance: float
    safety: float = 1.0
    traversable: bool = True
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass
class TopologicalWaypointGraph:
    """A* graph pathfinding network connecting discovered places across the Minecraft world."""

    edges: dict[str, list[WaypointEdge]] = field(default_factory=dict)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        distance: float,
        *,
        safety: float = 1.0,
        bidirectional: bool = True,
    ) -> None:
        if source_id not in self.edges:
            self.edges[source_id] = []
        self.edges[source_id].append(
            WaypointEdge(source_id=source_id, target_id=target_id, distance=distance, safety=safety)
        )
        if bidirectional:
            if target_id not in self.edges:
                self.edges[target_id] = []
            self.edges[target_id].append(
                WaypointEdge(source_id=target_id, target_id=source_id, distance=distance, safety=safety)
            )

    def find_path(
        self,
        start_id: str,
        goal_id: str,
        memory: SpatialPlaceMemory,
    ) -> list[str]:
        """A* shortest path between two waypoints/places using spatial coordinates."""
        if start_id == goal_id:
            return [start_id]
        if start_id not in self.edges or goal_id not in memory.places:
            return []

        goal_place = memory.get(goal_id)
        if goal_place is None:
            return []

        import heapq

        # Priority queue entries: (f_score, current_id, path)
        open_set: list[tuple[float, str, list[str]]] = [(0.0, start_id, [start_id])]
        visited: set[str] = set()

        while open_set:
            f_score, current_id, path = heapq.heappop(open_set)
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id == goal_id:
                return path

            for edge in self.edges.get(current_id, []):
                if not edge.traversable or edge.target_id in visited:
                    continue
                target_place = memory.get(edge.target_id)
                if target_place is None:
                    continue
                # Heuristic: remaining Euclidean distance to goal place
                h = target_place.distance_to(goal_place.x, goal_place.y, goal_place.z)
                cost = edge.distance / max(0.1, edge.safety)
                g_score = f_score - (0 if len(path) == 1 else memory.get(current_id).distance_to(goal_place.x, goal_place.y, goal_place.z)) + cost
                new_f = g_score + h
                heapq.heappush(open_set, (new_f, edge.target_id, path + [edge.target_id]))

        return []


@dataclass
class DynamicRegionCluster:
    cluster_id: str
    name: str
    primary_kind: PlaceKind
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    center_x: float
    center_y: float
    center_z: float
    place_ids: tuple[str, ...]
    resource_density: dict[str, int]
    safety_average: float


def cluster_places_into_regions(
    memory: SpatialPlaceMemory,
    *,
    cluster_radius: float = 64.0,
    dimension: str = "overworld",
) -> list[DynamicRegionCluster]:
    """Clusters nearby spatial records into named macro-regions (e.g. Base District, Pine Forest Grove, Mining Camp)."""
    places = [p for p in memory.places.values() if p.dimension == dimension]
    if not places:
        return []

    clusters: list[list[PlaceRecord]] = []
    for place in places:
        assigned = False
        for group in clusters:
            # Check distance to group center
            cx = sum(p.x for p in group) / len(group)
            cz = sum(p.z for p in group) / len(group)
            if math.sqrt((place.x - cx) ** 2 + (place.z - cz) ** 2) <= cluster_radius:
                group.append(place)
                assigned = True
                break
        if not assigned:
            clusters.append([place])

    result: list[DynamicRegionCluster] = []
    for idx, group in enumerate(clusters):
        xs = [p.x for p in group]
        ys = [p.y for p in group]
        zs = [p.z for p in group]
        resources: dict[str, int] = {}
        for p in group:
            for r in p.resource_types:
                resources[r] = resources.get(r, 0) + 1
        
        # Primary kind by frequency
        kind_counts: dict[PlaceKind, int] = {}
        for p in group:
            kind_counts[p.kind] = kind_counts.get(p.kind, 0) + 1
        primary_kind = max(kind_counts, key=kind_counts.get) if kind_counts else PlaceKind.LANDMARK

        cluster = DynamicRegionCluster(
            cluster_id=f"region_{dimension}_{idx:03d}",
            name=f"{primary_kind.value.replace('_', ' ').title()} Zone #{idx+1}",
            primary_kind=primary_kind,
            min_x=min(xs),
            max_x=max(xs),
            min_z=min(zs),
            max_z=max(zs),
            center_x=sum(xs) / len(xs),
            center_y=sum(ys) / len(ys),
            center_z=sum(zs) / len(zs),
            place_ids=tuple(p.place_id for p in group),
            resource_density=resources,
            safety_average=sum(p.safety_rating for p in group) / len(group),
        )
        result.append(cluster)

    return result
