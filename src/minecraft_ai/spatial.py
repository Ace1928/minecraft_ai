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


class PoseSource(StrEnum):
    VISUAL_ODOMETRY = "visual_odometry"
    DEAD_RECKONING = "dead_reckoning"
    LANDMARK_ALIGNMENT = "landmark_alignment"
    PERMITTED_DEBUG = "permitted_debug"
    UNKNOWN = "unknown"


class PoseBelief(BaseModel):
    """Uncertain player-relative pose; exact metric coordinates are optional."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_xyz: tuple[float, float, float] | None = None
    covariance_xyz: tuple[float, float, float] | None = None
    heading_degrees: float | None = Field(default=None, ge=-360.0, le=360.0)
    dimension: str = "overworld"
    source: PoseSource = PoseSource.UNKNOWN


class PlaceRecord(BaseModel):
    """Spatial and episodic place record with optional metric localization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    place_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    kind: PlaceKind
    pose_belief: PoseBelief = Field(default_factory=PoseBelief)
    # Legacy/debug coordinates remain readable during migration, but are optional.
    x: float | None = None
    y: float | None = None
    z: float | None = None
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

    def metric_xyz(self) -> tuple[float, float, float] | None:
        if self.pose_belief.metric_xyz is not None:
            return self.pose_belief.metric_xyz
        if self.x is None or self.y is None or self.z is None:
            return None
        return self.x, self.y, self.z

    def distance_to(self, x: float, y: float, z: float) -> float | None:
        metric = self.metric_xyz()
        if metric is None:
            return None
        own_x, own_y, own_z = metric
        return math.sqrt((own_x - x) ** 2 + (own_y - y) ** 2 + (own_z - z) ** 2)

    def horizontal_distance_to(self, x: float, z: float) -> float | None:
        metric = self.metric_xyz()
        if metric is None:
            return None
        own_x, _own_y, own_z = metric
        return math.sqrt((own_x - x) ** 2 + (own_z - z) ** 2)


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
            if dist is None:
                continue
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
            if dist is None:
                continue
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
        current_x: float | None,
        current_y: float | None,
        current_z: float | None,
        *,
        intent: str = "explore",
        dimension: str = "overworld",
        limit: int = 5,
        now_ns: int | None = None,
    ) -> list[tuple[float, PlaceRecord]]:
        """Rank known places without requiring privileged coordinates."""
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
            "explore": {
                PlaceKind.LANDMARK,
                PlaceKind.PORTAL,
                PlaceKind.WAYPOINT,
                PlaceKind.VILLAGE,
            },
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

            dist = (
                record.distance_to(current_x, current_y, current_z)
                if current_x is not None and current_y is not None and current_z is not None
                else None
            )
            distance_score = 0.5 if dist is None else 1.0 / (1.0 + (dist / 100.0))

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
    """Route graph whose correctness does not depend on exact XYZ coordinates."""

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
            WaypointEdge(
                source_id=source_id,
                target_id=target_id,
                distance=distance,
                safety=safety,
            )
        )
        if bidirectional:
            if target_id not in self.edges:
                self.edges[target_id] = []
            self.edges[target_id].append(
                WaypointEdge(
                    source_id=target_id,
                    target_id=source_id,
                    distance=distance,
                    safety=safety,
                )
            )

    def find_path(
        self,
        start_id: str,
        goal_id: str,
        memory: SpatialPlaceMemory,
    ) -> list[str]:
        """Find a minimum-risk route, using metric pose only as an optional heuristic."""
        if start_id == goal_id:
            return [start_id]
        if start_id not in self.edges:
            return []

        goal_place = memory.get(goal_id)

        import heapq

        def heuristic(place_id: str) -> float:
            place = memory.get(place_id)
            if place is None or goal_place is None:
                return 0.0
            goal_metric = goal_place.metric_xyz()
            if goal_metric is None:
                return 0.0
            return place.distance_to(*goal_metric) or 0.0

        open_set: list[tuple[float, float, str, list[str]]] = [
            (heuristic(start_id), 0.0, start_id, [start_id])
        ]
        best_cost: dict[str, float] = {start_id: 0.0}

        while open_set:
            _estimated_total, cost_so_far, current_id, path = heapq.heappop(open_set)
            if cost_so_far > best_cost.get(current_id, float("inf")):
                continue

            if current_id == goal_id:
                return path

            for edge in self.edges.get(current_id, []):
                if not edge.traversable:
                    continue
                next_cost = cost_so_far + edge.distance / max(0.1, edge.safety)
                if next_cost >= best_cost.get(edge.target_id, float("inf")):
                    continue
                best_cost[edge.target_id] = next_cost
                heapq.heappush(
                    open_set,
                    (
                        next_cost + heuristic(edge.target_id),
                        next_cost,
                        edge.target_id,
                        path + [edge.target_id],
                    ),
                )

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
    """Cluster metric-localized records into named macro-regions."""
    places = [
        place
        for place in memory.places.values()
        if place.dimension == dimension and place.metric_xyz() is not None
    ]
    if not places:
        return []

    clusters: list[list[PlaceRecord]] = []
    for place in places:
        assigned = False
        for group in clusters:
            # Check distance to group center
            coordinates = [item.metric_xyz() for item in group]
            cx = sum(item[0] for item in coordinates if item is not None) / len(group)
            cz = sum(item[2] for item in coordinates if item is not None) / len(group)
            metric = place.metric_xyz()
            assert metric is not None
            if math.sqrt((metric[0] - cx) ** 2 + (metric[2] - cz) ** 2) <= cluster_radius:
                group.append(place)
                assigned = True
                break
        if not assigned:
            clusters.append([place])

    result: list[DynamicRegionCluster] = []
    for idx, group in enumerate(clusters):
        metrics = [place.metric_xyz() for place in group]
        xs = [metric[0] for metric in metrics if metric is not None]
        ys = [metric[1] for metric in metrics if metric is not None]
        zs = [metric[2] for metric in metrics if metric is not None]
        resources: dict[str, int] = {}
        for p in group:
            for r in p.resource_types:
                resources[r] = resources.get(r, 0) + 1

        # Primary kind by frequency
        kind_counts: dict[PlaceKind, int] = {}
        for p in group:
            kind_counts[p.kind] = kind_counts.get(p.kind, 0) + 1
        primary_kind = (
            max(kind_counts, key=lambda kind: kind_counts[kind])
            if kind_counts
            else PlaceKind.LANDMARK
        )

        cluster = DynamicRegionCluster(
            cluster_id=f"region_{dimension}_{idx:03d}",
            name=f"{primary_kind.value.replace('_', ' ').title()} Zone #{idx + 1}",
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
