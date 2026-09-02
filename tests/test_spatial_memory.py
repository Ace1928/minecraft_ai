from __future__ import annotations

import tempfile
import time
from pathlib import Path

from minecraft_ai.spatial import PlaceKind, PlaceRecord, SpatialPlaceMemory
from minecraft_ai.storage import StateDatabase


def test_place_record_distance() -> None:
    place = PlaceRecord(
        place_id="base_1",
        name="Main Base",
        kind=PlaceKind.BASE,
        x=100.0,
        y=64.0,
        z=200.0,
        discovered_ns=1000,
        last_visited_ns=1000,
    )
    assert place.distance_to(100.0, 64.0, 200.0) == 0.0
    assert place.horizontal_distance_to(103.0, 204.0) == 5.0


def test_spatial_place_memory_operations() -> None:
    memory = SpatialPlaceMemory()
    now = time.time_ns()

    p1 = PlaceRecord(
        place_id="iron_mine",
        name="Iron Ore Vein",
        kind=PlaceKind.ORE_VEIN,
        x=50.0,
        y=30.0,
        z=50.0,
        resource_types=("iron_ore", "raw_iron"),
        discovered_ns=now,
        last_visited_ns=now,
        importance=0.8,
    )
    p2 = PlaceRecord(
        place_id="home_shelter",
        name="Home Hut",
        kind=PlaceKind.SHELTER,
        x=10.0,
        y=64.0,
        z=10.0,
        resource_types=("crafting_table", "bed"),
        discovered_ns=now,
        last_visited_ns=now,
        importance=0.9,
    )

    memory.upsert(p1)
    memory.upsert(p2)

    assert len(memory.places) == 2
    assert memory.get("iron_mine") == p1

    nearest = memory.find_nearest(0.0, 64.0, 0.0)
    assert nearest is not None
    assert nearest.place_id == "home_shelter"

    in_radius = memory.find_in_radius(0.0, 64.0, 0.0, radius=100.0)
    assert len(in_radius) == 2

    by_resource = memory.query_by_resource("iron_ore")
    assert len(by_resource) == 1
    assert by_resource[0].place_id == "iron_mine"


def test_spatial_recommendations() -> None:
    memory = SpatialPlaceMemory()
    now = time.time_ns()

    memory.upsert(
        PlaceRecord(
            place_id="ore_site",
            name="Deep Ore Vein",
            kind=PlaceKind.ORE_VEIN,
            x=20.0,
            y=12.0,
            z=20.0,
            discovered_ns=now,
            last_visited_ns=now,
            importance=0.9,
        )
    )
    memory.upsert(
        PlaceRecord(
            place_id="safe_base",
            name="Fortified Base",
            kind=PlaceKind.BASE,
            x=500.0,
            y=64.0,
            z=500.0,
            discovered_ns=now,
            last_visited_ns=now,
            importance=0.95,
        )
    )

    recs = memory.recommend_places(0.0, 64.0, 0.0, intent="mining", limit=2)
    assert len(recs) == 2
    # Ore vein should be top recommendation for mining intent
    assert recs[0][1].kind == PlaceKind.ORE_VEIN


def test_state_database_spatial_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_spatial.db"
        now = time.time_ns()
        place = PlaceRecord(
            place_id="p1",
            name="Village Outpost",
            kind=PlaceKind.VILLAGE,
            x=300.0,
            y=70.0,
            z=-150.0,
            discovered_ns=now,
            last_visited_ns=now,
        )

        with StateDatabase(db_path) as db:
            db.save_place(place)
            loaded = db.load_places()
            assert len(loaded.places) == 1
            assert loaded.get("p1") == place
