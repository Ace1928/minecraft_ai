from __future__ import annotations

import re
from dataclasses import dataclass

from .knowledge import KnowledgeGraph
from .planning import DependencyPlanner
from .spatial import SpatialPlaceMemory
from .wiki import WikiEvidence, WikiService


@dataclass
class WikiQueryAnswer:
    query: str
    intent: str
    answer_text: str
    recipe_steps: tuple[str, ...] = ()
    spatial_places: tuple[str, ...] = ()
    wiki_sources: tuple[WikiEvidence, ...] = ()


@dataclass
class WikiQueryAgent:
    """Integrated Knowledge Search, Crafting Recipe, and Spatial Lookup Agent."""

    graph: KnowledgeGraph | None = None
    wiki_service: WikiService | None = None
    spatial_memory: SpatialPlaceMemory | None = None

    def query(
        self,
        text: str,
        *,
        current_x: float = 0.0,
        current_y: float = 64.0,
        current_z: float = 0.0,
    ) -> WikiQueryAnswer:
        raw = text.strip()
        lower = raw.lower()

        # Intent 1: Crafting / Recipe / How to get
        if any(
            w in lower for w in ("craft", "make", "recipe", "build", "how do i get", "how to get")
        ):
            return self._handle_recipe_query(raw, lower)

        # Intent 2: Spatial / Where is / Location
        if any(
            w in lower
            for w in ("where", "location", "place", "find ore", "find base", "coordinates")
        ):
            return self._handle_spatial_query(raw, lower, current_x, current_y, current_z)

        # Intent 3: General Wiki / Lore / Entity behavior
        return self._handle_wiki_query(raw)

    def _handle_recipe_query(self, raw: str, lower: str) -> WikiQueryAnswer:
        # Extract target item candidate from query
        item_name = self._extract_item_name(lower)
        recipe_steps: list[str] = []
        answer_parts: list[str] = []

        if self.graph is not None:
            planner = DependencyPlanner(self.graph)
            node_id = f"item:minecraft:{item_name}"
            if node_id in self.graph.nodes:
                options = planner.acquisition_options(node_id)
                if options:
                    best = options[0]
                    answer_parts.append(f"To craft {item_name}: Process requires {best.method}.")
                    for idx, req_group in enumerate(best.requirements, start=1):
                        req_names = [
                            plan.target.replace("item:minecraft:", "") for plan in req_group
                        ]
                        step_desc = f"Step {idx}: Obtain {' OR '.join(req_names)}"
                        recipe_steps.append(step_desc)
                        answer_parts.append(step_desc)

        if not answer_parts:
            # Fallback to wiki service search
            if self.wiki_service is not None and self.graph is not None:
                evidence = self.wiki_service.search(
                    f"{item_name} crafting recipe", self.graph.version
                )
                if evidence:
                    first = evidence[0]
                    answer_parts.append(f"{first.title}: {first.extract[:200]}")
                    return WikiQueryAnswer(
                        query=raw,
                        intent="recipe",
                        answer_text=" ".join(answer_parts),
                        recipe_steps=tuple(recipe_steps),
                        wiki_sources=evidence[:2],
                    )

            answer_parts.append(
                f"To make {item_name}, search your crafting menu or check vanilla recipes."
            )

        return WikiQueryAnswer(
            query=raw,
            intent="recipe",
            answer_text=" ".join(answer_parts),
            recipe_steps=tuple(recipe_steps),
        )

    def _handle_spatial_query(
        self,
        raw: str,
        lower: str,
        current_x: float,
        current_y: float,
        current_z: float,
    ) -> WikiQueryAnswer:
        if self.spatial_memory is None or not self.spatial_memory.places:
            return WikiQueryAnswer(
                query=raw,
                intent="spatial",
                answer_text="I don't have any saved place memories yet.",
            )

        recommendations = self.spatial_memory.recommend_places(
            current_x, current_y, current_z, intent=lower, limit=3
        )

        if not recommendations:
            return WikiQueryAnswer(
                query=raw,
                intent="spatial",
                answer_text="No matching locations recorded in spatial memory.",
            )

        lines: list[str] = []
        place_names: list[str] = []
        for _utility, place in recommendations:
            metric = place.metric_xyz()
            if metric is None:
                line = f"{place.name} ({place.kind.value}); metric pose unavailable"
            else:
                x, y, z = metric
                distance = place.distance_to(current_x, current_y, current_z)
                suffix = "unknown distance" if distance is None else f"{int(distance)}m away"
                line = (
                    f"{place.name} ({place.kind.value}) at "
                    f"X:{int(x)} Y:{int(y)} Z:{int(z)} ({suffix})"
                )
            lines.append(line)
            place_names.append(place.name)

        return WikiQueryAnswer(
            query=raw,
            intent="spatial",
            answer_text=f"Nearest spatial locations: {'; '.join(lines)}",
            spatial_places=tuple(place_names),
        )

    def _handle_wiki_query(self, raw: str) -> WikiQueryAnswer:
        if self.wiki_service is not None and self.graph is not None:
            evidence = self.wiki_service.search(raw, self.graph.version)
            if evidence:
                first = evidence[0]
                return WikiQueryAnswer(
                    query=raw,
                    intent="wiki",
                    answer_text=f"{first.title}: {first.extract[:300]}...",
                    wiki_sources=evidence[:3],
                )

        return WikiQueryAnswer(
            query=raw,
            intent="wiki",
            answer_text=(
                f"Minecraft knowledge query: '{raw}'. "
                "Check the official Minecraft wiki for detailed stats."
            ),
        )

    def _extract_item_name(self, text: str) -> str:
        words = re.findall(r"\b[a-z0-9_]+\b", text)
        skip = {
            "how",
            "do",
            "i",
            "craft",
            "make",
            "build",
            "recipe",
            "get",
            "for",
            "a",
            "an",
            "the",
            "to",
        }
        candidates = [w for w in words if w not in skip]
        if candidates:
            return "_".join(candidates)
        return "stick"
