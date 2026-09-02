from __future__ import annotations

from minecraft_ai.roles import (
    BUILTIN_ROLES,
    RoleProfile,
    get_role,
    list_roles,
    register_custom_role,
)


def test_builtin_roles_include_new_archetypes() -> None:
    assert "wiki_assistant" in BUILTIN_ROLES
    assert "miner" in BUILTIN_ROLES
    assert "companion" in BUILTIN_ROLES

    wiki_role = get_role("wiki_assistant")
    assert wiki_role.role_id == "wiki_assistant"
    assert "lookup_recipes" in wiki_role.standing_goals


def test_register_custom_role() -> None:
    custom = RoleProfile(
        role_id="custom_architect",
        description="Designs and constructs megastructures.",
        standing_goals=("plan_blueprint", "gather_scaffolding", "build_structure"),
        utility_weights={"construction": 1.0, "design": 0.9},
        risk_tolerance=0.3,
    )
    register_custom_role(custom)

    retrieved = get_role("custom_architect")
    assert retrieved.role_id == "custom_architect"
    assert retrieved.description == "Designs and constructs megastructures."

    all_roles = list_roles()
    assert any(role.role_id == "custom_architect" for role in all_roles)
