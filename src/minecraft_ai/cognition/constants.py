"""Bounded prompt and repair limits for high-level cognition."""

_JSON_REPAIR_SYSTEM = (
    "You are a bounded JSON normalizer, not a planner. Repair exactly one malformed or "
    "truncated Minecraft cognition response into the strict wire schema supplied by the API. "
    "Wire keys are r=brief summary, g=goal id, s=skill id or null, p=parameter object, "
    "o=operator reply, c=authorized game chat, x=replan, q=at most two perception questions, "
    "w=research query. Always emit p. Preserve only explicit complete values from "
    "rejected_output and authority_bounds. Never invent observations, inventory, outcomes, "
    "goals, chat, or parameters. s must be null or an allowed skill. Preserve authority_goal_id "
    "and required_action_constraints exactly when present. If the action cannot be recovered "
    "without invention, emit s=null, p containing only required constraints, and x=true. "
    "Return JSON only."
)

_SEMANTIC_REPAIR_SYSTEM = (
    "Make exactly one bounded correction to a rejected Minecraft cognition decision. The user "
    "JSON below is the complete repair context; do not assume or invent omitted world state. "
    "Return only the strict compact wire JSON. Select s only from authority_bounds.allowed_skills "
    "and use only its listed parameter names. Preserve authority_goal_id and every false "
    "required_action_constraint exactly. If no allowed option is justified, return s=null, "
    "p containing the required constraints, x=true, and ask for at most two needed perceptions."
)

_MAX_REJECTED_OUTPUT_CHARS = 2_048
_MAX_REPAIR_REASON_CHARS = 640
_MAX_OPERATOR_FAST_PATH_INSTRUCTION_CHARS = 280
_MAX_HIGH_LEVEL_FACTS = 16
_MAX_HIGH_LEVEL_FACT_TEXT = 120
_MAX_HIGH_LEVEL_GOALS = 4
_MAX_HIGH_LEVEL_PAYLOAD_CHARS = 7_200
_MAX_OPERATOR_PROMPT_TEXT_CHARS = 640
_MOTOR_ONLY_FACT_PREFIXES = ("frame.", "perception.")
_MOTOR_ONLY_FACT_KEYS = frozenset({"scene.observation_dhash"})
_URGENT_FACT_KEYS = frozenset(
    {
        "danger.immediate",
        "environment.underwater",
        "player.critical_health",
        "scene.death",
        "scene.away",
        "scene.playable",
    }
)
