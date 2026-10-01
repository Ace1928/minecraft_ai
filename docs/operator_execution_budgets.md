# Bounded operator attempts

The optional `execution_budget` on the existing local message endpoint bounds one request's complete ordinary skill chain. For example:

```json
{
  "text": "explore_forward",
  "kind": "correction",
  "execution_budget": {"timeout_ms": 12000, "max_skills": 2}
}
```

The time limit starts when the local endpoint creates the durable request, so queue and cognition time consume it. The runtime admits one monotonic deadline and one skill-start counter. Initial actions, ordinary recoveries, continuation and plan children share these bounds. A child's own timeout cannot extend the request. Changing the same request's content cannot refresh its owner; use a new request ID through the endpoint. Unbounded legacy requests retain their existing behavior.

The runtime checks before policy work, after a slow policy tick and again at the motor dispatch boundary. An exhausted request releases inputs through the existing supervisor acknowledgement protocol, removes its continuation and suppresses further ordinary recovery. A failed release keeps the existing pending-release gate closed. Verified scene/death safety recovery retains independent authority; request expiration does not stop the game or retire its client. The limit constrains runtime admission and dispatch, not a claim of instantaneous physical release: capture/transport delay and actual release acknowledgement still need operational measurement.

The deadline/count owner is local to the current runtime. An acknowledged bounded request cannot rebuild that owner after a restart and is refused rather than receiving a fresh count. This is refusal on lost continuation, not a persistent execution migration claim. The source controls use mocked clocks, policies and supervisor responses. This change is not installed in the existing live capsule, and does not qualify exploration, Pokémon mastery or World planner quality.
