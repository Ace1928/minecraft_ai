# Operator system topology and model viewer

The operator dashboard (`/`) now renders the live system as a part graph and
each bound model as an interactive 3D population. Both views are built from
the same allowlisted payloads the dashboard already reads; nothing here
captures, infers or controls the game.

## Data contract

`GET /api/topology` (same source/stream preference rules as
`/api/observation`) returns `minecraft.topology.v1`:

- `parts`: the ten declared parts — frame capture, ROCKET-2 perception, motor
  policy, skill execution, Bedrock game, supervisor, cognition agent, native
  policy model, association brain, memory & trajectory — each with `state`
  (`live`/`idle`/`offline`), normalised `activity` and a short `detail`.
- `edges`: the dataflow between those parts with a `flow` value that animates
  in the viewer.
- `populations`: the observed `models[*].population` payloads (positions,
  layer sizes, bounded edges, sparse activity, identity) for models that are
  available and bound.
- `observation`: source id, sequence, frame age, action kind/buttons and
  receptive scope for the freshness ribbon.

## Guarantees

- Read-only: the route is GET-only, host-validated and preference-validated;
  it never captures a frame, runs inference or actuates.
- No inference: part activity comes from explicit status/telemetry fields and
  the model payload's own `source_active_kcs`/`calls`; pixels and activations
  are never derived from unrelated counters.
- Bounded: populations are capped by the observation budget (240 KB), edges
  are sampled for drawing only, and an unavailable/expired observation renders
  an honest waiting state.

## Verification

`tests/test_operator_topology.py` covers the offline map, a live packet with a
bound population, the dashboard markers and the route's GET-only validation.
For visual review, publish a synthetic observation and open the dashboard: the
part graph shows animated flows and the model canvas renders the point cloud
with layer colours, activity glow and drag-to-orbit.
