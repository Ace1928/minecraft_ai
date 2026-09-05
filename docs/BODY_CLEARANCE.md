# Body-clearance sensing: first executable slice

The desired local representation is the player's collision envelope plus nearby
occupied, free and unknown space. A clear crosshair ray is not evidence that the
whole body fits: feet, overhangs and lateral edges may still intersect the route.

Mojang's [default player definition](https://github.com/Mojang/bedrock-samples/blob/main/behavior_pack/entities/player.json)
uses a standing collision width of 0.6 and height of 1.8. These are a nominal
prior, not a measurement of the current stance or position. Partial blocks need
their collision shapes, not merely a solid/full-block label. A one-block grid
alone cannot resolve edge overlap or slabs.

## Implemented, observation-only

`BodyClearanceSurveyor` asks the existing local VLM for one visible surface:

```json
{"feature":"side_face","point":[0.3,0.6],"confidence":0.9}
```

The finite vocabulary is `underside`, `riser`, `side_face`, and `unknown`/null.
The point must lie inside the nominated visible face. Unknown fields remain
unknown. There is one model query, no format-repair query, and no repeated search
until a convenient answer appears. Invalid, conflicting/duplicate or extra
fields fail closed.

The model sees one aspect-preserved WORLD crop with no collage, header or border.
Evidence retains the exact original RGB crop hash and frame identity. The code
transforms the image-local point into full-frame coordinates; it does not invent
a bounding box, depth, block ID, collider intersection or empty voxel.

This result is deliberately **not** a general object track or `target.*` fact:
the motor router can implicitly select ordinary tracks. The survey has no
blackboard writes, input access, camera control or action permission. It is not
yet enabled in the live recovery path.

Run one retained-image qualification when the configured local model is idle:

```sh
.venv/bin/python scripts/probe_body_clearance.py retained-view.png --output survey.json
```

The tool requires the already-running single-slot server, refuses busy or
unconfirmed availability, never starts a model or game, and will not overwrite
an evidence record. Availability is advisory; the server's single slot serializes
a racing live request. Do not run batches against the live model. Offline frame
IDs/timestamps are explicitly synthetic and never published as live observations.

## What this does not establish

- An underside is not automatically a head collision; a distant ceiling is a
  negative control. A riser must not be confused with the floor supporting us.
- Image-left/right are not measured body-width boundaries. Their projection
  depends on depth, field of view and camera pose.
- Accepted mouse deltas are dead reckoning, not observed absolute orientation.
  The supervisor's homing flag records completed input, not visual proof of the
  horizon. Never reconstruct metric geometry solely from that flag.
- A camera-only turn provides no translational depth baseline. A useful 4×4×4
  local map still requires grounded surface depth and uncertain relative pose;
  unseen cells must remain unknown, not air.
- Pixel change, jump bob and an accepted forward key are not metric translation.
  Controller starvation is not evidence of a physical collision.

## Next gameplay qualification

Keep the existing one-shot idle/stall observation eligibility unchanged. Before
promoting new survey targeting into recovery, qualify overhead, footstep and
mirrored lateral-edge layouts, plus open, high-ceiling and occluded controls.
Construction labels belong to evaluation, not the agent's observations.

Inspect with bounded camera-only upper/forward/lower views while retaining the
original travel anchor. Those angles are search views, not collision heights.
Select one independently reviewed nearby candidate, recenter, then obtain a
**fresh** exact-crosshair block identity. Preserve existing material, safety,
operator, scene ownership and break-verification guards. Supporting-floor or
distant/ambiguous candidates must not be cleared.

The useful gate is one causally attributable clearance and retry along the
original route, with persistent landmark change after settling. Correct surface
nomination alone does not prove escape, item acquisition, or a working 3D map.
