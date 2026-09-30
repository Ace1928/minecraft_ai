# Retained Bedrock health controls

These are unmodified captured game frames from the existing private Bedrock
1.26.52.3 family session on 2026-09-30, after the Pokémon 1.3.146 pack upgrade.
The capture is 1920x1080, classic HUD, four pixels per heart sprite cell.
Retaining these frames does not qualify a camera profile, a learned model,
training labels, another Bedrock build or a different HUD/resource pack.

| File | Independently reviewed visible state | SHA-256 of original file bytes |
| --- | --- | --- |
| `full_health_1920x1080.png` | Ten full normal hearts after respawn: 20 health units | `8d04311268d7d1d132e98c10d8feb31a1e7a7241fe4827a0ba954a63172b5191` |
| `low_health_1920x1080.jpg` | Two full normal hearts and one half heart: 5 health units | `2b38959564549be5832f1ff40128f0a40a2af87968affa3a009e5f94ce4afcc5` |
| `low_health_jitter_1920x1080.jpg` | One full and one half heart visually; first and seventh glyphs are raised four pixels by low-health animation. Required decoder abstention. | `05d5a80c3aa63c3acb0bd9e1c8827c979536613b4fc2a05d8d57c99d74141037` |
| `death_1920x1080.png` | Actual death overlay. Required decoder abstention. | `bd0fcac51ccb24b51cb50039bdf12844cd39e1dcb26cecf9b2836b4ad1bf9ad8` |

The positive PNG originated from
`world007-handover/respawn-1.3.146-001/003-in-world.png` (frame 3,
capture monotonic time 2219273363554598). The JPEG controls originated from
`world007-handover/oak-reference.jpg` and `escape-reference.jpg`. The death
control originated from `respawn-1.3.146-001/001-death.png`.
No game input or service changes were performed while collecting these test
fixtures from the already retained captures.

Tests also construct explicitly synthetic negative controls and a 0..20-unit
bank by rearranging the captured full/half/empty glyph crops. Those derived
images test decoder behaviour; they are not live observations, learning
trajectories or evidence of gameplay ability. Effect-colour and flash controls
are deliberately unsupported mutations, not claims that an actual status
effect was observed in the family session.
