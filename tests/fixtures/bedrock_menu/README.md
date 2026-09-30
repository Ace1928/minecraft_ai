# Retained Bedrock menu frame

`play_lan_1920x1080.jpg` is the unmodified JPEG exported through the operator
dashboard's browser-visible game frame on 27 September 2026. It shows the
existing BedrockConnect LAN entry in Bedrock 1.26.51.1 at 1920 × 1080.

SHA-256: `0645b1407c4449ed7b0dc4cdfcb74d0637d216b8e10ee2dd39ee9a5b6a785fef`.

The exact-frame test runs local Tesseract over this file and checks the menu
stage and target geometry. Companion refusal tests remove independent OCR
anchors or request an unapproved target. All test clicks use an in-memory
recorder; the tests never capture a live display or send game input. The OCR
acceptance tests skip when Tesseract is unavailable.

`play_lan_server_list_caption_1920x1080.png` is the unmodified frame 011 from
the 30 September 2026 pack-1.3.147 reconnect attempt002, on Bedrock 1.26.52.3.
The LAN entry caption says "Join To Open Server List". This is the Worlds
browser, not the ServerList modal; selection must remain on the bounded LAN
entry path until a real server-form rectangle is observed. Its regression runs
the real local OCR reader and retains the older full-HD LAN target check.
SHA-256: `f2937947fc51f40db34ad3fa4c74ac4dac04f06a86a27faa3765b4e45ccfc510`.

`server_list_transfer_1920x1080.png` is the exact unmodified frame 012 from the
30 September 2026 family reconnect attempt002 on Bedrock 1.26.52.3. Its
translucent ServerList pane is shifted right over a complete survival HUD.
Full-frame OCR missed the title/captions; the regression must block playable
HUD classification and all clicks rather than manufacture menu observations.
SHA-256: `87c3bf9fcadeb2624b58fc417b9ae3476bedf34878d683136706e81ad2678bef`.

Derived synthetic position and destructive geometry controls are labelled as
tests. They are not retained gameplay, camera calibration, or live authority.

`clear_sky_world_1920x1080.png` is an unmodified captured family-world frame
from the failed pack-1.3.148 stability interval on 30 September 2026, Bedrock
1.26.52.3, classic four-pixel HUD. The earlier brightness-only top-bar guard
mistook its blue sky for a menu: 92.7003% of sampled top-band pixels exceeded
the luma threshold, but none met the independently measured neutral chrome
palette. The regression keeps the existing 90% coverage and luma threshold,
checks the same HUD and 15/20 health glyph readers, and retains real Content
Log and Play toolbar refusals. This is a safety regression, not training data,
camera qualification, current input authority, or proof of gameplay mastery.
SHA-256: `f679c25c93f7abaf73e42829445815e8cce10ca54145b2d1a3c7698f1ae185de`.
