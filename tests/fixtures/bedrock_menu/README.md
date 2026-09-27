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
