# Named Minecraft decisions through the shared World

This is a candidate integration. It is not enabled in the reviewed Minecraft
9f73bb9 runtime capsule or deployed World007. Those releases remain isolated
from development source until a separate release qualification succeeds.

`HighLevelController` passes the original frozen decision bounds to
`NativeWorldCognitionModel.complete_minecraft_decision`. The capsule preserves
the selected goal IDs, exact current operator goal, feasible skill and parameter
names, requested skill restrictions and required false action limits. Lossy
prompt compaction can remove explanatory context; it cannot change this capsule.
An operator question's mode comes from queue metadata rather than prompt prose.

The adapter forwards the named `erais.minecraft.cognition.v1` format through the
existing `/v1/tokenize` and `/v1/chat/completions` endpoints. Before generation it
checks private readiness against the same World runtime and compiler identity.
The token-budget and completion receipts must identify that runtime, the same
tokenizer, grammar and original authority hashes. Unknown schemas, caller
grammars, unavailable compiler support, malformed receipts and incomplete
output refuse. There is no silent plain-text or alternate-provider fallback.

The World owner generates the fixed compact decision fields, or `{g,o,q}` for an
operator reply, using a request-local matcher over its existing native decoder.
A complete allowed EOS and strict final validation are both necessary. A token
cap, deadline, cancellation or queue yield discards structured output. The
minimum-output example is only a pre-generation budget witness; it must never
be returned in place of a learned decision. Parser disagreement after named
validation does not trigger another inference or unconstrained repair.

The existing snapshot, model-attempt origin, priority, skill precondition,
freshness, game-chat authority and final publication checks remain necessary.
`q` is a bounded request for the existing perception vocabulary, not an observed
fact or a training label. Operator replies cannot request target directions or
grant gameplay action authority. Recipe delivery still requires an actual fresh
incoming player question and the current installed pack catalog.

Offline tests exercise model-free transport and control flow. They establish
contract admission and refusal, not useful strategic reasoning, task quality,
latency superiority, gameplay mastery or actual family chat delivery. Learned
decision comparison and a bounded live acceptance trace remain separate gates.
