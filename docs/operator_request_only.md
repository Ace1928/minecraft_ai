# One bounded camera survey

`operator_request_only: {skill_id: survey_surroundings}` is an optional, prospective
runtime fence. Its absence preserves ordinary gameplay. It currently supports
one literal camera survey, not arbitrary commands, natural-language reasoning,
inventory directions or general autonomous play.

The runtime captures fresh pixels and renews its existing input lease while it
waits. It does not start startup cognition, active VLM work or policy warmup.
It does not dispatch inventory directions, game chat, autonomous camera moves,
keepalive skills or model-driven planning. A 60-second wait deadline defaults
to a terminal held state; it can be configured between 50 and 180,000ms. An
expired wait never enables autonomy.

Exactly one pending message must be a fresh, queued ordinary instruction or
correction, created after this runtime was constructed. Its text must be the
literal `survey_surroundings`. It must carry `execution_budget` with at most
12,000ms and exactly one skill. Previously delivered/acknowledged, ambiguous,
paid, free-form and expired requests cannot authorize the survey. The existing
operator endpoint creates the message ID and creation timestamp; the fence
does not invent a request or restore a consumed owner after a restart.

The skill runs through the existing executor and installed policy. All seven
skill press permissions must remain false. The final motor boundary verifies
the same request, immutable skill, operator revision and deadline while holding
the existing operator and database admission boundaries. It refuses key/button
presses and cursor actions even if a policy or another caller proposes them.
Mandatory physical releases remain independent. Neither a refused action nor
an unacknowledged release grants accepted-action learning credit.

A verified unsafe/modal/death scene aborts the camera-only window and releases
inputs. It does not authorize an additional respawn, inventory click or recovery
skill. Ordinary recovery remains available outside this optional window after
its owner is retired. At completion, failure or deadline the runtime stays
held. A later queued message cannot restart this owner or turn ordinary behavior
back on. Changing the mode requires an explicit, separately reviewed lifecycle.

This source change is not installed or activated. The existing operator pause
must remain until the user resolves it. The prior activation proposal's seven
false permissions covered only the survey skill; it did not cover startup or
post-survey ordinary behavior. That old proposal cannot establish the complete
camera-only window.

The fence suppresses strategic/VLM requests and policy inference outside the
admitted goal; it does not remove process/factory construction, checkpoint load,
fast deterministic capture perception, telemetry, or closed learning-state
persistence. Those owners and their full costs require separate measurement.
Capture/policy/IPC may block cooperatively; the request clock suppresses late
returned actions but is not a measured physical release deadline. An externally
supervised process/lifecycle bound and actual release receipts remain necessary.
The current family association learner is still local; enabling this fence does
not claim that its computation has migrated into the shared World owner.

The recorded controls use the actual runtime startup/tick/executor/final motor path with fake clocks, fixture pixels, a fake actuator and temporary SQLite. They cover unwanted startup and terminal work, masks, authority changes, slow returns, release acknowledgement and unsafe-scene abort. Their bounded results are recorded below. They cannot establish camera calibration, live capture, gameplay progress or mastery.

## Recorded engineering qualification

The retained 2026-10-01 check passed all 109 controls: 35 new request-only
controls and 74 existing budget/startup/retirement controls. They exercised the
actual runtime and executor with simulated sensors/policies/actuation and a
temporary real SQLite database. The source snapshot contained 199 pinned files;
111 loaded Minecraft module origins were admitted and all source bytes still
matched afterward. No numerical imports, forbidden boundary attempts or leaked
threads were observed. This is source-level engineering evidence, not a live
camera, Minecraft mastery, model-quality or serving-performance qualification.

Two earlier attempts remain recorded: pytest setup was refused before collection
because its default logging target lay outside the private test directory; the
next attempt passed 98 controls and failed nine on the executor's terminal-reason
contract. The source fix preserves early input release and retries an
unacknowledged release in `finally` even if expiry bookkeeping raises.

Worker wall time for the passing attempt was 5.012 seconds, CPU 2.282 seconds and
process high-water memory 67,682,304 bytes. The manager reported a separate
61,034,496-byte peak and zero swap; complete supervision and owned cleanup took
5.563 seconds. These scopes must not be combined into a model-efficiency claim.
The raw-source and retained-result hashes, failures, bounds and qualification
limits are in [the public engineering proof](../evidence/public/operator_request_only_controls_20261001.json).

The accepted ERAIS factory forwards all runtime constructor arguments unchanged,
and the existing Minecraft factory checks supplied field identity. A separate
three-case check of the exact accepted factory functions passed with the real
runtime constructor and mocked association/observer initialization. Its bounds,
costs and independent physical release are retained in [the separate constructor
proof](../evidence/public/accepted_factory_request_only_controls_20261001.json).
It proves argument propagation, not actual model/NPZ loading, live factory
assembly or compatibility of the separately prepared deployment overlay. The
production profile and explicit operator pause have not been changed by this work.
