# Priority handoff: operational limited player directions

Lloyd explicitly requests this handoff so the active Minecraft agent can make
limited directions reliable enough for the business task to finish subscriptions
and prepaid per-direction charging. This is requested implementation work, not
an available service. Preserve the running game, public stream and other agents'
edits. Extend existing runtime machinery rather than create another controller.

## Smallest useful live product

Deliver a versioned menu of bounded instructions that the persistent player can
actually execute and verify. Start with one proven instruction, then add useful
look/inspect/navigation actions only as their outcome checks pass. Arbitrary text
may be rejected or clarified before admission. Do not sell acknowledgement as
execution, and do not require solving unrestricted Minecraft play first.

The retained diagnostic is: open inventory, observe it, close it; no movement or
attack. Earlier trials failed ordered completion. After the GUI fix, trial
de7548 reportedly stayed delivered without acknowledgement for 120 seconds;
that did not exercise the detector. Resolve admission/processing latency and
re-run the same case. A successful no-op close or unrelated recovery action is
not completion. Report fresh attributable evidence, including negative results.

## Exact contract required by the business adapter

Provide a narrow private service boundary, not a public proxy of the operator
dashboard. Existing POST /api/messages and its last-100-message listing lack the
required durable idempotency and completion contract. LAN origin checks and the
author text field are not subscriber authentication.

1. **Discovery:** protocol version, supported instruction IDs/typed arguments,
   validation limits, readiness, current player/world and session/control epoch,
   queue capacity and declared admission/execution timeouts. Return unavailable
   when paused, unsafe, disconnected or without a valid control lease.
2. **Submit:** service-authenticated request with caller-generated request ID,
   authenticated member reference, exact player/world, expected session epoch,
   supported instruction/arguments and deadline. Member identity comes from the
   private business gateway, never message text. Runtime assigns attempt/message
   and goal IDs. Bind the entire request to its durable receipt before dispatch.
3. **Idempotency:** same request ID and identical payload returns the existing
   receipt; changed payload conflicts. Retain lookup by exact ID across restarts.
   A lost response must be reconcilable without scanning only 100 messages or
   sending the command again. Uncertain execution remains unknown until resolved.
4. **Status:** queued, running, succeeded, failed, cancelled, expired, or unknown;
   include stable IDs, epoch, timestamps, safe reason code and terminal evidence
   reference. Delivered/acknowledged are intermediate states, never succeeded.
5. **Outcome:** success needs fresh observations proving the requested result
   and constraints for that attempt. Link relevant input sequences, skill runs
   and observation times; keep private frames/diagnostics out of public responses.
   For the inventory case prove open then closed, with no forbidden movement or
   attack. A generic skill success label alone is insufficient.
6. **Cancel and priority:** cancellation stops future owned dispatch and reports
   whether an action had already occurred. Operator pause/emergency and session
   changes revoke subscriber authority. Do not bypass existing supervisor gates.
   Bound queue length and per-member outstanding work so a buyer cannot monopolise
   the player. Expired requests cannot start later.

You may choose transport and field names to fit the existing architecture.
Return the actual schema plus a working private client example; these are
required semantics, not a demand to implement a speculative public HTTP API.
Keep credentials local, and share configuration variable names, never secrets.

## Acceptance needed before paid activation

- One real supported instruction from authenticated submission through observed
  terminal outcome, with the same identity chain throughout.
- Duplicate submission and delayed duplicate outcome produce one execution and
  one result. Changed payload under the same ID is rejected.
- Restart/lost response reconciliation does not blindly redispatch uncertain work.
- Wrong member/player/world, stale epoch, invalid arguments, full queue and expired
  deadline are refused; pause/cancel during execution respects operator priority.
- Missing or mismatched outcome evidence never becomes successful paid work.
- Measure admission latency, completion latency, failure rate and incremental
  CPU/GPU/network usage over a stated small trial set. Report sample size and
  limitations, not general capacity claims. Preserve stream continuity.

Deliver source commit, exact test commands/results, private receipt location,
supported-command list and remaining failures to the business task. A fixture
pass is valuable but must be distinguished from an exercised game result.

## Business-side integration and charging ownership

Private repo neuroforge-io/neuroforge_business already has hosted_agents/directions.py
and its synthetic rehearsal: reservation, durable receipts and completion debit.
Reuse it after mapping the actual runtime contract. Keep billing, customer
identities, API keys and proprietary ERAIS implementation out of this public repo.

Both monthly allowances and prepaid direction credits can use the same ledger:
reserve one credit at admission; debit once only after verified success; release
on failed/cancelled/expired work. Unknown work stays reserved pending bounded
reconciliation and customer resolution. Do not recharge retries. Prefer prepaid
credits to a separate card transaction for every message. Raw chat is not a
billable completed direction.

The business task owns price, terms, Square payment verification, renewals,
refund/reversal handling, member login, customer receipt and public checkout.
Square company account can accept a payment; bank verification and live billing
integration remain separate checks. No price or sale is authorised by this file.
Runtime completion is the main integration dependency this handoff asks you to
close, not permission to weaken completion accounting.

Share the world may progress independently: report current Bedrock join/auth,
allowlist, expiry/disconnection, backup and tested capacity. Do not block this
limited-directions work on a world-access launch or migrate the running world.
