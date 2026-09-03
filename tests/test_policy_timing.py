from minecraft_ai.policy_timing import InferenceRateHold


def test_inference_hold_tracks_latency_with_bounded_ema_horizon() -> None:
    hold = InferenceRateHold(
        minimum_ms=50,
        maximum_ms=250,
        latency_margin=1.2,
        ema_alpha=0.5,
    )

    assert hold.horizon_ms == 50
    hold.observe(200_000_000)
    assert hold.horizon_ms == 240
    hold.observe(400_000_000)
    assert hold.latency_ema_ms == 300.0
    assert hold.horizon_ms == 250


def test_inference_hold_only_renews_for_pending_prediction() -> None:
    hold = InferenceRateHold(minimum_ms=80, maximum_ms=200)

    assert hold.renew_until_ns(now_ns=1_000, prediction_pending=False) is None
    assert hold.renewals == 0
    assert hold.renew_until_ns(now_ns=1_000, prediction_pending=True) == 80_001_000
    assert hold.renewals == 1


def test_inference_hold_cannot_renew_past_request_deadline() -> None:
    hold = InferenceRateHold(minimum_ms=80, maximum_ms=200)

    assert (
        hold.renew_until_ns(
            now_ns=1_000,
            prediction_pending=True,
            request_deadline_ns=5_000,
        )
        == 5_000
    )
    assert (
        hold.renew_until_ns(
            now_ns=5_000,
            prediction_pending=True,
            request_deadline_ns=5_000,
        )
        is None
    )
    assert hold.deadline_expirations == 1
    assert hold.last_invalidation_reason == "request-deadline-expired"


def test_inference_hold_explicit_invalidation_never_returns_a_renewal() -> None:
    hold = InferenceRateHold()

    assert (
        hold.renew_until_ns(
            now_ns=1_000,
            prediction_pending=True,
            state_valid=False,
        )
        is None
    )
    assert hold.invalidations == 1
    assert hold.last_invalidation_reason == "learned-state-invalid"
