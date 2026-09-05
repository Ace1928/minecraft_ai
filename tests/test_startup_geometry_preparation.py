from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

import minecraft_ai.platforms.bedrock_session as sessions
import minecraft_ai.platforms.bedrock_x11 as x11


@pytest.fixture(autouse=True)
def _forbid_real_process_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sessions, "_signal_process_group", lambda *_args: (
        pytest.fail("pure startup geometry tests must never signal a real process")
    ))


def _preparation(monkeypatch: pytest.MonkeyPatch, *, x: int = -4, height: int = 1080) -> Any:
    state = SimpleNamespace(
        snapshot=[1, 2, 3, 1920, 1080, x, 26, 1920, height, -8],
        changes=[], clock=0.0, closed=False, permitted=True,
    )

    def move(**kwargs: int) -> None:
        state.changes.append(("window", kwargs))
        state.snapshot[5] += kwargs["x"] - state.snapshot[9]
        state.snapshot[9] = kwargs["x"]

    def fit(**kwargs: int) -> None:
        state.changes.append(("client", kwargs))
        state.snapshot[7:9] = [kwargs["width"], kwargs["height"]]

    minecraft = SimpleNamespace(configure=move)
    client = SimpleNamespace(configure=fit)
    display = SimpleNamespace(
        sync=lambda: None,
        close=lambda: setattr(state, "closed", True),
    )
    real_import = x11.importlib.import_module
    monkeypatch.setattr(x11.importlib, "import_module", lambda name: (
        SimpleNamespace(Display=lambda _name: display)
        if name == "Xlib.display" else real_import(name)
    ))
    monkeypatch.setattr(x11, "_new_wine_geometry", lambda _display: (
        minecraft, client, tuple(state.snapshot),
    ))
    monkeypatch.setattr(x11.time, "monotonic", lambda: state.clock)
    monkeypatch.setattr(
        x11.time, "sleep", lambda seconds: setattr(state, "clock", state.clock + seconds),
    )
    return state


def _run(state: Any) -> None:
    x11._prepare_new_isolated_window_geometry(
        ":71", ":0", preparation_permitted=lambda: state.permitted, timeout_s=2,
    )


def test_fresh_clipping_is_corrected_once_before_stable_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _preparation(monkeypatch)

    _run(state)

    assert state.changes == [
        ("window", {"x": -4}),
        ("client", {"width": 1920, "height": 1054}),
    ]
    assert state.snapshot[5:9] == [0, 26, 1920, 1054]
    assert state.closed


def test_fresh_contained_geometry_causes_no_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _preparation(monkeypatch, x=0, height=720)

    _run(state)

    assert state.changes == []
    assert state.snapshot[8] == 720


def test_delayed_exact_drawable_discovery_does_not_repair_prematurely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _preparation(monkeypatch)
    resolve = x11._new_wine_geometry

    def delayed(display: Any) -> Any:
        if state.clock < 0.5:
            assert state.changes == []
            raise x11.IsolationError("drawable not ready")
        return resolve(display)

    monkeypatch.setattr(x11, "_new_wine_geometry", delayed)

    _run(state)

    assert len(state.changes) == 2


def test_failed_correction_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _preparation(monkeypatch)
    minecraft = SimpleNamespace(configure=lambda **kw: state.changes.append(("window", kw)))
    client = SimpleNamespace(configure=lambda **kw: state.changes.append(("client", kw)))
    monkeypatch.setattr(x11, "_new_wine_geometry", lambda _display: (
        minecraft, client, tuple(state.snapshot),
    ))

    with pytest.raises(x11.IsolationError, match="clipped"):
        _run(state)

    assert len(state.changes) == 2
    assert state.closed


def test_lost_authority_prevents_fitting_after_horizontal_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _preparation(monkeypatch)
    resolve = x11._new_wine_geometry
    minecraft, client, _ = resolve(None)
    move = minecraft.configure

    def interrupted(**kwargs: int) -> None:
        move(**kwargs)
        state.permitted = False

    minecraft.configure = interrupted

    with pytest.raises(x11.IsolationError, match="interrupted"):
        _run(state)

    assert state.changes == [("window", {"x": -4})]
    assert state.closed


def test_missing_drawable_exhausts_bounded_wait_without_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _preparation(monkeypatch)

    def missing(_display: Any) -> Any:
        raise x11.IsolationError("not found")

    monkeypatch.setattr(x11, "_new_wine_geometry", missing)

    with pytest.raises(x11.IsolationError, match="timed out"):
        _run(state)

    assert state.clock == 2
    assert state.changes == []
    assert state.closed


def test_replaced_drawable_after_correction_is_rejected_without_more_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _preparation(monkeypatch)
    resolve = x11._new_wine_geometry

    def replaced(display: Any) -> Any:
        if state.changes:
            state.snapshot[2] = 99
        return resolve(display)

    monkeypatch.setattr(x11, "_new_wine_geometry", replaced)

    with pytest.raises(x11.IsolationError, match="drawable changed"):
        _run(state)

    assert len(state.changes) == 2


@pytest.mark.parametrize("mode", ["direct", "host-monitor"])
def test_host_session_cannot_request_startup_geometry_repair(mode: str) -> None:
    with pytest.raises(x11.IsolationError, match="new isolated"):
        sessions._prepare_new_isolated_bedrock_geometry(SimpleNamespace(mode=mode))


@pytest.mark.parametrize("bad", [None, "name", "class", "hidden", "ambiguous_client"])
def test_startup_repair_requires_positive_viewable_exact_game_identity(bad: str | None) -> None:
    root = SimpleNamespace(id=1)
    geometry = SimpleNamespace(width=1920, height=1080)
    attributes = SimpleNamespace(map_state=2, win_class=1)
    client = SimpleNamespace(
        id=3, get_attributes=lambda: attributes,
        get_geometry=lambda: geometry,
        get_wm_class=lambda: (), get_wm_name=lambda: None,
        query_tree=lambda: SimpleNamespace(children=[]),
    )
    minecraft = SimpleNamespace(
        id=2,
        get_attributes=lambda: SimpleNamespace(map_state=0 if bad == "hidden" else 2),
        get_geometry=lambda: geometry,
        get_wm_class=lambda: ("explorer.exe",) if bad == "class" else ("minecraft.windows.exe",),
        get_wm_name=lambda: "Other" if bad == "name" else "Minecraft",
        query_tree=lambda: SimpleNamespace(
            parent=root, children=[client, client] if bad == "ambiguous_client" else [client],
        ),
    )
    root.query_tree = lambda: SimpleNamespace(children=[minecraft])
    root.get_attributes = lambda: attributes
    root.get_geometry = lambda: geometry
    root.translate_coords = lambda *_args: SimpleNamespace(x=0, y=0)
    display = SimpleNamespace(
        screen=lambda: SimpleNamespace(root=root),
        create_resource_object=lambda _kind, _id: minecraft,
    )

    if bad is None:
        observed_window, observed_client, snapshot = x11._new_wine_geometry(display)
        assert observed_window is minecraft
        assert observed_client is client
        assert snapshot[:3] == (1, 2, 3)
    else:
        with pytest.raises(x11.IsolationError):
            x11._new_wine_geometry(display)


def test_existing_live_session_launch_is_a_noop_for_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minecraft_ai import cli

    existing = SimpleNamespace(mode="weston")
    monkeypatch.setattr(cli.BedrockSession, "load", lambda: existing)
    monkeypatch.setattr(cli, "bedrock_session_alive", lambda _session: True)
    monkeypatch.setattr(cli, "_session_payload", lambda _session: {})
    monkeypatch.setattr(sessions, "_prepare_new_isolated_bedrock_geometry", lambda _session: (
        pytest.fail("existing game must not be repaired by a launch retry")
    ))
    monkeypatch.setattr(cli, "launch_isolated_bedrock_session", lambda **_kwargs: (
        pytest.fail("existing game must not be relaunched")
    ))

    cli._bedrock_launch_locked(width=1920, height=1080, fullscreen=True, direct=False)


def test_dead_launcher_does_not_authorize_replacing_a_remaining_legacy_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import typer
    from minecraft_ai import cli

    existing = SimpleNamespace(mode="weston", input_isolation="unverified")
    monkeypatch.setattr(cli.BedrockSession, "load", lambda: existing)
    monkeypatch.setattr(cli, "bedrock_session_alive", lambda _session: False)
    monkeypatch.setattr(cli, "bedrock_session_resources_absent", lambda _session: False)
    monkeypatch.setattr(cli, "stop_bedrock_session", lambda _session: (
        pytest.fail("remaining legacy game must be preserved")
    ))
    monkeypatch.setattr(cli, "launch_isolated_bedrock_session", lambda **_kwargs: (
        pytest.fail("legacy game must not be replaced")
    ))

    with pytest.raises(typer.BadParameter, match="preserved for observation"):
        cli._bedrock_launch_locked(width=1920, height=1080, fullscreen=True, direct=False)


def test_proven_absent_session_can_be_cleaned_up_before_new_headless_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minecraft_ai import cli

    existing = SimpleNamespace(mode="weston", input_isolation="unverified")
    calls: list[str] = []
    monkeypatch.setattr(cli.BedrockSession, "load", lambda: existing)
    monkeypatch.setattr(cli, "bedrock_session_alive", lambda _session: False)
    monkeypatch.setattr(cli, "bedrock_session_resources_absent", lambda _session: True)
    monkeypatch.setattr(cli, "stop_bedrock_session", lambda _session: calls.append("cleanup"))
    monkeypatch.setattr(cli, "launch_isolated_bedrock_session", lambda **_kwargs: (
        calls.append("launch") or object()
    ))
    monkeypatch.setattr(cli, "_session_payload", lambda _session: {})

    cli._bedrock_launch_locked(width=1920, height=1080, fullscreen=True, direct=False)
    assert calls == ["cleanup", "launch"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux launch path")
def test_direct_debug_launch_never_enters_isolated_geometry_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted: list[Any] = []
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(sessions.subprocess, "Popen", lambda *_args, **_kwargs: (
        SimpleNamespace(pid=123)
    ))
    monkeypatch.setattr(sessions, "_required_process_identity", lambda *_args, **_kwargs: (
        1, "digest",
    ))
    monkeypatch.setattr(
        sessions.BedrockSession, "persist", lambda session: persisted.append(session),
    )
    monkeypatch.setattr(sessions, "_terminate_spawned_process_group", lambda _child: True)
    monkeypatch.setattr(sessions, "_prepare_new_isolated_bedrock_geometry", lambda _session: (
        pytest.fail("host launch must not enter isolated geometry preparation")
    ))

    session = sessions.launch_direct_bedrock_session(
        launcher_command=("/usr/bin/bedrock-on-linux", "play"),
    )

    assert session.mode == "direct"
    assert persisted == [session]
