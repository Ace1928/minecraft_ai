"""Preparation may extend a valid launch lease, never revive a rejected one."""

import pytest

import minecraft_ai.agent.process as process


@pytest.mark.parametrize("accepted", (True, False))
def test_preparation_renews_exact_launch_authority_before_opening_resources(monkeypatch, accepted):
    calls = []

    def renew(command, **kwargs):
        assert command == "renew"
        assert kwargs == {"lease_id": "launch-lease", "ttl_ms": 5000}
        calls.append("renew")
        if not accepted:
            raise RuntimeError("expired launch lease")
        return {}

    def load_config(path):
        calls.append("prepare")
        raise RuntimeError("end fixture before resource construction")

    def forbidden(*args, **kwargs):
        pytest.fail("preparation test reached capture/database/input owners")

    monkeypatch.setattr(process, "send_command", renew)
    monkeypatch.setattr(process, "load_config", load_config)
    monkeypatch.setattr(process, "StateDatabase", forbidden)
    monkeypatch.setattr(process, "create_bedrock_capture", forbidden)
    monkeypatch.setattr(process, "run_agent_runtime", forbidden)
    message = "end fixture" if accepted else "expired launch lease"
    with pytest.raises(RuntimeError, match=message):
        process.main([
            "--lease-id", "launch-lease", "--display", ":2", "--window-id", "42",
            "--instance-id", "test-only", "--capture-source", "x11",
        ])
    assert calls == (["renew", "prepare"] if accepted else ["renew"])
