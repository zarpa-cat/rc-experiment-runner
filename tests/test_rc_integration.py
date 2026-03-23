"""Tests for RevenueCat integration: RCClient and assign_with_rc_sync."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rc_experiment_runner.models import Experiment, Variant
from rc_experiment_runner.rc_client import RCClient
from rc_experiment_runner.runner import RC_EXPERIMENT_ATTR_PREFIX, ExperimentRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment() -> Experiment:
    return Experiment(
        id="pricing-test",
        name="Pricing Test",
        variants=[
            Variant(id="control", name="Control ($9.99)", weight=0.5),
            Variant(id="treatment", name="Treatment ($12.99)", weight=0.5),
        ],
        start_date=datetime(2026, 1, 1, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# RCClient: configuration checks
# ---------------------------------------------------------------------------


def test_rc_client_not_configured_without_credentials() -> None:
    client = RCClient(api_key="", project_id="")
    assert not client._configured()


def test_rc_client_configured_with_credentials() -> None:
    client = RCClient(api_key="sk_live_xxx", project_id="proj_yyy")
    assert client._configured()


def test_rc_client_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RC_API_KEY", "env_key")
    monkeypatch.setenv("RC_PROJECT_ID", "env_proj")
    client = RCClient()
    assert client.api_key == "env_key"
    assert client.project_id == "env_proj"


def test_rc_client_env_overridden_by_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RC_API_KEY", "env_key")
    client = RCClient(api_key="explicit_key")
    assert client.api_key == "explicit_key"


# ---------------------------------------------------------------------------
# RCClient: set_subscriber_attribute — not configured → returns False
# ---------------------------------------------------------------------------


def test_set_attribute_not_configured_returns_false() -> None:
    client = RCClient(api_key="", project_id="")
    result = asyncio.run(client.set_subscriber_attribute("user1", "rce_exp", "control"))
    assert result is False


def test_get_subscriber_not_configured_returns_none() -> None:
    client = RCClient(api_key="", project_id="")
    result = asyncio.run(client.get_subscriber("user1"))
    assert result is None


def test_set_active_offering_not_configured_returns_false() -> None:
    client = RCClient(api_key="", project_id="")
    result = asyncio.run(client.set_active_offering("user1", "default"))
    assert result is False


# ---------------------------------------------------------------------------
# RCClient: HTTP mocking — success paths
# ---------------------------------------------------------------------------


def test_set_subscriber_attribute_success() -> None:
    client = RCClient(api_key="sk_live_xxx", project_id="proj")

    mock_response = MagicMock()
    mock_response.is_success = True

    with patch("httpx.AsyncClient") as mock_httpx:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_ctx.post = AsyncMock(return_value=mock_response)
        mock_httpx.return_value = mock_ctx

        result = asyncio.run(client.set_subscriber_attribute("user1", "rce_exp", "control"))

    assert result is True
    mock_ctx.post.assert_called_once()
    call_kwargs = mock_ctx.post.call_args
    assert "user1" in call_kwargs.args[0]
    assert "attributes" in call_kwargs.kwargs["json"]


def test_set_subscriber_attribute_failure() -> None:
    client = RCClient(api_key="sk_live_xxx", project_id="proj")

    mock_response = MagicMock()
    mock_response.is_success = False

    with patch("httpx.AsyncClient") as mock_httpx:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_ctx.post = AsyncMock(return_value=mock_response)
        mock_httpx.return_value = mock_ctx

        result = asyncio.run(client.set_subscriber_attribute("user1", "rce_exp", "control"))

    assert result is False


def test_set_subscriber_attribute_network_error() -> None:
    import httpx as _httpx

    client = RCClient(api_key="sk_live_xxx", project_id="proj")

    with patch("httpx.AsyncClient") as mock_httpx:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_ctx.post = AsyncMock(side_effect=_httpx.RequestError("timeout"))
        mock_httpx.return_value = mock_ctx

        result = asyncio.run(client.set_subscriber_attribute("user1", "rce_exp", "control"))

    assert result is False


def test_get_subscriber_success() -> None:
    client = RCClient(api_key="sk_live_xxx", project_id="proj")

    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.json = MagicMock(return_value={"subscriber": {"original_app_user_id": "user1"}})

    with patch("httpx.AsyncClient") as mock_httpx:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_ctx.get = AsyncMock(return_value=mock_response)
        mock_httpx.return_value = mock_ctx

        result = asyncio.run(client.get_subscriber("user1"))

    assert result == {"original_app_user_id": "user1"}


def test_get_subscriber_not_found() -> None:
    client = RCClient(api_key="sk_live_xxx", project_id="proj")

    mock_response = MagicMock()
    mock_response.is_success = False

    with patch("httpx.AsyncClient") as mock_httpx:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_ctx.get = AsyncMock(return_value=mock_response)
        mock_httpx.return_value = mock_ctx

        result = asyncio.run(client.get_subscriber("user1"))

    assert result is None


def test_set_active_offering_success() -> None:
    client = RCClient(api_key="sk_live_xxx", project_id="proj")

    mock_response = MagicMock()
    mock_response.is_success = True

    with patch("httpx.AsyncClient") as mock_httpx:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_ctx.put = AsyncMock(return_value=mock_response)
        mock_httpx.return_value = mock_ctx

        result = asyncio.run(client.set_active_offering("user1", "premium"))

    assert result is True


# ---------------------------------------------------------------------------
# ExperimentRunner.assign_with_rc_sync
# ---------------------------------------------------------------------------


def test_assign_with_rc_sync_no_rc_client(tmp_path: "pytest.TempPathFactory") -> None:
    """Without RC client, falls back to plain local assign."""
    runner = ExperimentRunner(db_path=str(tmp_path / "test.db"), rc_client=None)
    exp = _make_experiment()
    runner.create_experiment(exp)

    variant = asyncio.run(runner.assign_with_rc_sync("user1", "pricing-test"))
    assert variant.id in ("control", "treatment")


def test_assign_with_rc_sync_calls_set_attribute(tmp_path: "pytest.TempPathFactory") -> None:
    """With RC client, syncs assignment as subscriber attribute."""
    mock_rc = AsyncMock(spec=RCClient)
    mock_rc.set_subscriber_attribute = AsyncMock(return_value=True)
    mock_rc.set_active_offering = AsyncMock(return_value=True)

    runner = ExperimentRunner(db_path=str(tmp_path / "test.db"), rc_client=mock_rc)
    exp = _make_experiment()
    runner.create_experiment(exp)

    variant = asyncio.run(runner.assign_with_rc_sync("user1", "pricing-test"))

    expected_key = f"{RC_EXPERIMENT_ATTR_PREFIX}pricing-test"
    mock_rc.set_subscriber_attribute.assert_called_once_with("user1", expected_key, variant.id)


def test_assign_with_rc_sync_offering_map(tmp_path: "pytest.TempPathFactory") -> None:
    """With offering_map, calls set_active_offering for the assigned variant."""
    mock_rc = AsyncMock(spec=RCClient)
    mock_rc.set_subscriber_attribute = AsyncMock(return_value=True)
    mock_rc.set_active_offering = AsyncMock(return_value=True)

    runner = ExperimentRunner(db_path=str(tmp_path / "test.db"), rc_client=mock_rc)
    exp = _make_experiment()
    runner.create_experiment(exp)

    offering_map = {"control": "default", "treatment": "premium"}
    variant = asyncio.run(
        runner.assign_with_rc_sync("user1", "pricing-test", offering_map=offering_map)
    )

    expected_offering = offering_map[variant.id]
    mock_rc.set_active_offering.assert_called_once_with("user1", expected_offering)


def test_assign_with_rc_sync_no_offering_map_skips_offering(
    tmp_path: "pytest.TempPathFactory",
) -> None:
    """Without offering_map, set_active_offering is not called."""
    mock_rc = AsyncMock(spec=RCClient)
    mock_rc.set_subscriber_attribute = AsyncMock(return_value=True)
    mock_rc.set_active_offering = AsyncMock(return_value=True)

    runner = ExperimentRunner(db_path=str(tmp_path / "test.db"), rc_client=mock_rc)
    exp = _make_experiment()
    runner.create_experiment(exp)

    asyncio.run(runner.assign_with_rc_sync("user1", "pricing-test"))

    mock_rc.set_active_offering.assert_not_called()


def test_assign_with_rc_sync_idempotent(tmp_path: "pytest.TempPathFactory") -> None:
    """Re-assigning returns the same variant (deterministic)."""
    mock_rc = AsyncMock(spec=RCClient)
    mock_rc.set_subscriber_attribute = AsyncMock(return_value=True)

    runner = ExperimentRunner(db_path=str(tmp_path / "test.db"), rc_client=mock_rc)
    exp = _make_experiment()
    runner.create_experiment(exp)

    v1 = asyncio.run(runner.assign_with_rc_sync("user1", "pricing-test"))
    v2 = asyncio.run(runner.assign_with_rc_sync("user1", "pricing-test"))
    assert v1.id == v2.id
