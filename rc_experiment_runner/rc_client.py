"""Async RevenueCat REST API client for experiment integration.

Provides two integration points:
- Switch a subscriber's active offering per variant assignment
- Sync experiment assignment as a RC subscriber attribute

Uses httpx for async HTTP. Falls back gracefully if RC credentials
are not configured (returns None / no-ops).
"""

import os
from typing import Any

import httpx

RC_BASE_URL = "https://api.revenuecat.com/v1"
DEFAULT_TIMEOUT = 10.0


class RCClient:
    """Minimal async RevenueCat REST client for experiment integration."""

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        base_url: str = RC_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        # Explicit empty string means "no key" — don't fall back to env
        if api_key is None:
            self.api_key = os.environ.get("RC_API_KEY", "")
        else:
            self.api_key = api_key
        if project_id is None:
            self.project_id = os.environ.get("RC_PROJECT_ID", "")
        else:
            self.project_id = project_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Platform": "stripe",
        }

    def _configured(self) -> bool:
        return bool(self.api_key and self.project_id)

    async def set_subscriber_attribute(
        self,
        subscriber_id: str,
        key: str,
        value: str,
    ) -> bool:
        """Set a custom subscriber attribute on a RevenueCat subscriber.

        Returns True on success, False if not configured or on error.
        """
        if not self._configured():
            return False

        url = f"{self.base_url}/subscribers/{subscriber_id}/attributes"
        payload: dict[str, Any] = {
            "attributes": {
                key: {"value": value},
            }
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, headers=self._headers(), json=payload)
                return resp.is_success
            except httpx.RequestError:
                return False

    async def get_subscriber(self, subscriber_id: str) -> dict[str, Any] | None:
        """Fetch a subscriber record from RevenueCat.

        Returns the subscriber dict or None on error / not found.
        """
        if not self._configured():
            return None

        url = f"{self.base_url}/subscribers/{subscriber_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(url, headers=self._headers())
                if resp.is_success:
                    data = resp.json()
                    return data.get("subscriber")
                return None
            except httpx.RequestError:
                return None

    async def set_active_offering(
        self,
        subscriber_id: str,
        offering_id: str,
    ) -> bool:
        """Override the active offering for a subscriber.

        Uses the RC override endpoint: POST /subscribers/{id}/offering
        (requires a V2 key or project-scoped secret key in some RC setups).

        Returns True on success, False otherwise.
        Note: This endpoint may require specific key scopes; check RC docs.
        """
        if not self._configured():
            return False

        # RC v1 offering override endpoint
        url = f"{self.base_url}/subscribers/{subscriber_id}/offering"
        payload = {"offering_identifier": offering_id}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.put(url, headers=self._headers(), json=payload)
                return resp.is_success
            except httpx.RequestError:
                return False
