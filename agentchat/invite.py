"""Invite claiming and credential management for AgentChat.

Provides functions to claim invite codes, save/load credentials,
and query invite info — everything needed for zero-config agent onboarding.

Usage:
    from agentchat.invite import claim_invite, save_credentials

    result = await claim_invite("https://agentchat-backend.fly.dev", "inv_Abc123")
    save_credentials(result)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("agentchat.invite")

_CREDENTIALS_DIR = Path.home() / ".agentchat"
_CREDENTIALS_FILE = _CREDENTIALS_DIR / "credentials.json"


@dataclass
class ClaimResult:
    """Result of claiming an invite code."""

    agent_id: str
    api_key: str
    gateway_url: str
    display_name: str
    executor_id: Optional[str] = None


async def claim_invite(
    gateway_url: str,
    code: str,
    executor_key: Optional[str] = None,
    executor_display_name: Optional[str] = None,
    executor_capabilities: Optional[list] = None,
) -> ClaimResult:
    """Claim an invite code and get back agent credentials.

    Args:
        gateway_url: Base URL of the AgentChat backend (e.g. https://agentchat-backend.fly.dev)
        code: The invite code (with or without inv_ prefix)
        executor_key: Optional executor key to auto-register a gateway executor
        executor_display_name: Optional display name for the executor
        executor_capabilities: Optional list of executor capabilities

    Returns:
        ClaimResult with agent_id, api_key, gateway_url, display_name, and optional executor_id

    Raises:
        httpx.HTTPStatusError: If the claim fails (404, 409, 410, etc.)
    """
    base_url = gateway_url.rstrip("/")

    # Ensure code has inv_ prefix
    if not code.startswith("inv_"):
        code = f"inv_{code}"

    body = {}
    if executor_key:
        body["executorKey"] = executor_key
    if executor_display_name:
        body["executorDisplayName"] = executor_display_name
    if executor_capabilities:
        body["executorCapabilities"] = executor_capabilities

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{base_url}/api/invites/{code}/claim",
            json=body if body else None,
        )

    if resp.status_code == 404:
        raise ValueError(f"Invite not found: {code}")
    if resp.status_code == 409:
        raise ValueError(f"Invite already claimed or revoked: {code}")
    if resp.status_code == 410:
        raise ValueError(f"Invite expired or exhausted: {code}")
    if resp.status_code != 201:
        raise ValueError(f"Claim failed (HTTP {resp.status_code}): {resp.text}")

    data = resp.json()

    executor_id = None
    if data.get("executor"):
        executor_id = data["executor"].get("id")

    return ClaimResult(
        agent_id=data["agentId"],
        api_key=data["apiKey"],
        gateway_url=data.get("gatewayUrl", base_url),
        display_name=data["agent"]["displayName"],
        executor_id=executor_id,
    )


async def get_invite_info(gateway_url: str, code: str) -> dict:
    """Get public info about an invite code (no auth required).

    Returns dict with displayName, description, capabilities, status,
    creatorName, expiresAt, maxUses, uses.
    """
    base_url = gateway_url.rstrip("/")

    if not code.startswith("inv_"):
        code = f"inv_{code}"

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base_url}/api/invites/{code}/info")

    if resp.status_code == 404:
        raise ValueError(f"Invite not found: {code}")
    if resp.status_code != 200:
        raise ValueError(f"Info request failed (HTTP {resp.status_code}): {resp.text}")

    return resp.json()


def save_credentials(result: ClaimResult) -> Path:
    """Save claim result to ~/.agentchat/credentials.json.

    Returns the path to the saved file.
    """
    _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

    data = asdict(result)
    _CREDENTIALS_FILE.write_text(json.dumps(data, indent=2) + "\n")
    os.chmod(_CREDENTIALS_FILE, 0o600)

    logger.info("Credentials saved to %s", _CREDENTIALS_FILE)
    return _CREDENTIALS_FILE


def load_credentials() -> Optional[ClaimResult]:
    """Load saved credentials from ~/.agentchat/credentials.json.

    Returns ClaimResult or None if no credentials file exists.
    """
    if not _CREDENTIALS_FILE.exists():
        return None

    try:
        data = json.loads(_CREDENTIALS_FILE.read_text())
        return ClaimResult(
            agent_id=data["agent_id"],
            api_key=data["api_key"],
            gateway_url=data["gateway_url"],
            display_name=data["display_name"],
            executor_id=data.get("executor_id"),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load credentials: %s", e)
        return None
