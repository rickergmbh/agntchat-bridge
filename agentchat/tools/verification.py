"""Post-action verification for side-effecting tool calls.

When an agent performs a side-effecting action (save_draft, send_email,
create_calendar_event), the system MUST verify the action actually occurred
before reporting success. This module provides that verification layer.

Architecture:
    ToolExecutor calls execute() → side-effect happens → PostActionVerifier
    checks the result using a read-back API call → returns verified result
    or raises VerificationError.

The verification registry maps (executor_method → verify function).
Each verify function receives (client, action_result) and returns a
VerificationResult with verified=True/False.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("agentchat.tools.verification")


@dataclass
class VerificationResult:
    """Result of a post-action verification check."""

    verified: bool
    action: str
    detail: str
    resource_id: str | None = None


class VerificationError(Exception):
    """Raised when a side-effecting action cannot be verified."""

    def __init__(self, action: str, detail: str) -> None:
        self.action = action
        self.detail = detail
        super().__init__(f"Verification failed for {action}: {detail}")


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------


async def _verify_save_draft(
    client: Any, result: dict[str, Any]
) -> VerificationResult:
    """Verify a Gmail draft was actually saved by reading it back."""
    draft_id = None
    if isinstance(result, dict):
        draft_id = result.get("draft_id") or result.get("draftId")

    if not draft_id:
        return VerificationResult(
            verified=False,
            action="save_draft",
            detail="No draft_id returned from save_draft call",
        )

    try:
        check = await client._get(f"/api/google/gmail/drafts/{draft_id}")
        if isinstance(check, dict) and check.get("exists") is True:
            logger.info(
                "[verify] save_draft verified: draft %s exists", draft_id
            )
            return VerificationResult(
                verified=True,
                action="save_draft",
                detail=f"Draft {draft_id} confirmed in Gmail",
                resource_id=draft_id,
            )
        else:
            logger.warning(
                "[verify] save_draft FAILED: draft %s not found", draft_id
            )
            return VerificationResult(
                verified=False,
                action="save_draft",
                detail=f"Draft {draft_id} not found in Gmail after saving",
                resource_id=draft_id,
            )
    except Exception as e:
        logger.warning("[verify] save_draft verification error: %s", e)
        return VerificationResult(
            verified=False,
            action="save_draft",
            detail=f"Verification check failed: {e}",
            resource_id=draft_id,
        )


async def _verify_send_email(
    client: Any, result: dict[str, Any]
) -> VerificationResult:
    """Verify a sent email exists by reading back its message_id."""
    message_id = None
    if isinstance(result, dict):
        message_id = result.get("message_id") or result.get("messageId")

    if not message_id:
        return VerificationResult(
            verified=False,
            action="send_email",
            detail="No message_id returned from send_email call",
        )

    try:
        check = await client._get(
            f"/api/google/gmail/messages/{message_id}"
        )
        if isinstance(check, dict) and check.get("id"):
            logger.info(
                "[verify] send_email verified: message %s exists", message_id
            )
            return VerificationResult(
                verified=True,
                action="send_email",
                detail=f"Sent email {message_id} confirmed in Gmail",
                resource_id=message_id,
            )
        else:
            logger.warning(
                "[verify] send_email FAILED: message %s not found",
                message_id,
            )
            return VerificationResult(
                verified=False,
                action="send_email",
                detail=f"Sent email {message_id} not found in Gmail",
                resource_id=message_id,
            )
    except Exception as e:
        logger.warning("[verify] send_email verification error: %s", e)
        return VerificationResult(
            verified=False,
            action="send_email",
            detail=f"Verification check failed: {e}",
            resource_id=message_id,
        )


async def _verify_create_calendar_event(
    client: Any, result: dict[str, Any]
) -> VerificationResult:
    """Verify a calendar event was created by reading it back."""
    event_id = None
    if isinstance(result, dict):
        event_id = result.get("event_id") or result.get("eventId") or result.get("id")

    if not event_id:
        return VerificationResult(
            verified=False,
            action="create_calendar_event",
            detail="No event_id returned from create_calendar_event call",
        )

    try:
        check = await client._get(
            f"/api/google/calendar/events/{event_id}"
        )
        if isinstance(check, dict) and check.get("exists") is True:
            logger.info(
                "[verify] create_calendar_event verified: event %s exists",
                event_id,
            )
            return VerificationResult(
                verified=True,
                action="create_calendar_event",
                detail=f"Event {event_id} confirmed in Google Calendar",
                resource_id=event_id,
            )
        else:
            logger.warning(
                "[verify] create_calendar_event FAILED: event %s not found",
                event_id,
            )
            return VerificationResult(
                verified=False,
                action="create_calendar_event",
                detail=f"Event {event_id} not found in Google Calendar after creation",
                resource_id=event_id,
            )
    except Exception as e:
        logger.warning(
            "[verify] create_calendar_event verification error: %s", e
        )
        return VerificationResult(
            verified=False,
            action="create_calendar_event",
            detail=f"Verification check failed: {e}",
            resource_id=event_id,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps executor_method name → verification function.
# Only side-effecting methods that create/modify external resources are listed.
VERIFICATION_REGISTRY: dict[str, Any] = {
    "save_draft": _verify_save_draft,
    "send_email": _verify_send_email,
    "create_calendar_event": _verify_create_calendar_event,
}


def needs_verification(executor_method: str) -> bool:
    """Check if an executor method has a registered verification function."""
    return executor_method in VERIFICATION_REGISTRY


async def verify_action(
    client: Any, executor_method: str, result: Any
) -> VerificationResult | None:
    """Run post-action verification for a side-effecting tool call.

    Returns VerificationResult if the method has a verifier, None otherwise.
    Never raises — verification failures are returned as VerificationResult
    with verified=False.
    """
    verifier = VERIFICATION_REGISTRY.get(executor_method)
    if not verifier:
        return None

    # Parse result if it's a JSON string
    parsed = result
    if isinstance(result, str):
        import json

        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            parsed = {}

    try:
        return await verifier(client, parsed)
    except Exception as e:
        logger.error(
            "[verify] Unexpected error verifying %s: %s",
            executor_method,
            e,
        )
        return VerificationResult(
            verified=False,
            action=executor_method,
            detail=f"Verification crashed: {e}",
        )


__all__ = [
    "PostActionVerifier",
    "VerificationResult",
    "VerificationError",
    "VERIFICATION_REGISTRY",
    "needs_verification",
    "verify_action",
]
