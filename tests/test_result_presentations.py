"""Tests for bridge ResultPresentation parsing and delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent_bridge import send_parsed_presentations


@pytest.mark.asyncio
async def test_send_parsed_presentations_preserves_dynamic_response_template_payload():
    executor = SimpleNamespace(send_message=AsyncMock(return_value={}))
    presentation = {
        "result_type": "screenplay_page",
        "title": "Scene 62",
        "items": [
            {
                "type": "screenplay_page",
                "title": "Scene 62 — Working-Mother Test",
                "detail_template": "screenplay_page",
                "details": {
                    "scene_number": "62",
                    "content": "INT. HOLLIS APARTMENT — NIGHT",
                },
            }
        ],
    }

    with patch("agent_bridge.enrich_presentation_photos", new=AsyncMock()):
        sent = await send_parsed_presentations(
            executor,
            "conv-1",
            [presentation],
            correlation_id="corr-1",
            last_seen_message_id="msg-1",
        )

    assert sent == 1
    executor.send_message.assert_awaited_once_with(
        "conv-1",
        "[Results] Scene 62 (1 items)",
        content_type="structured",
        message_type="ResultPresentation",
        content_structured={
            "schema_version": "2.0",
            "type": "ResultPresentation",
            "data": presentation,
        },
        correlation_id="corr-1",
        last_seen_message_id="msg-1",
    )
