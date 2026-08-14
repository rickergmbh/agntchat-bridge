"""Tests for bridge ResultPresentation parsing and delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge import parse_result_presentations, send_parsed_presentations


def test_parse_extracts_valid_block_and_strips_it():
    text = (
        "Here are some options:\n\n"
        '<result_presentation>{"result_type": "restaurant", "title": "Cafés", '
        '"items": [{"type": "restaurant", "title": "Café Gretchen"}]}</result_presentation>'
        "\n\nWant directions?"
    )
    remaining, presentations = parse_result_presentations(text)

    assert len(presentations) == 1
    assert presentations[0]["result_type"] == "restaurant"
    # The tag is removed from visible text; surrounding prose is preserved.
    assert "<result_presentation>" not in remaining
    assert "Here are some options:" in remaining
    assert "Want directions?" in remaining


def test_invalid_json_block_is_salvaged_not_dropped():
    # A block the model wrote as XML children instead of JSON — this is the
    # café-list bug: it must NOT leave an empty gap in the reply.
    text = (
        "Here's what's close to you:\n\n"
        "<result_presentation>\n"
        "  <title>Café Gretchen</title>\n"
        "  <subtitle>Aachener Str. 1</subtitle>\n"
        "</result_presentation>\n\n"
        "Want directions?"
    )
    remaining, presentations = parse_result_presentations(text)

    assert presentations == []
    # Content survives as text rather than vanishing.
    assert "Café Gretchen" in remaining
    assert "Aachener Str. 1" in remaining
    # No raw protocol tags bleed through.
    assert "<result_presentation>" not in remaining
    assert "<title>" not in remaining
    # No empty hole where the block was.
    assert "\n\n\n\n" not in remaining


def test_invalid_json_object_block_is_salvaged_as_bullets():
    # Valid JSON dict but missing result_type/items → salvage titles/subtitles.
    text = (
        "Results:\n\n"
        '<result_presentation>{"title": "Cafés", "items": ['
        '{"title": "Café Gretchen", "subtitle": "Aachener Str. 1"},'
        '{"title": "Ernst Kaffeeröster"}]}</result_presentation>'
    )
    remaining, presentations = parse_result_presentations(text)

    assert presentations == []
    assert "Café Gretchen" in remaining
    assert "Aachener Str. 1" in remaining
    assert "Ernst Kaffeeröster" in remaining


def test_text_without_blocks_is_unchanged():
    text = "Just a normal reply with no structured output."
    remaining, presentations = parse_result_presentations(text)
    assert presentations == []
    assert remaining == text


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
