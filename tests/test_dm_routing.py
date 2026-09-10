"""<dm> routing: what lands in the group conversation afterwards.

Bridge follow-up 3 of the 2026-09-06 coordination audit. The DM ROUTING
directive promises "the tag is stripped from your group message", and the
server-side router (`Messaging.route_dm_blocks/6`) keeps the remaining text,
posting the hidden `EndTurn(thread_redirect)` only when nothing remains and
the visible failure notice only when nothing remains AND nothing routed.
`_route_dm_blocks_and_settle` must match that exactly; before this the
bridge dropped the whole reply once any target routed.

Follow-up 2: the hidden redirect is the canonical EndTurn JSON payload
(`{"reason": "thread_redirect", "message": ...}`), not prose.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from agent_bridge import (
    _parse_dm_blocks,
    _route_dm_blocks_and_settle,
    _send_hidden_thread_redirect,
)

MEMBERS = [
    {"displayName": "Bob", "participantId": "bob-id"},
    {"displayName": "Carol", "participantId": "carol-id"},
    {"displayName": "Me", "participantId": "me-id"},
]


def _executor(dm_id: str | None = "dm-1") -> NS:
    return NS(
        find_or_create_dm=AsyncMock(return_value={"id": dm_id} if dm_id else {}),
        send_message=AsyncMock(return_value={}),
    )


def _msg() -> NS:
    return NS(
        conversation_members=MEMBERS,
        conversation_id="conv-1",
        message_id="m1",
        latest_seen_message_id="m2",
    )


async def _settle(executor, reply, blocks, **kw):
    return await _route_dm_blocks_and_settle(
        executor, reply, blocks, _msg(), "exec-key",
        {"model": "m"}, kw.pop("directives", None), {"model": "m", "stream_id": "s1"},
        kw.pop("behavioral_config", None), **kw,
    )


def _end_turn_calls(executor):
    return [
        c for c in executor.send_message.await_args_list
        if c.kwargs.get("message_type") == "EndTurn"
    ]


class TestParse:
    def test_strips_tag_and_keeps_remaining_text(self):
        remaining, blocks = _parse_dm_blocks(
            'Sure, on it.\n<dm target="Bob" topic="lunch">pick a place?</dm>\nBack in a sec.'
        )
        assert blocks == [{"target": "Bob", "content": "pick a place?", "topic": "lunch"}]
        assert remaining == "Sure, on it.\n\nBack in a sec."

    def test_only_tag_leaves_nothing(self):
        remaining, blocks = _parse_dm_blocks('<dm target="Bob">hi</dm>')
        assert remaining == ""
        assert len(blocks) == 1


class TestSettle:
    @pytest.mark.asyncio
    async def test_routed_with_remaining_text_posts_text_no_redirect(self):
        ex = _executor()
        reply, blocks = _parse_dm_blocks('Looping Bob in. <dm target="Bob">details</dm>')
        out = await _settle(ex, reply, blocks)
        assert out == "Looping Bob in."
        # The DM itself went out, and nothing else.
        ex.find_or_create_dm.assert_awaited_once()
        assert ex.find_or_create_dm.await_args.args == ("bob-id",)
        assert ex.find_or_create_dm.await_args.kwargs["source_conversation_id"] == "conv-1"
        assert ex.find_or_create_dm.await_args.kwargs["source_message_id"] == "m1"
        ex.send_message.assert_awaited_once()
        assert ex.send_message.await_args.args == ("dm-1", "details")
        assert _end_turn_calls(ex) == []

    @pytest.mark.asyncio
    async def test_routed_with_nothing_remaining_posts_hidden_redirect(self):
        ex = _executor()
        reply, blocks = _parse_dm_blocks('<dm target="Bob">details</dm>')
        out = await _settle(ex, reply, blocks)
        assert out is None
        assert ex.send_message.await_count == 2
        (end_turn,) = _end_turn_calls(ex)
        assert end_turn.args[0] == "conv-1"
        assert json.loads(end_turn.args[1]) == {
            "reason": "thread_redirect",
            "message": "[Continuing in DM with Bob]",
        }
        assert end_turn.kwargs["content_type"] == "structured"
        assert end_turn.kwargs["metadata"] == {
            "model": "m", "stream_id": "s1", "thread_redirect_ack_hidden": True,
        }
        assert end_turn.kwargs["last_seen_message_id"] == "m2"

    @pytest.mark.asyncio
    async def test_whitespace_only_remaining_counts_as_nothing(self):
        ex = _executor()
        out = await _settle(ex, "  \n ", [{"target": "Bob", "content": "x"}])
        assert out is None
        assert len(_end_turn_calls(ex)) == 1

    @pytest.mark.asyncio
    async def test_unknown_target_with_remaining_text_keeps_text(self):
        ex = _executor()
        reply, blocks = _parse_dm_blocks('Hmm. <dm target="Nobody">x</dm>')
        out = await _settle(ex, reply, blocks)
        assert out == "Hmm."
        ex.find_or_create_dm.assert_not_awaited()
        ex.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_target_with_nothing_remaining_posts_failure_notice(self):
        ex = _executor()
        reply, blocks = _parse_dm_blocks('<dm target="Nobody">x</dm><dm target="Ghost">y</dm>')
        out = await _settle(ex, reply, blocks)
        assert out == "[Could not start agent thread with Nobody, Ghost]"
        ex.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dm_open_failure_with_nothing_remaining_posts_failure_notice(self):
        ex = _executor(dm_id=None)  # find_or_create_dm returned no id
        out = await _settle(ex, "", [{"target": "Bob", "content": "x"}])
        assert out == "[Could not start agent thread with Bob]"
        ex.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_family_agents_from_directives_are_routable(self):
        ex = _executor()
        out = await _settle(
            ex, "", [{"target": "Cousin", "content": "x"}],
            directives={"familyAgents": [{"displayName": "Cousin", "participantId": "cousin-id"}]},
        )
        assert out is None
        assert ex.find_or_create_dm.await_args.args == ("cousin-id",)

    @pytest.mark.asyncio
    async def test_partial_routing_with_nothing_remaining_names_only_routed(self):
        ex = _executor()
        out = await _settle(
            ex, "", [{"target": "Bob", "content": "x"}, {"target": "Nobody", "content": "y"}]
        )
        assert out is None
        (end_turn,) = _end_turn_calls(ex)
        assert json.loads(end_turn.args[1])["message"] == "[Continuing in DM with Bob]"


class TestMultiTarget:
    """`target="A, B"` is ONE thread with everyone in it (bridge 2.9.6)."""

    @pytest.mark.asyncio
    async def test_comma_target_opens_one_thread_with_a_peer_list(self):
        ex = _executor()
        reply, blocks = _parse_dm_blocks('<dm target="Bob, Carol">three-way</dm>')
        assert blocks[0]["target"] == "Bob, Carol"  # kept raw at parse time
        out = await _settle(ex, reply, blocks)
        assert out is None
        ex.find_or_create_dm.assert_awaited_once()
        assert ex.find_or_create_dm.await_args.args == (["bob-id", "carol-id"],)
        ex.send_message.assert_any_await("dm-1", "three-way", metadata={"model": "m"})
        (end_turn,) = _end_turn_calls(ex)
        assert json.loads(end_turn.args[1])["message"] == "[Continuing in DM with Bob, Carol]"

    @pytest.mark.asyncio
    async def test_dedups_and_trims_names_case_insensitively(self):
        ex = _executor()
        out = await _settle(ex, "", [{"target": " bob ,Carol, BOB ", "content": "x"}])
        assert out is None
        assert ex.find_or_create_dm.await_args.args == (["bob-id", "carol-id"],)

    @pytest.mark.asyncio
    async def test_one_unresolvable_name_skips_the_whole_block(self):
        ex = _executor()
        out = await _settle(ex, "", [{"target": "Bob, Nobody", "content": "x"}])
        assert out == "[Could not start agent thread with Bob, Nobody]"
        ex.find_or_create_dm.assert_not_awaited()
        ex.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_name_still_calls_with_a_bare_id(self):
        ex = _executor()
        await _settle(ex, "", [{"target": "Bob", "content": "x"}])
        assert ex.find_or_create_dm.await_args.args == ("bob-id",)


class TestHiddenRedirectPayload:
    @pytest.mark.asyncio
    async def test_canonical_end_turn_json(self):
        ex = _executor()
        await _send_hidden_thread_redirect(
            ex, "conv-9", ["Bob", "Eve"], "exec-key",
            metadata={"model": "m"}, last_seen_message_id="m7",
        )
        ex.send_message.assert_awaited_once()
        call = ex.send_message.await_args
        assert call.args[0] == "conv-9"
        assert json.loads(call.args[1]) == {
            "reason": "thread_redirect",
            "message": "[Continuing in DM with Bob, Eve]",
        }
        assert call.kwargs["content_type"] == "structured"
        assert call.kwargs["message_type"] == "EndTurn"
        assert call.kwargs["metadata"] == {"model": "m", "thread_redirect_ack_hidden": True}
        assert call.kwargs["last_seen_message_id"] == "m7"

    @pytest.mark.asyncio
    async def test_server_template_is_honoured_inside_the_payload(self):
        ex = _executor()
        await _send_hidden_thread_redirect(
            ex, "conv-9", ["Bob"], "exec-key",
            behavioral_config={"dmRedirectTemplate": "[Now talking to {targets}]"},
        )
        assert json.loads(ex.send_message.await_args.args[1]) == {
            "reason": "thread_redirect",
            "message": "[Now talking to Bob]",
        }

    @pytest.mark.asyncio
    async def test_no_targets_posts_nothing(self):
        ex = _executor()
        await _send_hidden_thread_redirect(ex, "conv-9", [], "exec-key")
        ex.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_failure_is_swallowed(self):
        ex = _executor()
        ex.send_message = AsyncMock(side_effect=RuntimeError("down"))
        await _send_hidden_thread_redirect(ex, "conv-9", ["Bob"], "exec-key")
