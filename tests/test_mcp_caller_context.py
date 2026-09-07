"""`/api/mcp` JSON-RPC `tools/call` carries the caller context as
`params._meta.context` (2026-09-06 coordination audit, bridge follow-up 1).

The backend (`mcp_controller.ex` tools/call -> `ToolRegistry
.build_caller_context/1`) keeps non-empty strings under exactly
conversation_id / source_message_id / task_id / active_conversation_id /
last_seen_message_id — the same whitelist `/api/mcp/call` accepts. These
pin the wire shape with and without context, and that every `/api/mcp`
poster goes through the one helper.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentchat.errors import AgentChatError
from agentchat.executor import (
    MCP_CALLER_CONTEXT_KEYS,
    ExecutorClient,
    GatewayTask,
    _mcp_caller_context,
)


@pytest.fixture
def executor(base_url, agent_id, api_key):
    client = ExecutorClient(base_url, agent_id, api_key, "test-executor")
    client._executor_id = "executor-1"
    return client


def _ok(text: str = "ok") -> dict:
    return {"result": {"isError": False, "content": [{"type": "text", "text": text}]}}


def _rpc_body(post: AsyncMock) -> dict:
    """The JSON body of the single /api/mcp POST."""
    calls = [c for c in post.await_args_list if c.args[0] == "/api/mcp"]
    assert len(calls) == 1, post.await_args_list
    return calls[0].kwargs["json"]


class TestWhitelist:
    def test_matches_backend_whitelist(self):
        assert MCP_CALLER_CONTEXT_KEYS == (
            "conversation_id",
            "source_message_id",
            "task_id",
            "active_conversation_id",
            "last_seen_message_id",
        )

    def test_drops_none_empty_and_unknown_keys(self):
        assert _mcp_caller_context(
            {
                "conversation_id": "c1",
                "task_id": None,
                "source_message_id": "",
                "last_seen_message_id": "m9",
                "bogus": "x",
                "active_conversation_id": 42,
            }
        ) == {"conversation_id": "c1", "last_seen_message_id": "m9"}

    def test_empty_inputs(self):
        assert _mcp_caller_context(None) == {}
        assert _mcp_caller_context({}) == {}


class TestCallPlatformTool:
    @pytest.mark.asyncio
    async def test_bare_params_without_context(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok())) as post:
            await executor._call_platform_tool("complete_task", {"task_id": "t1"})
        assert _rpc_body(post) == {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "complete_task", "arguments": {"task_id": "t1"}},
        }

    @pytest.mark.asyncio
    async def test_context_rides_in_meta(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok())) as post:
            await executor._call_platform_tool(
                "complete_task",
                {"task_id": "t1"},
                context={"task_id": "t1", "conversation_id": "c1", "bogus": "x", "source_message_id": None},
            )
        assert _rpc_body(post)["params"] == {
            "name": "complete_task",
            "arguments": {"task_id": "t1"},
            "_meta": {"context": {"task_id": "t1", "conversation_id": "c1"}},
        }

    @pytest.mark.asyncio
    async def test_all_none_context_sends_no_meta(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok())) as post:
            await executor._call_platform_tool(
                "fail_task", {"task_id": "t1"}, context={"conversation_id": None}
            )
        assert "_meta" not in _rpc_body(post)["params"]

    @pytest.mark.asyncio
    async def test_is_error_still_raises(self, executor):
        body = {"result": {"isError": True, "content": [{"type": "text", "text": "nope"}]}}
        with patch.object(executor, "_post", new=AsyncMock(return_value=body)):
            with pytest.raises(AgentChatError, match="nope"):
                await executor._call_platform_tool("complete_task", {}, context={"task_id": "t1"})


class TestCompleteAndFailTask:
    @pytest.mark.asyncio
    async def test_invoke_complete_task_always_includes_task_id(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok())) as post:
            await executor._invoke_complete_task("task-1", response_text="done", context={"conversation_id": "conv-w"})
        params = _rpc_body(post)["params"]
        assert params["name"] == "complete_task"
        assert params["arguments"] == {"task_id": "task-1", "silent": False, "response": "done"}
        assert params["_meta"] == {"context": {"conversation_id": "conv-w", "task_id": "task-1"}}

    @pytest.mark.asyncio
    async def test_invoke_fail_task_without_context_still_sends_task_id(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok())) as post:
            await executor._invoke_fail_task("task-2", error_text="boom")
        params = _rpc_body(post)["params"]
        assert params["name"] == "fail_task"
        assert params["_meta"] == {"context": {"task_id": "task-2"}}

    @pytest.mark.asyncio
    async def test_public_complete_task_forwards_conversation_kwargs(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok())) as post:
            await executor.complete_task(
                "task-3", response="hi", conversation_id="conv-a", active_conversation_id="conv-a"
            )
        assert _rpc_body(post)["params"]["_meta"]["context"] == {
            "conversation_id": "conv-a",
            "active_conversation_id": "conv-a",
            "task_id": "task-3",
        }

    @pytest.mark.asyncio
    async def test_public_fail_task_without_conversation_only_task_id(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok())) as post:
            await executor.fail_task("task-4", error="x")
        assert _rpc_body(post)["params"]["_meta"] == {"context": {"task_id": "task-4"}}


class TestTaskHandlerContext:
    """`_handle_task` knows the work conversation — it must ride along."""

    @pytest.mark.asyncio
    async def test_completion_carries_work_conversation(self, executor):
        @executor.on_task
        async def handler(_task):
            return "deliverable"

        task = GatewayTask(
            id="queue-1",
            task_id="task-1",
            title="T",
            conversation_id="conv-parent",
            work_conversation_id="conv-work",
        )
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok())) as post:
            await executor._handle_task(task)
        params = _rpc_body(post)["params"]
        assert params["name"] == "complete_task"
        assert params["_meta"]["context"] == {
            "conversation_id": "conv-work",
            "active_conversation_id": "conv-work",
            "task_id": "task-1",
        }

    @pytest.mark.asyncio
    async def test_failure_falls_back_to_task_conversation(self, executor):
        @executor.on_task
        async def handler(_task):
            raise RuntimeError("kaboom")

        task = GatewayTask(id="queue-2", task_id="task-2", title="T", conversation_id="conv-parent")
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok())) as post:
            await executor._handle_task(task)
        params = _rpc_body(post)["params"]
        assert params["name"] == "fail_task"
        assert params["_meta"]["context"] == {
            "conversation_id": "conv-parent",
            "active_conversation_id": "conv-parent",
            "task_id": "task-2",
        }


class TestOtherPosters:
    @pytest.mark.asyncio
    async def test_search_memory_passes_conversation_as_context(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok("hits"))) as post:
            out = await executor.search_memory("q", conversation_id="conv-1")
        assert out == "hits"
        params = _rpc_body(post)["params"]
        assert params["name"] == "memory_search"
        assert params["arguments"] == {"query": "q", "scope": "all", "conversation_id": "conv-1"}
        assert params["_meta"] == {"context": {"conversation_id": "conv-1"}}

    @pytest.mark.asyncio
    async def test_search_memory_without_conversation_sends_no_meta(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok("hits"))) as post:
            await executor.search_memory("q")
        assert "_meta" not in _rpc_body(post)["params"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method, tool, kwargs",
        [
            ("search_jobs_adzuna", "search_jobs_adzuna", {"query": "dev"}),
            ("search_jobs_google", "search_jobs_google", {"query": "dev"}),
            ("get_salary_data", "get_salary_data", {"query": "dev"}),
            ("search_jobs_theirstack", "search_jobs_theirstack", {"job_title_or": ["dev"]}),
        ],
    )
    async def test_job_posters_accept_context(self, executor, method, tool, kwargs):
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok("{}"))) as post:
            await getattr(executor, method)(**kwargs)
            assert "_meta" not in _rpc_body(post)["params"]
            assert _rpc_body(post)["params"]["name"] == tool
        with patch.object(executor, "_post", new=AsyncMock(return_value=_ok("{}"))) as post:
            await getattr(executor, method)(**kwargs, context={"conversation_id": "c1"})
            assert _rpc_body(post)["params"]["_meta"] == {"context": {"conversation_id": "c1"}}

    @pytest.mark.asyncio
    async def test_empty_result_default(self, executor):
        with patch.object(executor, "_post", new=AsyncMock(return_value={"result": {}})):
            assert await executor.search_jobs_google("dev") == "{}"
            assert await executor.search_memory("q") == ""
