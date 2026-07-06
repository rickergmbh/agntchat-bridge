"""Regression coverage for wake-time toolkit refresh (issue #54).

A running bridge fetched `resolvedTools` exactly once at startup and froze it
for the whole process lifetime. Any tool seeded/enabled after boot — e.g.
`update_pulse` added by a later migration — was invisible: `execute_tool_calls`
rejects any name not in the frozen `resolved_tools` as "Unknown tool" before it
can reach the backend passthrough, so agents silently rotted behind the catalog
until a manual restart.

The fix re-fetches the profile on task/message pickup and rebuilds the toolkit.
`_diff_resolved_toolkit` is the pure decision at the heart of that path — it
partitions the fetched catalog (server_tool vs. dispatchable) and diffs it
against the loaded toolkit. These tests pin the diff semantics AND prove the
end-to-end contract: a tool that appears in a later fetch becomes dispatchable
without a restart.
"""

from __future__ import annotations

import pytest

from agent_bridge import _diff_resolved_toolkit, execute_tool_calls


def _tool(name: str, *, category: str | None = None, method: str | None = None) -> dict:
    t: dict = {
        "name": name,
        "description": f"{name} tool",
        "executorMethod": method or name,
        "inputSchema": {"type": "object", "properties": {}},
    }
    if category is not None:
        t["category"] = category
    return t


# ---------------------------------------------------------------------------
# _diff_resolved_toolkit — pure diff/partition
# ---------------------------------------------------------------------------

class TestDiffResolvedToolkit:
    def test_grown_catalog_reports_added_tool(self):
        current = [_tool("pulse_report")]
        fresh = [_tool("pulse_report"), _tool("update_pulse"), _tool("read_pulse")]

        diff = _diff_resolved_toolkit(fresh, current, current_server=[])

        assert diff is not None
        assert diff["added"] == ["read_pulse", "update_pulse"]
        assert diff["removed"] == []
        names = {t["name"] for t in diff["resolved"]}
        assert {"pulse_report", "update_pulse", "read_pulse"} == names

    def test_unchanged_catalog_returns_none(self):
        current = [_tool("send_message"), _tool("get_messages")]
        # Same names, different list identity/order — still a no-op.
        fresh = [_tool("get_messages"), _tool("send_message")]
        assert _diff_resolved_toolkit(fresh, current, current_server=[]) is None

    def test_empty_fetch_returns_none(self):
        # Transient blip — never clobber a live toolkit to empty.
        current = [_tool("send_message")]
        assert _diff_resolved_toolkit([], current, current_server=[]) is None

    def test_removed_tool_reported(self):
        current = [_tool("send_message"), _tool("deprecated_tool")]
        fresh = [_tool("send_message")]
        diff = _diff_resolved_toolkit(fresh, current, current_server=[])
        assert diff is not None
        assert diff["removed"] == ["deprecated_tool"]
        assert diff["added"] == []

    def test_server_tools_partitioned_out_of_resolved(self):
        current = [_tool("send_message")]
        fresh = [
            _tool("send_message"),
            _tool("web_search", category="server_tool"),
        ]
        diff = _diff_resolved_toolkit(fresh, current, current_server=[])
        assert diff is not None
        resolved_names = {t["name"] for t in diff["resolved"]}
        server_names = {t["name"] for t in diff["server"]}
        # web_search must NOT enter the dispatchable set — it stays server-only.
        assert resolved_names == {"send_message"}
        assert server_names == {"web_search"}

    def test_new_server_tool_alone_triggers_rebuild(self):
        # Same dispatchable names but a server tool appeared — the server-count
        # guard must catch it so server tool defs get rebuilt.
        current = [_tool("send_message")]
        fresh = [_tool("send_message"), _tool("web_search", category="server_tool")]
        diff = _diff_resolved_toolkit(fresh, current, current_server=[])
        assert diff is not None
        assert {t["name"] for t in diff["server"]} == {"web_search"}


# ---------------------------------------------------------------------------
# End-to-end: a tool that grows into the catalog becomes dispatchable
# ---------------------------------------------------------------------------

class FakeClient:
    """Backs `update_pulse`; records that dispatch actually reached it."""

    def __init__(self) -> None:
        self.called_with: dict | None = None

    async def update_pulse(self, **kwargs):
        self.called_with = kwargs
        return {"ok": True}


@pytest.mark.asyncio
async def test_new_tool_is_undispatchable_before_refresh_and_dispatchable_after():
    """The core issue-#54 contract, exercised through the real dispatch gate.

    Simulates a profile whose resolvedTools grows between fetches (update_pulse
    seeded after boot) and asserts the new tool flips from "Unknown tool" to
    dispatched — no process restart.
    """
    client = FakeClient()

    # Toolkit frozen at boot — update_pulse not yet seeded.
    booted = [_tool("pulse_report")]

    # Before refresh: the gate rejects update_pulse outright.
    before = await execute_tool_calls(
        client, [{"name": "update_pulse", "arguments": {"enabled": True}}],
        resolved_tools=booted,
    )
    assert before[0].get("error") == "Unknown tool: update_pulse"
    assert client.called_with is None

    # A later profile fetch carries the newly-seeded tool.
    fresh = [_tool("pulse_report"), _tool("update_pulse")]
    diff = _diff_resolved_toolkit(fresh, booted, current_server=[])
    assert diff is not None
    refreshed = diff["resolved"]

    # After refresh: same call now dispatches to the executor, no restart.
    after = await execute_tool_calls(
        client, [{"name": "update_pulse", "arguments": {"enabled": True}}],
        resolved_tools=refreshed,
    )
    assert "error" not in after[0]
    assert after[0]["result"] == {"ok": True}
    assert client.called_with == {"enabled": True}
