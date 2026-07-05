"""CLI token-usage extraction — feeds ModelResult.usage → POST /api/usage.

Without these helpers the entire claude_cli backend reports no token usage
(the platform's model-usage analytics stayed empty for weeks before anyone
noticed), so pin the extraction semantics.
"""

from agentchat.backends.claude_cli import _extract_cli_usage, _merge_usage


def test_extracts_known_keys_from_result_event():
    event = {
        "type": "result",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 900,
            "server_tool_use": {"web_search_requests": 0},  # ignored
        },
    }
    assert _extract_cli_usage(event) == {
        "input_tokens": 100,
        "output_tokens": 25,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 900,
    }


def test_missing_or_malformed_usage_is_none():
    assert _extract_cli_usage({"type": "result"}) is None
    assert _extract_cli_usage({"usage": "n/a"}) is None
    assert _extract_cli_usage({"usage": {}}) is None
    assert _extract_cli_usage({"usage": {"input_tokens": "many"}}) is None


def test_merge_accumulates_across_iterations():
    total = None
    total = _merge_usage(total, {"input_tokens": 10, "output_tokens": 5})
    total = _merge_usage(total, None)
    total = _merge_usage(total, {"input_tokens": 7, "cache_read_input_tokens": 3})
    assert total == {"input_tokens": 17, "output_tokens": 5, "cache_read_input_tokens": 3}
