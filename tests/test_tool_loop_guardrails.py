"""Tests for intra-turn tool-loop guardrails (agentchat/tools/guardrails.py)."""

import json

from agentchat.tools.guardrails import (
    ToolCallGuardrail,
    _looks_like_failure,
    _signature,
)


def _err(msg="boom"):
    return json.dumps({"error": msg})


def test_failure_detection():
    assert _looks_like_failure(_err()) is True
    assert _looks_like_failure('{"ok": true}') is False
    assert _looks_like_failure("") is False
    assert _looks_like_failure(None) is False
    assert _looks_like_failure("plain text") is False


def test_signature_is_order_independent():
    a = _signature("t", {"x": 1, "y": 2})
    b = _signature("t", {"y": 2, "x": 1})
    assert a == b
    assert _signature("t", {"x": 1}) != _signature("t", {"x": 2})
    assert _signature("t1", {}) != _signature("t2", {})


def test_exact_failure_warns_then_blocks():
    g = ToolCallGuardrail({"exactFailureWarnAfter": 2, "exactFailureBlockAfter": 3})
    args = {"q": "same"}

    # 1st failure: no warning yet, still allowed next time
    assert g.before_call("search", args).action == "allow"
    assert g.after_call("search", args, _err()).action == "allow"

    # 2nd failure: warn
    assert g.before_call("search", args).action == "allow"
    assert g.after_call("search", args, _err()).action == "warn"

    # 3rd failure still warns (>= warnAfter), reaches block count
    assert g.before_call("search", args).action == "allow"
    assert g.after_call("search", args, _err()).action == "warn"

    # 4th attempt is blocked before execution (exact count >= blockAfter)
    decision = g.before_call("search", args)
    assert decision.blocked is True
    assert "search" in decision.message


def test_success_resets_failure_counter():
    g = ToolCallGuardrail({"exactFailureWarnAfter": 2, "exactFailureBlockAfter": 3})
    args = {"q": "x"}
    g.after_call("search", args, _err())
    g.after_call("search", args, _err())
    # Success clears the streak
    g.after_call("search", args, '{"ok": 1}')
    # Now a fresh failure should not be at warn threshold
    assert g.after_call("search", args, _err()).action == "allow"


def test_idempotent_no_progress_warns_then_blocks():
    g = ToolCallGuardrail({
        "noProgressWarnAfter": 2,
        "noProgressBlockAfter": 3,
        "idempotentTools": ["list_emails"],
    })
    args = {"folder": "inbox"}
    same = json.dumps({"emails": [1, 2, 3]})

    assert g.after_call("list_emails", args, same).action == "allow"   # repeat 1
    assert g.after_call("list_emails", args, same).action == "warn"    # repeat 2
    g.after_call("list_emails", args, same)                            # repeat 3
    assert g.before_call("list_emails", args).blocked is True


def test_no_progress_only_applies_to_idempotent_tools():
    g = ToolCallGuardrail({
        "noProgressWarnAfter": 2,
        "idempotentTools": ["list_emails"],
    })
    args = {"to": "a@b.com"}
    same = json.dumps({"sent": True})
    # send_email is NOT idempotent: identical results never warn
    assert g.after_call("send_email", args, same).action == "allow"
    assert g.after_call("send_email", args, same).action == "allow"


def test_changed_result_resets_no_progress():
    g = ToolCallGuardrail({
        "noProgressWarnAfter": 2,
        "idempotentTools": ["search"],
    })
    args = {"q": "x"}
    g.after_call("search", args, json.dumps({"r": 1}))
    # Different result resets the repeat count
    assert g.after_call("search", args, json.dumps({"r": 2})).action == "allow"


def test_same_tool_failure_halts_across_args():
    g = ToolCallGuardrail({"sameToolFailureHaltAfter": 3, "exactFailureBlockAfter": 99})
    # Different args each time — exact-failure never trips, but same-tool does
    assert g.after_call("search", {"q": "a"}, _err()).action == "allow"
    assert g.after_call("search", {"q": "b"}, _err()).action == "allow"
    assert g.after_call("search", {"q": "c"}, _err()).action == "block"


def test_disabled_is_noop():
    g = ToolCallGuardrail({"enabled": False, "exactFailureBlockAfter": 1})
    args = {"q": "x"}
    g.after_call("search", args, _err())
    g.after_call("search", args, _err())
    assert g.before_call("search", args).action == "allow"


def test_per_turn_isolation():
    # Two separate instances (= two turns) share no state
    g1 = ToolCallGuardrail({"exactFailureBlockAfter": 2})
    g2 = ToolCallGuardrail({"exactFailureBlockAfter": 2})
    args = {"q": "x"}
    g1.after_call("search", args, _err())
    g1.after_call("search", args, _err())
    assert g1.before_call("search", args).blocked is True
    # g2 is a fresh turn — not blocked
    assert g2.before_call("search", args).blocked is False
