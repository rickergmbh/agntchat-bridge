"""Intra-turn tool-loop guardrails.

A "dumb pipe" executor of the policy the backend ships in
``behavioralConfig.toolLoopGuardrails``. It detects, within a single turn:

  * exact-failure   — identical ``(tool, args)`` call fails repeatedly
  * same-tool       — one tool fails many times this turn (any args)
  * no-progress     — an idempotent (read-only) tool returns the same result

and returns warn/block decisions. It makes NO policy judgment of its own; all
thresholds come from the server config. Counters are per-turn: instantiate one
``ToolCallGuardrail`` per ``chat_with_tools`` invocation.

This complements (does not replace) the backend's conversational LoopPrevention,
which guards agent reply chains across messages rather than a single tool loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Mirror of Agentchat.Agents.ToolLoopGuardrails.defaults/0. Used only when the
# server sends no config (older backend, or guardrails disabled upstream).
_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "exactFailureWarnAfter": 2,
    "exactFailureBlockAfter": 5,
    "sameToolFailureHaltAfter": 8,
    "noProgressWarnAfter": 2,
    "noProgressBlockAfter": 5,
    "idempotentTools": [],
    "warnMessage": (
        "Heads up: {tool} has returned the same result / failed {n}× this turn. "
        "Re-running it unchanged won't help — change the arguments or approach."
    ),
    "blockMessage": (
        "Blocked {tool}: it failed or returned the same result {n}× this turn with no "
        "progress. Stop repeating it unchanged. Use what you already have or change strategy."
    ),
}


@dataclass
class GuardrailDecision:
    """Outcome of a before/after check."""

    action: str = "allow"  # "allow" | "warn" | "block"
    message: str = ""
    code: str = ""

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    @property
    def has_message(self) -> bool:
        return bool(self.message) and self.action in ("warn", "block")


def _looks_like_failure(result_str: str | None) -> bool:
    """Detect a failed tool result.

    ToolExecutor.execute never raises and encodes failures as a JSON object with
    a top-level ``error`` key (see executor.py). We also treat empty results as
    non-failures (a tool legitimately returning nothing isn't a loop signal).
    """
    if not result_str:
        return False
    s = result_str.lstrip()
    if not s.startswith("{"):
        return False
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict) and "error" in parsed and len(parsed) <= 3


def _signature(tool_name: str, args: dict[str, Any] | None) -> str:
    """Stable hash of a (tool, args) pair — canonical, order-independent."""
    try:
        canonical = json.dumps(args or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canonical = repr(args)
    return hashlib.sha256(f"{tool_name}\x00{canonical}".encode()).hexdigest()


def _result_hash(result_str: str | None) -> str:
    return hashlib.sha256((result_str or "").encode()).hexdigest()


class ToolCallGuardrail:
    """Per-turn guardrail controller. One instance per tool-use turn."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(_DEFAULTS)
        if isinstance(config, dict):
            cfg.update({k: v for k, v in config.items() if v is not None})
        self._cfg = cfg
        self._enabled: bool = bool(cfg.get("enabled", True))
        self._idempotent: set[str] = set(cfg.get("idempotentTools") or [])

        # signature -> consecutive failure count
        self._exact_failures: dict[str, int] = {}
        # tool_name -> failure count this turn (any args)
        self._same_tool_failures: dict[str, int] = {}
        # signature -> (result_hash, repeat_count) for idempotent no-progress
        self._no_progress: dict[str, tuple[str, int]] = {}

    # ---- decisioning -----------------------------------------------------

    def before_call(self, tool_name: str, args: dict[str, Any] | None) -> GuardrailDecision:
        """Gate a call BEFORE execution. Returns a block decision to skip it."""
        if not self._enabled:
            return GuardrailDecision()

        sig = _signature(tool_name, args)

        exact = self._exact_failures.get(sig, 0)
        if exact >= self._cfg["exactFailureBlockAfter"]:
            return self._block(tool_name, exact, "exact_failure_block")

        if tool_name in self._idempotent:
            record = self._no_progress.get(sig)
            if record and record[1] >= self._cfg["noProgressBlockAfter"]:
                return self._block(tool_name, record[1], "no_progress_block")

        return GuardrailDecision()

    def after_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None,
        result_str: str | None,
        *,
        failed: bool | None = None,
    ) -> GuardrailDecision:
        """Record the outcome AFTER execution and return any warn decision."""
        if not self._enabled:
            return GuardrailDecision()

        sig = _signature(tool_name, args)
        if failed is None:
            failed = _looks_like_failure(result_str)

        if failed:
            exact = self._exact_failures.get(sig, 0) + 1
            self._exact_failures[sig] = exact
            self._no_progress.pop(sig, None)

            same = self._same_tool_failures.get(tool_name, 0) + 1
            self._same_tool_failures[tool_name] = same

            if same >= self._cfg["sameToolFailureHaltAfter"]:
                return self._block(tool_name, same, "same_tool_halt")
            if exact >= self._cfg["exactFailureWarnAfter"]:
                return self._warn(tool_name, exact, "exact_failure_warn")
            return GuardrailDecision()

        # Success — clear failure bookkeeping for this signature/tool.
        self._exact_failures.pop(sig, None)
        self._same_tool_failures.pop(tool_name, None)

        if tool_name not in self._idempotent:
            self._no_progress.pop(sig, None)
            return GuardrailDecision()

        # Idempotent success: track repeated identical results (no progress).
        rhash = _result_hash(result_str)
        prev = self._no_progress.get(sig)
        repeat = prev[1] + 1 if prev and prev[0] == rhash else 1
        self._no_progress[sig] = (rhash, repeat)

        if repeat >= self._cfg["noProgressWarnAfter"]:
            return self._warn(tool_name, repeat, "no_progress_warn")
        return GuardrailDecision()

    # ---- message builders ------------------------------------------------

    def _warn(self, tool: str, n: int, code: str) -> GuardrailDecision:
        msg = str(self._cfg.get("warnMessage", _DEFAULTS["warnMessage"]))
        return GuardrailDecision(action="warn", code=code, message=self._fmt(msg, tool, n))

    def _block(self, tool: str, n: int, code: str) -> GuardrailDecision:
        msg = str(self._cfg.get("blockMessage", _DEFAULTS["blockMessage"]))
        logger.warning("Tool-loop guardrail BLOCK (%s): %s ×%d", code, tool, n)
        return GuardrailDecision(action="block", code=code, message=self._fmt(msg, tool, n))

    @staticmethod
    def _fmt(template: str, tool: str, n: int) -> str:
        try:
            return template.format(tool=tool, n=n)
        except (KeyError, IndexError, ValueError):
            return template
