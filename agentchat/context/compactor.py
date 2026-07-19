"""In-turn context compaction for long tool-use loops.

A "dumb pipe" executor of the policy the backend ships in
``behavioralConfig.compaction``. When a tool-use turn's message list grows
toward the model's context window, this compacts the MIDDLE of the
conversation into a structured checkpoint summary while protecting the head
(first user turn) and a token-budgeted tail of recent messages — never
splitting an assistant tool_use from its following tool_result.

Three passes (Hermes-style):
  1. prune — replace bulky old tool_result blocks with 1-line digests
  2. protect — keep head + a token-budget tail verbatim
  3. summarize — fold the middle into one summary message (iteratively updated
     across repeated compactions)

There is also an EARLY tier below the full-compaction trigger: a TTL-gated
soft-trim of old tool_result payloads (keep a head/tail excerpt, elide the
middle). Any byte changed in the message list invalidates the provider's
cached prefix from that point on, so the early tier only fires when the
prompt cache has already gone cold — i.e. the gap since the previous API
call in this loop exceeds ``cacheTtlSeconds`` (a tool that ran longer than
the TTL: computer use, long execs). While the cache is warm, trimming would
cost a full prefix re-read to save a few KB — never worth it. The full
compactor at ``triggerRatio`` stays unconditional: at that point the
context window itself is at risk, which beats any cache economics.

The summary TEMPLATE and anti-re-answer PREFIX come from the server config; the
bridge only decides *mechanics* (what fits the token budget), never *wording*.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger("agentchat.context.compactor")

# Rough chars-per-token; matches the heuristic used elsewhere in the bridge.
_CHARS_PER_TOKEN = 4
# Old tool_result blocks larger than this (chars) get digested during prune.
_TOOL_RESULT_DIGEST_OVER = 600

_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "triggerRatio": 0.5,
    "tailTokenBudget": 20_000,
    "minTailMessages": 4,
    "maxSummaryTokens": 2_000,
    # Early tier (TTL-gated soft-trim). earlyPruneRatio is a fraction of the
    # input budget, below triggerRatio; cacheTtlSeconds matches Anthropic's
    # default 5m cache TTL (server config overrides for 1h-TTL models).
    "earlyPruneRatio": 0.3,
    "cacheTtlSeconds": 300,
    "softTrimOver": 4_000,
    "softTrimHeadChars": 1_200,
    "softTrimTailChars": 300,
    "summaryPrefix": "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted below; respond only to messages after this summary.",
    "summaryPrompt": "Summarize the conversation so far as a structured checkpoint.",
    "summaryUpdatePrompt": "Update the existing summary with the new turns, preserving still-relevant detail.",
}


def _text_len(content: Any) -> int:
    """Approximate character length of a message's content (str or blocks)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                # tool_result / text blocks carry their payload under "content"
                # or "text"; tool_use carries "input".
                for key in ("text", "content", "input"):
                    val = block.get(key)
                    if isinstance(val, str):
                        total += len(val)
                    elif val is not None:
                        total += len(str(val))
            else:
                total += len(str(block))
        return total
    return len(str(content)) if content is not None else 0


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate for a message list (content + small per-msg overhead)."""
    return sum(_text_len(m.get("content")) // _CHARS_PER_TOKEN + 8 for m in messages)


def _has_tool_use(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
    return False


def _has_tool_result(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


class ContextCompactor:
    """Compacts ``api_messages`` in place-style (returns a new list) per turn."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        context_window: int,
        max_output_tokens: int,
    ) -> None:
        cfg = dict(_DEFAULTS)
        if isinstance(config, dict):
            cfg.update({k: v for k, v in config.items() if v is not None})
        self._cfg = cfg
        self._enabled = bool(cfg.get("enabled", True)) and context_window > 0
        self._context_window = context_window
        # Reserve output space; the input budget is what's left.
        self._input_budget = max(1, context_window - max(0, max_output_tokens))
        self._trigger_tokens = int(self._input_budget * float(cfg.get("triggerRatio", 0.5)))
        self._early_tokens = int(self._input_budget * float(cfg.get("earlyPruneRatio", 0.3)))
        # The running summary, carried across repeated compactions this turn.
        self._previous_summary: str | None = None
        # Monotonic timestamp of the loop's most recent API call — the
        # TTL gate for the early tier. None until the first call completes.
        self._last_request_at: float | None = None

    def should_compact(self, messages: list[dict[str, Any]]) -> bool:
        if not self._enabled:
            return False
        return estimate_tokens(messages) >= self._trigger_tokens

    # ---- early tier (TTL-gated soft-trim) --------------------------------

    def note_request(self) -> None:
        """Record that an API call just went out (resets the TTL clock)."""
        self._last_request_at = time.monotonic()

    def _cache_cold(self) -> bool:
        if self._last_request_at is None:
            # First iteration: the previous turn's prefix may still be warm
            # and we can't tell — don't risk busting it.
            return False
        ttl = float(self._cfg.get("cacheTtlSeconds", 300))
        return (time.monotonic() - self._last_request_at) >= ttl

    def should_early_prune(self, messages: list[dict[str, Any]]) -> bool:
        """True when the context has real bulk AND the cache is already cold,
        so trimming is free (the next call re-reads the prefix either way)."""
        if not self._enabled or not self._cache_cold():
            return False
        return estimate_tokens(messages) >= self._early_tokens

    def early_prune(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Soft-trim bulky tool_result payloads outside the protected tail:
        keep a head/tail excerpt, elide the middle. Gentler than the full
        compactor's 1-line digest — recent-ish detail stays usable."""
        over = int(self._cfg.get("softTrimOver", 4_000))
        head_chars = int(self._cfg.get("softTrimHeadChars", 1_200))
        tail_chars = int(self._cfg.get("softTrimTailChars", 300))

        _, tail_start = self._partition(messages)
        out: list[dict[str, Any]] = []
        trimmed = 0
        for idx, msg in enumerate(messages):
            if idx >= tail_start or not isinstance(msg.get("content"), list):
                out.append(msg)
                continue
            new_content = []
            for block in msg["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and len(block["content"]) > over
                ):
                    body = block["content"]
                    elided = len(body) - head_chars - tail_chars
                    new_block = dict(block)
                    new_block["content"] = (
                        body[:head_chars]
                        + f"\n…[{elided} chars trimmed]…\n"
                        + body[-tail_chars:]
                    )
                    new_content.append(new_block)
                    trimmed += 1
                else:
                    new_content.append(block)
            out.append({**msg, "content": new_content})

        if trimmed:
            logger.info(
                "Early-pruned %d tool result(s) (~%d→%d tokens, cache cold)",
                trimmed, estimate_tokens(messages), estimate_tokens(out),
            )
        return out

    async def compact(
        self,
        messages: list[dict[str, Any]],
        summarize: Callable[[str, str], Awaitable[str | None]],
    ) -> list[dict[str, Any]]:
        """Return a compacted copy of ``messages``.

        ``summarize(system_prompt, user_content)`` performs the LLM summary call
        and returns the summary text (or None on failure — we then fall back to
        a no-LLM structural prune so the turn can still proceed).
        """
        if not self._enabled or len(messages) <= self._cfg["minTailMessages"] + 1:
            return messages

        # Pass 1: cheap structural prune of old tool_result payloads.
        pruned = self._prune_tool_results(messages)
        if estimate_tokens(pruned) < self._trigger_tokens:
            return pruned

        # Pass 2: choose head + token-budget tail; the middle gets summarized.
        head_end, tail_start = self._partition(pruned)
        if tail_start <= head_end:
            # Nothing safely summarizable (everything is head/tail) — return prune.
            return pruned

        head = pruned[:head_end]
        middle = pruned[head_end:tail_start]
        tail = pruned[tail_start:]

        # Pass 3: summarize the middle (iteratively updating any prior summary).
        transcript = self._render_for_summary(middle)
        is_update = self._previous_summary is not None
        system_prompt = (
            self._cfg["summaryUpdatePrompt"] if is_update else self._cfg["summaryPrompt"]
        )
        user_content = (
            f"PREVIOUS SUMMARY:\n{self._previous_summary}\n\nNEW TURNS:\n{transcript}"
            if is_update
            else transcript
        )

        summary_text = None
        try:
            summary_text = await summarize(system_prompt, user_content)
        except Exception as e:  # noqa: BLE001 — summary failure must never break the turn
            logger.warning("Compaction summary call failed: %s", e)

        if not summary_text:
            # LLM unavailable — fall back to the structural prune (still helps).
            logger.info("Compaction summary unavailable; using structural prune only")
            return pruned

        self._previous_summary = summary_text
        summary_msg = {
            "role": "user",
            "content": f"{self._cfg['summaryPrefix']}\n\n{summary_text}",
        }

        compacted = head + [summary_msg] + tail
        logger.info(
            "Compacted %d→%d messages (~%d→%d tokens)",
            len(messages), len(compacted),
            estimate_tokens(messages), estimate_tokens(compacted),
        )
        return compacted

    # ---- passes ----------------------------------------------------------

    def _prune_tool_results(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace bulky tool_result payloads (outside the protected tail) with
        a 1-line digest. Cheap, no LLM. Keeps block structure intact."""
        _, tail_start = self._partition(messages)
        out: list[dict[str, Any]] = []
        for idx, msg in enumerate(messages):
            if idx >= tail_start or not isinstance(msg.get("content"), list):
                out.append(msg)
                continue
            new_content = []
            for block in msg["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and isinstance(block.get("content"), str)
                    and len(block["content"]) > _TOOL_RESULT_DIGEST_OVER
                ):
                    digest = block["content"][:200].replace("\n", " ")
                    new_block = dict(block)
                    new_block["content"] = f"[tool result digested: {digest}… ({len(block['content'])} chars)]"
                    new_content.append(new_block)
                else:
                    new_content.append(block)
            out.append({**msg, "content": new_content})
        return out

    def _partition(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """Return (head_end, tail_start) indices.

        head = messages[:head_end] (the first user turn, kept verbatim)
        tail = messages[tail_start:] (recent messages within the token budget)
        Guarantees tail_start never lands between a tool_use and its tool_result.
        """
        n = len(messages)
        # Head: keep the first message (and a second if the first is a bare
        # "(conversation start)" filler), so the model retains the original ask.
        head_end = 1

        # Tail: walk back accumulating tokens until the budget is hit, but keep
        # at least minTailMessages.
        budget = self._cfg["tailTokenBudget"]
        min_tail = self._cfg["minTailMessages"]
        acc = 0
        tail_start = n
        for i in range(n - 1, head_end - 1, -1):
            acc += _text_len(messages[i].get("content")) // _CHARS_PER_TOKEN + 8
            kept = n - i
            if acc > budget and kept >= min_tail:
                tail_start = i + 1
                break
            tail_start = i

        # Never start the tail on a tool_result whose tool_use sits in the
        # middle — that would orphan the result. Push the boundary back to
        # include the preceding assistant tool_use message.
        if 0 < tail_start < n and _has_tool_result(messages[tail_start]):
            if tail_start - 1 >= head_end and _has_tool_use(messages[tail_start - 1]):
                tail_start -= 1

        return head_end, tail_start

    def _render_for_summary(self, messages: list[dict[str, Any]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            else:
                parts = []
                for block in content if isinstance(content, list) else []:
                    if not isinstance(block, dict):
                        parts.append(str(block))
                    elif block.get("type") == "tool_use":
                        parts.append(f"[called {block.get('name')}({_short(block.get('input'))})]")
                    elif block.get("type") == "tool_result":
                        parts.append(f"[result: {_short(block.get('content'))}]")
                    elif block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    else:
                        parts.append(_short(block))
                text = " ".join(p for p in parts if p)
            lines.append(f"{role.upper()}: {text[:2000]}")
        return "\n".join(lines)[:24_000]


def _short(value: Any, limit: int = 300) -> str:
    s = value if isinstance(value, str) else str(value)
    return s[:limit]
