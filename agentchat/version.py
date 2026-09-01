"""Bridge protocol version — leaf module so any part of the package can
import it without cycles.

Reported to the backend at executor registration and WS gateway join
(audit-remediation-plan H3). The backend compares it against
Agentchat.Protocol's @min_bridge_version and refuses/flags outdated bridges
once enforcement is on. Bump on any protocol-relevant change (payload fields
the backend requires, structured-tag semantics, fail-loud contracts).
Distinct from the ACP message-envelope schema_version ("2.0") — they version
different things.
"""

# 2.2.0 — task-request sequencing moved server-side (H4 item 4, issue #86):
# the bridge submits parsed <task_request> blocks to
# POST /api/gateway/task-requests instead of running the orchestrator
# scope→create flow and default-assignee policy locally.
# 2.3.0 — compound-task DAG walk moved server-side (H4 item 4 flows 2–3):
# the bridge loops on POST /api/gateway/tasks/:id/claim-step and only
# runs the per-step LLM; the dead memory_flush handler (no producer,
# zero tasks ever created) was deleted outright.
# 2.4.0 — prompt-cache restructure: per-turn-fresh blocks (temporal
# context, live presence, speaking order) moved out of promptDirectives
# into directives.volatileContext, which the bridge appends to the USER
# turn. A 2.3.x bridge on a 2.4 backend would silently lose those blocks.
# 2.4.1 — cross-turn history cache continuity: cache breakpoint pinned at
# the stable-history boundary (per-turn tail — volatile context, trigger
# echo, identity anchor — moved after it), anchored non-sliding history
# window in _cached_get_messages, and trigger echo deduped when it already
# rendered as the newest history message. Bridge-internal (no backend
# payload change), but listed for fleet-roll tracking.
# 2.5.0 — per-turn model override consumed: the bridge now applies
# task metadata `model_override` (stamped by PulseExecutionWorker from
# pulse config `model`) to every LLM call in that task's turn via the
# MODEL_OVERRIDE contextvar; backends resolve it at request time
# (_request_model). Older bridges silently ran pulses on the agent's
# static model.
# 2.6.0 — humanlike bubble delivery moved fully server-side (audit
# Theme 5.3): the bridge posts its raw <msg>-tagged reply ONCE; the
# backend (HumanlikeDelivery + StaggeredBubbleWorker) owns splitting,
# pacing, humanlike_bubble metadata, and peer-wake routing. Older
# bridges split client-side and make their own peer-wake routing
# decision, so WS/SDK agents and bridge agents diverge; the
# behavioralConfig.humanlikePacing key they read is gone (they fall
# back to local defaults, harmless during the roll).
# 2.6.1 — open-subtask completion rejection treated as a wait signal:
# when the backend's complete_task guard answers with the
# "[open_subtasks]" marker, _handle_task leaves the task open for the
# sub-task-completion wake instead of failing the task. Older bridges
# fail the root task and strand the sub-tasks' output (Morning Brief
# 2026-08-12). Backend marker ships in the same change; safe either
# order — without the marker the old failure mode simply persists.
# 2.7.0 — runnability reporting: the bridge preflights its model backend
# before registering and sends `backend_health` {status, detail} on both
# the register and heartbeat payloads, plus `bridge_version` on heartbeat.
# The server (Agentchat.Agents.Runnability) gates agent presence on it, so
# a machine with no `claude login` now reads offline-with-a-reason and gets
# an in-chat explanation instead of a green dot and permanent silence.
# claude_cli additionally classifies auth-shaped CLI failures as
# BackendAuthError and flips its own health, so a mid-session credential
# loss (or recovery) surfaces within one heartbeat. Older bridges simply
# don't report health — they're treated as healthy on the version check
# alone, so the roll is safe in either order.
# 2.7.1 — preflight probes the Bedrock/Vertex credential chain instead of
# assuming it's configured. The exemption meant a cloud-connection agent on
# a machine with no AWS/GCP chain reported `ok` forever: every turn died on
# "Could not load credentials from any providers", and every restart re-ran
# preflight and laundered the dead state back to green, so it kept reading
# online and kept getting handed tasks. Structural probe only (no network,
# so an instance-role-only machine now reads unauthenticated — the safe
# direction). Not gated on by the server; the roll is safe in either order.
# 2.7.2 — audit hardening: missing agentgram_mcp_server.py is a hard error
# for claude_cli tool use (no silent XML-loop degrade), the stale repo
# scripts/ fallback paths were removed from every script lookup, WS event
# handlers hold strong task refs (GC could silently drop events parked on
# the semaphore), the dead batch_complete_tasks stub was deleted, and
# heartbeat/location failure paths now log. Bridge-internal; not gated on
# by the server, safe to roll in either order.
# 2.7.4 — subscription preflight no longer counts a bare ~/.claude.json as
# a credential: it's the CLI's config file, written on any first launch,
# so a fresh machine that never ran `claude login` read `ok` forever and
# the first-run onboarding "greeting" step stalled with no warning. The
# probe now looks for account markers (oauthAccount / primaryApiKey keys
# only, never values) and recognises CLAUDE_CODE_OAUTH_TOKEN. Not gated on
# by the server; safe to roll in either order.
# 2.7.5 — turn-time auth failures reach the user: _AUTH_FAILURE_RE learns
# the CLI's "Failed to authenticate" / "OAuth session expired" phrasings
# (previously only "authentication failed" / "oauth token expired", so an
# expired login posted the generic modelFailure apology and never flipped
# health), and BackendAuthError turns now reply with the server's
# errorMessages.authFailure copy instead of the generic fallback. Pairs
# with the backend adding that key; older backends fall back to a built-in
# string, safe to roll in either order.
# 2.7.6 — authFailure replies carry metadata.errorKind="auth_failure" so
# clients can render a one-click fix (the desktop's "Sign in to Claude"
# button) under the error bubble instead of leaving the user to parse the
# copy. Pure metadata addition; clients that don't know the key ignore it
# and the server stores it as-is, safe to roll in either order.
# 2.7.7 — bedrock/vertex auth failures are no longer silent. Two bugs, one
# incident (Jarvis/Bedrock, Aug 2026): an expired AWS SSO session's error
# text ("Token is expired. To refresh this SSO session run 'aws sso login'
# ...") didn't match _AUTH_FAILURE_RE at all, so it fell through to a plain
# RuntimeError — health never flipped, and the turn eventually posted the
# generic "I ran into an issue" apology ~3 minutes later (the CLI's own AWS
# auth-refresh timeout) with zero indication of what actually broke. Even
# once classified correctly, the existing authFailure copy ("sign in to
# Claude") is wrong for a cloud-authenticated agent. Fixed: the regex learns
# AWS SSO / GCP ADC phrasings, and _model_failure_reply now branches on the
# backend's cli_connection — subscription keeps the existing copy+button,
# bedrock/vertex get the new errorMessages.authFailureCloud copy with the
# CLI's own (already-correct) remedy text appended verbatim. Pairs with the
# backend adding that key; older backends fall back to a built-in string,
# safe to roll in either order.
# 2.8.0 — Claude usage/rate-limit turns are classified distinctly from
# credential failures and generic errors. New BackendRateLimitError
# (claude_cli.py: _RATE_LIMIT_RE + best-effort _extract_reset_time;
# anthropic.py: RateLimitError / APIStatusError 429/529) reports
# backend_status "rate_limited" — self-healing, same as BackendAuthError's
# "unauthenticated" clears on the next successful turn — and replies with
# the server's errorMessages.rateLimitFailure copy, appending a "resumes
# around HH:MM" ETA when one could be determined. Turns also carry
# metadata.errorKind="rate_limit" (no button, unlike auth_failure — there's
# nothing to click to fix a usage limit). Before this, a usage limit fell
# into the generic modelFailure apology, indistinguishable from a real bug,
# and invited retrying during the exact window that burns more of the same
# quota. Pairs with the backend's Runnability :llm_rate_limited blocker
# code and errorMessages.rateLimitFailure key; older backends fall back to
# a built-in string, safe to roll in either order.
# 2.9.0 — MCP routing context moved into the MCP_CONTEXT contextvar
# (backends/__init__.py), off the shared backend instance's mutable
# _mcp_* attributes. Prerequisite for max_concurrent > 1: with two
# in-flight turns, instance attributes interleave write→write→read→read
# and turn A's tool calls route into turn B's conversation/task. NOT
# safe to roll in either order — a pre-2.9 bridge handed 2 slots has the
# race live, so the backend raises @min_bridge_version to 2.9.0 in the
# same change that flips the max_concurrent_tasks default to 2.
# 2.9.1 — every REST call carries X-Bridge-Version (rest.py _request).
# The backend's /api/agents/my/settings clamps max_concurrent_tasks to 1
# for callers that don't prove the 2.9.0 floor; since pre-2.9.1 bridges
# never send the header, the concurrency floor now holds by construction
# instead of resting on the enforce_bridge_version feature flag staying
# on. Safe to roll in either order: an older backend ignores the header,
# a newer backend clamps older bridges to the single slot they had
# before 2.9.0 anyway.
# 2.9.2 — rate_limited health carries a reset deadline. backend_health
# gains `reset_at` (unix epoch): the parsed ETA when the CLI/API error
# named one, else now + RATE_LIMIT_RESET_FALLBACK_SECONDS. Closes the
# stale-offline gap in the 2.8.0 self-healing story: the blocker only
# cleared on a *successful turn*, so an agent that got no traffic after
# the limit reset read offline indefinitely. The server (Runnability)
# now expires an :llm_rate_limited blocker once reset_at passes; if the
# guess was early the next turn fails fast and re-marks with a fresh
# deadline. Safe to roll in either order: an older backend ignores the
# extra key, and an older bridge sends no reset_at, which keeps the
# pre-2.9.2 clear-on-success behaviour for that executor.
BRIDGE_VERSION = "2.9.2"
