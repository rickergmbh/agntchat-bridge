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
BRIDGE_VERSION = "2.7.3"
