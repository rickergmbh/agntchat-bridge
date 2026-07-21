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
BRIDGE_VERSION = "2.6.0"
