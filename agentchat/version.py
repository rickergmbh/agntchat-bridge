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
BRIDGE_VERSION = "2.4.0"
