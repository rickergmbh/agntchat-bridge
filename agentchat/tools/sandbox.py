"""Code-as-Action sandbox — subprocess execution of LLM-generated Python code.

Inspired by the CodeAct pattern (arxiv 2402.01030) and Anthropic's programmatic
tool calling. The LLM generates Python code that calls pre-defined agentchat.*
functions. Code runs in a subprocess with restricted environment and timeout.

Usage:
    from agentchat.tools.sandbox import CodeSandbox

    sandbox = CodeSandbox(
        base_url="https://agentchat-backend.fly.dev",
        api_key="ak_...",
        agent_id="agent_123",
        conversation_id="conv_456",
    )

    output = await sandbox.execute('''
import agentchat

messages = agentchat.get_messages("conv_456", limit=5)
for msg in messages:
    print(f"[{msg['senderName']}]: {msg['content'][:50]}")

agentchat.send_message("conv_456", "Here's a summary of recent messages...")
''')
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agentchat.tools.sandbox")

# Maximum execution time for sandbox subprocess (seconds)
_DEFAULT_TIMEOUT = 30

# Maximum output capture (characters)
_MAX_OUTPUT = 50_000


@dataclass
class SandboxResult:
    """Result from executing code in the sandbox."""

    output: str
    return_code: int
    error: str = ""
    timed_out: bool = False


# The runtime stub is injected into the subprocess as `agentchat` module.
# It uses stdlib urllib.request for HTTP calls — no external dependencies.
_RUNTIME_STUB = textwrap.dedent('''\
import json
import os
import sys
import urllib.request
import urllib.error

_BASE_URL = os.environ["AGENTCHAT_BASE_URL"]
_API_KEY = os.environ["AGENTCHAT_API_KEY"]
_AGENT_ID = os.environ["AGENTCHAT_AGENT_ID"]
_CONVERSATION_ID = os.environ.get("AGENTCHAT_CONVERSATION_ID", "")

def _headers():
    return {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _api(method, path, body=None):
    """Make an API call and return parsed JSON."""
    url = f"{_BASE_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=_headers(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        return {"error": f"HTTP {e.code}: {error_body[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def send_message(conversation_id, content, metadata=None):
    """Send a message to a conversation."""
    body = {"content": content}
    if metadata:
        body["metadata"] = metadata
    return _api("POST", f"/api/conversations/{conversation_id}/messages", body)


def get_messages(conversation_id, limit=10, before_id=None):
    """Fetch recent messages from a conversation."""
    params = f"?limit={limit}"
    if before_id:
        params += f"&before={before_id}"
    return _api("GET", f"/api/conversations/{conversation_id}/messages{params}")


def create_task(conversation_id, title, description="", assigned_to=None):
    """Create a task in a conversation."""
    body = {"title": title, "description": description}
    if assigned_to:
        body["assigned_to"] = assigned_to if isinstance(assigned_to, list) else [assigned_to]
    return _api("POST", f"/api/conversations/{conversation_id}/tasks", body)


def get_memory(conversation_id):
    """Get conversation memory/summary."""
    result = _api("GET", f"/api/conversations/{conversation_id}/memory")
    if isinstance(result, dict) and "memory" in result:
        return result["memory"]
    return result


def search_memory(query, limit=5):
    """Search agent persistent memories."""
    return _api("GET", f"/api/agents/me/memories?q={urllib.request.quote(query)}&limit={limit}")


def save_memory(category, key, content, confidence=0.8):
    """Save a fact/learning to persistent memory."""
    body = {"category": category, "key": key, "content": content, "confidence": confidence}
    return _api("POST", "/api/agents/me/memories", body)


def find_or_create_dm(participant_id):
    """Find or create a DM conversation with a participant."""
    return _api("POST", f"/api/conversations/dm/{participant_id}")


def update_task_status(task_id, status, summary=None):
    """Update a task's status."""
    body = {"status": status}
    if summary:
        body["summary"] = summary
    return _api("PATCH", f"/api/tasks/{task_id}", body)


def get_owner_location():
    """Get the human owner's GPS location."""
    return _api("GET", "/api/owner/location")


# Convenience: default conversation_id from env
def reply(content, metadata=None):
    """Send a reply to the current conversation."""
    if not _CONVERSATION_ID:
        return {"error": "No conversation context"}
    return send_message(_CONVERSATION_ID, content, metadata)
''')


class CodeSandbox:
    """Execute LLM-generated Python code in an isolated subprocess."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        agent_id: str,
        conversation_id: str = "",
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._agent_id = agent_id
        self._conversation_id = conversation_id
        self._timeout = timeout

    async def execute(self, code: str) -> SandboxResult:
        """Execute Python code in a sandboxed subprocess.

        The code has access to an `agentchat` module with functions like
        send_message, get_messages, create_task, etc. that call the
        AgentChat REST API.

        Returns:
            SandboxResult with captured stdout/stderr.
        """
        # Write the runtime stub and user code to temp files
        with tempfile.TemporaryDirectory(prefix="agentchat_sandbox_") as tmpdir:
            # Write the agentchat module stub
            stub_path = os.path.join(tmpdir, "agentchat.py")
            with open(stub_path, "w") as f:
                f.write(_RUNTIME_STUB)

            # Write the user code
            code_path = os.path.join(tmpdir, "user_code.py")
            with open(code_path, "w") as f:
                f.write(code)

            # Build restricted environment
            env = {
                "AGENTCHAT_BASE_URL": self._base_url,
                "AGENTCHAT_API_KEY": self._api_key,
                "AGENTCHAT_AGENT_ID": self._agent_id,
                "AGENTCHAT_CONVERSATION_ID": self._conversation_id,
                "PYTHONPATH": tmpdir,
                "HOME": tmpdir,
                "TMPDIR": tmpdir,
                # Minimal PATH for Python only
                "PATH": os.path.dirname(sys.executable),
            }

            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, code_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=tmpdir,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=self._timeout,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return SandboxResult(
                        output="",
                        return_code=-1,
                        error=f"Execution timed out after {self._timeout}s",
                        timed_out=True,
                    )

                output = stdout.decode("utf-8", errors="replace")[:_MAX_OUTPUT]
                error = stderr.decode("utf-8", errors="replace")[:_MAX_OUTPUT]

                return SandboxResult(
                    output=output,
                    return_code=proc.returncode or 0,
                    error=error,
                )

            except Exception as e:
                logger.error("Sandbox execution failed: %s", e)
                return SandboxResult(
                    output="",
                    return_code=-1,
                    error=f"Sandbox error: {e}",
                )


def extract_python_code(text: str) -> str | None:
    """Extract Python code from LLM response (```python ... ``` fence).

    Returns the code string or None if no code block found.
    """
    import re

    # Try fenced code block first
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code fence
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Heuristic: check if it looks like Python
        if any(kw in code for kw in ("import ", "def ", "agentchat.", "print(", "for ", "if ")):
            return code

    return None


__all__ = ["CodeSandbox", "SandboxResult", "extract_python_code"]
