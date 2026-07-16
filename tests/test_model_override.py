"""Per-turn MODEL_OVERRIDE contextvar must swap the model at request time.

PulseExecutionWorker stamps pulse config's `model` onto task metadata as
`model_override`; the bridge sets the MODEL_OVERRIDE contextvar for the
duration of that task's handler. Backends resolve the model per request via
ModelBackend._request_model(), so the override applies to every LLM call in
the turn — and, because contextvars are per-asyncio-task, never to turns
running concurrently on the same backend instance.
"""

import asyncio
import contextvars
import sys

import pytest

from agentchat.backends import MODEL_OVERRIDE, ModelBackend, ModelResult


class _StubBackend(ModelBackend):
    """Minimal concrete backend exposing _request_model()."""

    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system_prompt, user_prompt, on_progress=None):
        return ModelResult(
            text="", model=self._request_model() or "", elapsed_seconds=0.0
        )


def test_request_model_defaults_to_configured_model():
    backend = _StubBackend("claude-opus-4-8")
    assert backend._request_model() == "claude-opus-4-8"


def test_request_model_prefers_override():
    backend = _StubBackend("claude-opus-4-8")

    def _in_fresh_context():
        MODEL_OVERRIDE.set("claude-haiku-4-5-20251001")
        return backend._request_model()

    # Run in a copied context so the set() can't leak into other tests.
    ctx = contextvars.copy_context()
    assert ctx.run(_in_fresh_context) == "claude-haiku-4-5-20251001"
    assert backend._request_model() == "claude-opus-4-8"


def test_empty_override_falls_back_to_configured_model():
    backend = _StubBackend("claude-opus-4-8")

    def _in_fresh_context():
        MODEL_OVERRIDE.set("")
        return backend._request_model()

    ctx = contextvars.copy_context()
    assert ctx.run(_in_fresh_context) == "claude-opus-4-8"


@pytest.mark.asyncio
async def test_override_is_isolated_between_concurrent_tasks():
    """Two concurrent handlers on one backend instance must each see their
    own override — the exact race a mutable backend attribute would lose."""
    backend = _StubBackend("claude-opus-4-8")
    started = asyncio.Event()
    release = asyncio.Event()

    async def pulse_turn():
        MODEL_OVERRIDE.set("claude-haiku-4-5-20251001")
        started.set()
        await release.wait()
        return (await backend.generate("s", "u")).model

    async def normal_turn():
        await started.wait()
        model = (await backend.generate("s", "u")).model
        release.set()
        return model

    pulse_model, normal_model = await asyncio.gather(pulse_turn(), normal_turn())
    assert pulse_model == "claude-haiku-4-5-20251001"
    assert normal_model == "claude-opus-4-8"


@pytest.mark.asyncio
async def test_anthropic_backend_passes_override_to_api(monkeypatch):
    """AnthropicBackend must send the override as the API `model` param."""
    anthropic = pytest.importorskip("anthropic")  # noqa: F841

    from agentchat.backends.anthropic import AnthropicBackend

    backend = AnthropicBackend(model="claude-opus-4-8", api_key="test-key")
    seen: dict = {}

    class _Content:
        text = "ok"

    class _Response:
        content = [_Content()]
        usage = None

    async def _fake_create(**kwargs):
        seen.update(kwargs)
        return _Response()

    monkeypatch.setattr(backend._client.messages, "create", _fake_create)

    async def _pulse_turn():
        MODEL_OVERRIDE.set("claude-haiku-4-5-20251001")
        return await backend.generate("system", "user")

    result = await asyncio.ensure_future(_pulse_turn())
    assert seen["model"] == "claude-haiku-4-5-20251001"
    assert result.model == "claude-haiku-4-5-20251001"

    # Outside the override context the configured model is used again.
    seen.clear()
    result = await backend.generate("system", "user")
    assert seen["model"] == "claude-opus-4-8"
