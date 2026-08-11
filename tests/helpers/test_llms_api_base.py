"""api_base / base_url override tests for the llms package.

Mirrors litellm's ``api_base`` override: a per-request ``api_base`` kwarg (via
the shim or directly on ``pipeline.acompletion``) overrides the provider's
default endpoint, ``{PROVIDER}_API_BASE`` env vars override globally at config
resolution, and each family adapter builds its request URL from the resolved
base. No network: the family adapter / package dispatch are monkeypatched.
"""

import asyncio

import cecli.helpers.llms as llms_pkg
import cecli.helpers.llms.pipeline as pipeline
from cecli.helpers.llms.config import resolve_model_config
from cecli.helpers.llms.litellm_compat import litellm
from cecli.helpers.llms.types import Choice, CompletionResponse, Message

MSGS = [{"role": "user", "content": "hi"}]


def _fake_response(model):
    return CompletionResponse(
        id="x",
        model=model,
        choices=[Choice(index=0, message=Message(role="assistant", content="hi"))],
    )


# ---------------------------------------------------------------------------
# Config-level precedence
# ---------------------------------------------------------------------------


def test_default_api_base_from_provider_defaults():
    resolved = resolve_model_config("deepseek/deepseek-v4-flash")
    assert resolved["api_base"] == "https://api.deepseek.com/v1"


def test_env_api_base_overrides_default(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://env.example.com/v1")
    resolved = resolve_model_config("deepseek/deepseek-v4-flash")
    assert resolved["api_base"] == "https://env.example.com/v1"


def test_env_api_base_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://env.example.com/v1/")
    resolved = resolve_model_config("deepseek/deepseek-v4-flash")
    assert resolved["api_base"] == "https://env.example.com/v1"


# ---------------------------------------------------------------------------
# Per-request override via the pipeline
# ---------------------------------------------------------------------------


def test_pipeline_api_base_override_reaches_adapter(monkeypatch):
    """pipeline.acompletion(api_base=...) overrides the resolved base (rstripped)."""
    captured = {}

    async def fake_chat_complete(resolved, messages, tools, key, headers, kwargs):
        captured["api_base"] = resolved["api_base"]
        captured["family"] = resolved["family"]
        return _fake_response(resolved["model"])

    monkeypatch.setattr(pipeline, "chat_complete", fake_chat_complete)

    asyncio.run(
        pipeline.acompletion(
            model="deepseek/deepseek-v4-flash",
            messages=MSGS,
            api_base="https://my-proxy.example.com/v1/",
        )
    )

    assert captured["api_base"] == "https://my-proxy.example.com/v1"
    assert captured["family"] == "chat"


def test_pipeline_without_api_base_uses_default(monkeypatch):
    captured = {}

    async def fake_chat_complete(resolved, messages, tools, key, headers, kwargs):
        captured["api_base"] = resolved["api_base"]
        return _fake_response(resolved["model"])

    monkeypatch.setattr(pipeline, "chat_complete", fake_chat_complete)

    asyncio.run(pipeline.acompletion(model="deepseek/deepseek-v4-flash", messages=MSGS))

    assert captured["api_base"] == "https://api.deepseek.com/v1"


# ---------------------------------------------------------------------------
# Shim forwarding
# ---------------------------------------------------------------------------


def test_shim_forwards_api_base_to_dispatch(monkeypatch):
    """litellm.acompletion(api_base=...) reaches the package dispatch."""
    captured = {}

    async def fake_dispatch(**kwargs):
        captured.update(kwargs)
        return _fake_response(kwargs.get("model"))

    monkeypatch.setattr(llms_pkg, "acompletion", fake_dispatch)

    asyncio.run(
        litellm.acompletion(
            model="deepseek/deepseek-v4-flash",
            messages=MSGS,
            stream=False,
            api_base="https://my-proxy.example.com/v1",
        )
    )

    assert captured.get("api_base") == "https://my-proxy.example.com/v1"
