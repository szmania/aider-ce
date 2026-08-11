"""providers.json coverage: every OpenAI-compatible provider resolves.

``cecli/resources/providers.json`` lists the OpenAI-compatible providers cecli
ships with (base URL + API-key env var). This locks in that each one:

- is loaded by ``ModelProviderManager`` (supports_provider == True)
- resolves through ``resolve_model_config`` to the ``chat`` family (the
  OpenAI-compatible /chat/completions wire, which is what these providers
  advertise); ``chatgpt`` is the exception and uses OpenAI's newer
  ``responses`` family
  OpenAI-compatible /chat/completions wire, which is what these providers
  advertise)
- keeps its configured base URL and key env var
- routes through the base OpenAI-style provider adapter (Bearer auth), since
  none of them register a custom adapter

No network: resolution and adapter dispatch are offline.
"""

import importlib.resources as importlib_resources
import json

from cecli.helpers.llms.config import resolve_model_config
from cecli.helpers.llms.providers import ProviderAdapter, get_provider_adapter
from cecli.helpers.model_providers import ModelProviderManager

RESOURCE_FILE = "providers.json"


def _providers_json() -> dict:
    resource = importlib_resources.files("cecli.resources").joinpath(RESOURCE_FILE)
    return json.loads(resource.read_text())


def test_every_providers_json_entry_is_supported():
    providers = _providers_json()
    assert providers, "providers.json should not be empty"
    mpm = ModelProviderManager()

    for name, cfg in providers.items():
        assert mpm.supports_provider(name), f"{name} should be supported"
        pcfg = mpm.get_provider_config(name) or {}
        assert pcfg.get("api_base") == cfg["api_base"]
        assert pcfg.get("api_key_env") == cfg["api_key_env"]


def test_every_provider_resolves_to_chat_family_with_base_and_key():
    providers = _providers_json()

    for name, cfg in providers.items():
        resolved = resolve_model_config(f"{name}/sample-model")
        if name == "chatgpt":  # ChatGPT subscription routes via /v1/responses
            assert resolved["family"] == "responses", f"{name} should use responses"
        else:
            assert resolved["family"] == "chat", f"{name} should use chat completions"
        assert resolved["provider"] == name
        assert resolved["api_base"] == cfg["api_base"].rstrip("/")
        assert resolved["api_key_env"] in cfg["api_key_env"]


def test_unregistered_providers_use_openai_style_base_adapter():
    registry_names = set(_providers_json())

    # None of the providers.json entries register a dedicated adapter, so
    # dispatch falls back to the base OpenAI-style adapter (Bearer auth).
    for name in registry_names:
        adapter = get_provider_adapter(name)
        assert isinstance(adapter, ProviderAdapter)
        assert adapter.provider == "openai"
