"""
llm.py - build chat models from config, never from literals.

WHY A FACTORY
-------------
Provider and model are declared in settings.yaml. This module turns that
declaration into a live LangChain chat model by importing the integration
package BY NAME at call time:

    module: "langchain_anthropic"
    class:  "ChatAnthropic"
    params: { model: "...", temperature: 0 }

Adding a provider is therefore a YAML edit plus a pip install - no code change,
and no `if provider == "openai"` chain to keep in sync. It also means the
framework imports only the provider you actually use, so an unused
langchain-openai does not need to be installed.

STRUCTURED OUTPUT
-----------------
`structured(model, schema)` wraps `with_structured_output` and falls back to
JSON-mode parsing if the provider does not support tool-calling. Agents rely on
getting a validated pydantic object back, so this fallback keeps a
less-capable local model (Ollama, say) usable instead of crashing the run.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from config import Config

TModel = TypeVar("TModel", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a model cannot be constructed. Always actionable."""


class LLMFactory:
    """Creates chat models from the `llm:` section of settings.yaml."""

    def __init__(self, settings: Config):
        self.settings = settings
        self._cache: dict[str, Any] = {}

    # -- provider resolution ----------------------------------------------
    @property
    def default_provider(self) -> str:
        name = self.settings.get("llm.provider", "")
        if not name:
            raise LLMError(
                "llm.provider is not set in settings.yaml and no "
                "AGENTIC_LLM_PROVIDER environment variable was found.")
        return str(name)

    def provider_spec(self, provider: str | None = None) -> dict[str, Any]:
        name = provider or self.default_provider
        spec = self.settings.section(f"llm.providers.{name}")
        if not spec:
            known = ", ".join(self.settings.section("llm.providers")) or "none"
            raise LLMError(
                f"Unknown LLM provider '{name}'. Configured providers: {known}")
        return spec

    def supports_vision(self, provider: str | None = None) -> bool:
        """Whether this provider can be shown screenshots.

        The executor and verifier degrade to numeric frame analysis when it
        cannot - which is weaker but still honest, rather than pretending to
        have looked at the screen.
        """
        return bool(self.provider_spec(provider).get("supports_vision", False))

    # -- construction ------------------------------------------------------
    def build(self, provider: str | None = None, model: str | None = None,
              **overrides: Any) -> Any:
        """Instantiate a chat model.

        Precedence: explicit kwargs > `model` arg > provider params in YAML.
        Results are cached per (provider, params) so agents sharing a config
        share one client and one connection pool.
        """
        name = provider or self.default_provider
        spec = self.provider_spec(name)

        params: dict[str, Any] = dict(spec.get("params") or {})
        if model:
            params["model"] = model
        params.update({k: v for k, v in overrides.items() if v is not None})

        cache_key = f"{name}:{json.dumps(params, sort_keys=True, default=str)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check the API key BEFORE importing, so the error names the missing
        # variable instead of surfacing as an opaque auth failure mid-run.
        key_env = spec.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            raise LLMError(
                f"Provider '{name}' needs the {key_env} environment variable. "
                f"Set it, or put it in a .env file next to the CLI.")

        module_name = spec.get("module")
        class_name = spec.get("class")
        if not module_name or not class_name:
            raise LLMError(
                f"Provider '{name}' must declare both 'module' and 'class' "
                f"in settings.yaml.")

        try:
            module = importlib.import_module(str(module_name))
        except ImportError as exc:
            raise LLMError(
                f"Provider '{name}' needs the '{module_name}' package: "
                f"pip install {str(module_name).replace('_', '-')}\n"
                f"  ({exc})") from exc

        try:
            cls = getattr(module, str(class_name))
        except AttributeError as exc:
            raise LLMError(
                f"'{module_name}' has no class '{class_name}'. Check the "
                f"provider entry in settings.yaml.") from exc

        try:
            client = cls(**params)
        except Exception as exc:                      # provider-specific errors
            raise LLMError(
                f"Could not construct {class_name} with {params}: {exc}") from exc

        self._cache[cache_key] = client
        return client

    def for_agent(self, agent_spec: dict[str, Any]) -> Any:
        """Build the model for one agent, honouring its per-agent overrides.

        This is what allows a cheap fast model for routing and an expensive
        vision model for the verifier, without any code knowing which is which.
        """
        return self.build(
            provider=agent_spec.get("provider") or None,
            model=agent_spec.get("model") or None,
            **(agent_spec.get("llm_params") or {}),
        )


# ===========================================================================
# Structured output
# ===========================================================================
def structured(model: Any, schema: type[TModel]) -> Any:
    """Return a runnable that emits `schema` instances.

    Prefers the provider's native structured output. If that is unavailable we
    return a shim that asks for JSON and parses it - slower and less reliable,
    but it keeps local/basic models working rather than hard-failing.
    """
    try:
        return model.with_structured_output(schema)
    except (AttributeError, NotImplementedError, TypeError):
        return _JsonFallback(model, schema)


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class _JsonFallback:
    """Ask for JSON in the prompt and parse the reply into `schema`."""

    def __init__(self, model: Any, schema: type[BaseModel]):
        self.model = model
        self.schema = schema

    def invoke(self, messages: Any, **kwargs: Any) -> BaseModel:
        instruction = (
            "\n\nReply with ONLY a JSON object matching this schema. "
            "No prose, no markdown fences.\n"
            f"{json.dumps(self.schema.model_json_schema(), indent=2)}"
        )
        payload = _append_instruction(messages, instruction)
        reply = self.model.invoke(payload, **kwargs)
        text = getattr(reply, "content", reply)
        if isinstance(text, list):        # some providers return content blocks
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in text)
        return self.schema.model_validate(_extract_json(str(text)))

    def __call__(self, messages: Any, **kwargs: Any) -> BaseModel:
        return self.invoke(messages, **kwargs)


def _append_instruction(messages: Any, instruction: str) -> Any:
    if isinstance(messages, str):
        return messages + instruction
    if isinstance(messages, list) and messages:
        out = list(messages)
        last = out[-1]
        if isinstance(last, tuple) and len(last) == 2:
            out[-1] = (last[0], f"{last[1]}{instruction}")
        elif isinstance(last, dict) and "content" in last:
            out[-1] = {**last, "content": f"{last['content']}{instruction}"}
        elif hasattr(last, "content"):
            out.append(("human", instruction))
        return out
    return messages


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Models wrap JSON in fences or add a sentence before it even when told not
    to, so we try three strategies before giving up.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = _JSON_BLOCK.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Model reply did not contain valid JSON:\n{text[:500]}")
