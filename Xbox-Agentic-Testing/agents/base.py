"""
base.py - shared machinery for every agent.

WHAT AN AGENT IS HERE
---------------------
A callable node: `agent(state) -> partial state update`. LangGraph merges the
returned dict. Subclasses implement `run()`; the base class handles the parts
that must behave identically everywhere:

  * building the LLM from config (per-agent overrides honoured)
  * binding only the tools this agent was granted in agents.yaml
  * rendering its prompt template with the live state
  * retries, timing, and turning an exception into a state error rather than a
    crashed graph
  * recording every agent's output in `agent_outputs` for the RCA agent to read

DETERMINISTIC AGENTS
--------------------
Not every agent needs an LLM. The health agent, for example, is pure probing -
running a model over "is the capture card present?" would add cost, latency and
a chance of hallucination to a question with a factual answer. Those agents set
`uses_llm = False` and the base class skips model construction entirely, which
also means the framework can run health checks with no API key at all.

FAILURE POLICY
--------------
An agent that raises does not kill the run. The error is captured, appended to
`state["errors"]`, and the graph continues so the reporter can still explain
what happened. An agent marked `optional: true` degrades even more quietly.
A framework that dies on the first exception cannot report on its own failures.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from pydantic import BaseModel

from artifacts import ArtifactStore
from config import Config
from llm import LLMFactory, structured
from prompts import PromptLibrary
from registry import ToolRegistry
from state import AgenticState, note, state_digest


class AgentContext:
    """Everything an agent needs, assembled once by the runner."""

    def __init__(self, settings: Config, agents_config: Config,
                 llm_factory: LLMFactory, prompts: PromptLibrary,
                 tools: ToolRegistry, artifacts: ArtifactStore):
        self.settings = settings
        self.agents_config = agents_config
        self.llm_factory = llm_factory
        self.prompts = prompts
        self.tools = tools
        self.artifacts = artifacts

    def spec_for(self, role: str) -> dict[str, Any]:
        """An agent's config with defaults applied."""
        defaults = self.agents_config.section("defaults")
        spec = self.agents_config.section(f"agents.{role}")
        if not spec:
            raise KeyError(
                f"No agent '{role}' in agents.yaml. Defined: "
                f"{', '.join(self.agents_config.section('agents'))}")
        return {**defaults, **spec}


class BaseAgent:
    """Common behaviour for all agents."""

    role: str = "agent"
    uses_llm: bool = True

    def __init__(self, context: AgentContext, role: str | None = None):
        self.context = context
        self.role = role or self.role
        self.spec = context.spec_for(self.role)

        self.prompt_name: str = self.spec.get("prompt") or ""
        self.tool_selectors: list[str] = list(self.spec.get("tools") or [])
        self.max_retries: int = int(self.spec.get("max_retries", 2))
        self.optional: bool = bool(self.spec.get("optional", False))
        self.description: str = str(self.spec.get("description", "")).strip()

        self._llm: Any | None = None
        self._tools: list[Any] | None = None

    # -- lazily-built resources -------------------------------------------
    @property
    def llm(self) -> Any:
        """The chat model. Built on first use so a deterministic agent never
        triggers provider construction (and never needs an API key)."""
        if self._llm is None:
            self._llm = self.context.llm_factory.for_agent(self.spec)
        return self._llm

    @property
    def tools(self) -> list[Any]:
        """Only the tools this agent was granted."""
        if self._tools is None:
            self._tools = self.context.tools.build_for(self.tool_selectors)
        return self._tools

    def tool_catalogue(self) -> list[dict[str, Any]]:
        """Descriptions of this agent's tools, for its prompt."""
        return self.context.tools.describe(self.tool_selectors)

    def call_tool(self, tool_name: str, /, **kwargs: Any) -> dict[str, Any]:
        """Invoke a tool directly, bypassing the LLM.

        Deterministic agents use this. It also gives LLM agents a way to run a
        mandatory step (always capture a frame before acting) without hoping
        the model chooses to.

        `tool_name` is positional-only (the `/`). Several tools take their own
        `name` argument - resolve_control(name=...) among them - and a plain
        `def call_tool(self, name, **kwargs)` would collide with it, raising
        "got multiple values for argument 'name'". The marker makes tool
        keywords structurally unable to shadow this parameter.
        """
        tool = self.context.tools.build(tool_name)
        func = getattr(tool, "func", tool)
        try:
            return func(**kwargs)
        except Exception as exc:
            return {"ok": False, "error": f"{tool_name} raised: {exc}"}

    # -- prompting ---------------------------------------------------------
    def build_context(self, state: AgenticState,
                      **extra: Any) -> dict[str, Any]:
        """Variables handed to the prompt template.

        `controls` is included so prompts can enumerate the real control
        surface instead of naming buttons. That is what keeps the prompts
        console-agnostic: point the framework at a PS4 profile and the same
        template describes a DualShock.
        """
        return {
            "role": self.role,
            "description": self.description,
            "state": state_digest(state),
            "tools": self.tool_catalogue(),
            "controls": self.context.tools.context.hardware.controls_summary(),
            "runtime": self.context.settings.section("runtime"),
            "scenario_input": state.get("scenario_input", ""),
            **extra,
        }

    def render_prompt(self, state: AgenticState, **extra: Any) -> str:
        if not self.prompt_name:
            raise ValueError(f"Agent '{self.role}' has no 'prompt' configured")
        return self.context.prompts.render(
            self.prompt_name, self.build_context(state, **extra))

    def invoke_structured(self, schema: type[BaseModel], prompt: str,
                          images: list[dict[str, str]] | None = None) -> BaseModel:
        """One LLM call returning a validated `schema` instance.

        Retries on transient errors and on schema-validation failures - a model
        that returns a slightly wrong shape usually gets it right on a second
        attempt, and one bad response should not fail a hardware run.
        """
        runnable = structured(self.llm, schema)
        messages = _build_messages(prompt, images)

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return runnable.invoke(messages)
            except Exception as exc:
                last = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))       # simple backoff
        raise RuntimeError(
            f"Agent '{self.role}' could not get a valid "
            f"{schema.__name__} after {self.max_retries + 1} attempts: {last}")

    # -- node interface ----------------------------------------------------
    def __call__(self, state: AgenticState) -> dict[str, Any]:
        """LangGraph entry point. Never raises."""
        started = time.time()
        try:
            update = self.run(state)
        except Exception as exc:
            return self._handle_error(state, exc, time.time() - started)

        elapsed = round(time.time() - started, 2)
        outputs = dict(update.get("agent_outputs") or {})
        outputs.setdefault(self.role, {
            "ok": True,
            "duration_seconds": elapsed,
        })
        update["agent_outputs"] = outputs
        update.setdefault("messages", []).append(
            note(self.role, f"completed in {elapsed}s"))
        return update

    def run(self, state: AgenticState) -> dict[str, Any]:
        raise NotImplementedError(
            f"Agent '{self.role}' must implement run(state)")

    def _handle_error(self, state: AgenticState, exc: Exception,
                      elapsed: float) -> dict[str, Any]:
        detail = f"{exc.__class__.__name__}: {exc}"
        self.context.artifacts.append_log(
            "agent-errors.log",
            f"[{self.role}] {detail}\n{traceback.format_exc()}")

        return {
            "errors": [f"{self.role}: {detail}"],
            "agent_outputs": {
                self.role: {
                    "ok": False,
                    "error": detail,
                    "optional": self.optional,
                    "duration_seconds": round(elapsed, 2),
                }
            },
            "messages": [note(self.role, f"FAILED: {detail}", level="error")],
            # A required agent failing means the run cannot be trusted. We stop
            # and let the reporter say so, instead of continuing on partial
            # data and producing a confident-looking verdict from a broken run.
            "should_stop": not self.optional,
            "stop_reason": ("" if self.optional
                            else f"Required agent '{self.role}' failed: {detail}"),
        }


# ===========================================================================
# Helpers
# ===========================================================================
def _build_messages(prompt: str,
                    images: list[dict[str, str]] | None) -> Any:
    """Build the message payload, attaching screenshots when supplied.

    Uses LangChain's multimodal content-block format, which the vision-capable
    providers all understand.
    """
    if not images:
        return [("human", prompt)]

    blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img.get('media_type', 'image/png')};"
                       f"base64,{img['base64']}"
            },
        })
    try:
        from langchain_core.messages import HumanMessage
        return [HumanMessage(content=blocks)]
    except ImportError:                                   # pragma: no cover
        return [("human", prompt)]
