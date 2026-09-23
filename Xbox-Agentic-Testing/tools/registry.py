"""Tool registry for the agentic Xbox test framework."""

from __future__ import annotations

import difflib
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from adapters import HardwareBridge
from artifacts import ArtifactStore
from config import Config


@dataclass
class ToolContext:
    hardware: HardwareBridge
    artifacts: ArtifactStore
    settings: Config
    dry_run: bool = False
    scratch: dict[str, Any] = field(default_factory=dict)

    def threshold(self, key: str, fallback: float) -> float:
        return self.settings.get(f"verification.{key}", fallback)


@dataclass
class ToolSpec:
    name: str
    description: str
    tags: list[str]
    factory: Callable[[ToolContext], Any]
    mutates_hardware: bool = False


class ToolRegistry:
    def __init__(self, context: ToolContext):
        self.context = context
        self._specs: dict[str, ToolSpec] = {}
        self._built: dict[str, Any] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate tool name '{spec.name}'")
        self._specs[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def load_modules(self, module_names: list[str]) -> None:
        for name in module_names:
            try:
                module = importlib.import_module(name)
            except ImportError as exc:
                raise ImportError(f"Tool module '{name}' could not be imported: {exc}") from exc
            provide = getattr(module, "provide", None)
            if provide is None:
                raise AttributeError(f"Tool module '{name}' must define provide() -> list[ToolSpec]")
            self.register_many(provide())

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown tool '{name}'. Registered: {', '.join(self.names)}")
        return self._specs[name]

    def by_tag(self, tag: str) -> list[ToolSpec]:
        return [s for s in self._specs.values() if tag in s.tags]

    def resolve(self, selectors: list[str]) -> list[ToolSpec]:
        out: list[ToolSpec] = []
        seen: set[str] = set()
        for selector in selectors or []:
            selector = str(selector).strip()
            matches = self.by_tag(selector[4:]) if selector.startswith("tag:") else [self.spec(selector)]
            for spec in matches:
                if spec.name not in seen:
                    seen.add(spec.name)
                    out.append(spec)
        return out

    def build(self, name: str) -> Any:
        if name not in self._built:
            self._built[name] = self.spec(name).factory(self.context)
        return self._built[name]

    def build_for(self, selectors: list[str]) -> list[Any]:
        tools: list[Any] = []
        for spec in self.resolve(selectors):
            if self.context.dry_run and spec.mutates_hardware:
                continue
            tools.append(self.build(spec.name))
        return tools

    def describe(self, selectors: list[str] | None = None) -> list[dict[str, Any]]:
        specs = self.resolve(selectors) if selectors else sorted(self._specs.values(), key=lambda s: s.name)
        return [{"name": s.name, "description": s.description.strip(), "tags": s.tags,
                 "mutates_hardware": s.mutates_hardware} for s in specs]


def ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def fail(error: str, **payload: Any) -> dict[str, Any]:
    return {"ok": False, "error": error, **payload}


def tolerant(func: Callable[..., Any], name: str) -> Callable[..., Any]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func

    accepted = set(signature.parameters)
    aliases = {
        "stability_duration": "settle", "stable_duration": "settle", "settle_time": "settle",
        "timeout_seconds": "timeout", "max_wait": "timeout", "seconds": "timeout",
        "frame1": "path_a", "frame2": "path_b", "frame_a": "path_a", "frame_b": "path_b",
        "before": "before_path", "after": "after_path", "repeat": "times", "count": "times",
        "button_name": "button", "control": "button", "query": "text", "search_text": "text",
        "path": "frame_path", "image_path": "frame_path",
    }

    def wrapper(**kwargs: Any) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        renamed: dict[str, str] = {}
        dropped: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in accepted:
                cleaned[key] = value
                continue
            target = aliases.get(key)
            if target and target in accepted and target not in kwargs:
                cleaned[target] = value
                renamed[key] = target
                continue
            close = difflib.get_close_matches(key, accepted, n=1, cutoff=0.75)
            if close and close[0] not in kwargs:
                cleaned[close[0]] = value
                renamed[key] = close[0]
                continue
            dropped[key] = value

        missing = [p.name for p in signature.parameters.values()
                   if p.default is inspect.Parameter.empty
                   and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                   and p.name not in cleaned]
        if missing:
            return fail(f"{name} is missing required argument(s): {', '.join(missing)}. "
                        f"Accepted arguments: {', '.join(sorted(accepted)) or 'none'}", dispatched=False)
        result = func(**cleaned)
        if isinstance(result, dict) and (renamed or dropped):
            notes = []
            if renamed:
                notes.append("renamed " + ", ".join(f"{k}->{v}" for k, v in renamed.items()))
            if dropped:
                notes.append(f"ignored unknown: {', '.join(dropped)}")
            result["argument_notes"] = "; ".join(notes)
            result["accepted_arguments"] = sorted(accepted)
        return result

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = dict(getattr(func, "__annotations__", {}))
    wrapper.__signature__ = signature  # type: ignore[attr-defined]
    wrapper.__module__ = getattr(func, "__module__", wrapper.__module__)
    if hasattr(func, "__globals__"):
        wrapper.__wrapped__ = func  # type: ignore[attr-defined]
    return wrapper


def make_tool(func: Callable[..., Any], name: str, description: str, args_schema: Any = None) -> Any:
    func = tolerant(func, name)
    try:
        signature = inspect.signature(func)
        params = ", ".join(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}"
                             for p in signature.parameters.values())
        description = f"{description}\n\nSignature: {name}({params})"
    except (TypeError, ValueError):
        pass
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        func.__name__ = name  # type: ignore[attr-defined]
        func.__doc__ = description  # type: ignore[attr-defined]
        return func
    kwargs: dict[str, Any] = {"func": func, "name": name, "description": description}
    if args_schema is not None:
        kwargs["args_schema"] = args_schema
    return StructuredTool.from_function(**kwargs)


def build_default_registry(hardware: HardwareBridge, artifacts: ArtifactStore,
                           settings: Config, module_names: list[str] | None = None) -> ToolRegistry:
    context = ToolContext(hardware=hardware, artifacts=artifacts, settings=settings,
                          dry_run=settings.get("runtime.dry_run", False))
    registry = ToolRegistry(context)
    registry.load_modules(module_names or [
        "hardware_tools", "input_tools", "vision_tools", "report_tools", "game_tools",
    ])
    return registry
