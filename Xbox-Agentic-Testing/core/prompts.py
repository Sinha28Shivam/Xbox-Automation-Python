"""
prompts.py - load agent prompts from files, not from string literals.

WHY PROMPTS LIVE IN FILES
-------------------------
Prompts are the behaviour of an LLM agent. Burying them in Python means every
tuning tweak is a code change, they cannot be diffed meaningfully, and
non-engineers cannot read or improve them. Here each agent names a template in
agents.yaml and this loader renders it with the live run state.

Templates are Jinja2 when it is installed, with a small built-in renderer as a
fallback so the framework has no hard dependency on it. The fallback supports
{{ var }}, {% if %} and {% for %}, which covers what the shipped templates use.

WHAT A TEMPLATE RECEIVES
------------------------
Everything in `context`, which the agent base class fills with the state digest,
the control surface read from controls.yaml, and the tool list. That is why no
prompt hardcodes a button name: the template asks what exists at render time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
    _HAS_JINJA = True
except ImportError:                                   # pragma: no cover
    _HAS_JINJA = False


class PromptError(RuntimeError):
    pass


class PromptLibrary:
    """Renders prompt templates from a directory."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self._env: Any | None = None
        if _HAS_JINJA and self.directory.is_dir():
            self._env = Environment(
                loader=FileSystemLoader(str(self.directory)),
                # StrictUndefined turns a typo like {{ scenaro }} into a loud
                # error instead of an empty gap in the prompt. A silently
                # truncated instruction is far harder to notice than a crash.
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )
            self._env.filters["tojson"] = lambda v, indent=2: json.dumps(
                v, indent=indent, default=str, ensure_ascii=False)

    def exists(self, name: str) -> bool:
        return (self.directory / name).is_file()

    def render(self, name: str, context: dict[str, Any]) -> str:
        """Render `name` with `context`."""
        path = self.directory / name
        if not path.is_file():
            raise PromptError(
                f"Prompt template '{name}' not found in {self.directory}. "
                f"Check the 'prompt:' key for this agent in agents.yaml.")

        if self._env is not None:
            try:
                return self._env.get_template(name).render(**context)
            except Exception as exc:
                raise PromptError(f"Error rendering '{name}': {exc}") from exc

        return _render_basic(path.read_text(encoding="utf-8"), context)

    def raw(self, name: str) -> str:
        path = self.directory / name
        if not path.is_file():
            raise PromptError(f"Prompt template '{name}' not found")
        return path.read_text(encoding="utf-8")


# ===========================================================================
# Minimal fallback renderer (used only when Jinja2 is absent)
# ===========================================================================
_VAR = re.compile(r"\{\{\s*([a-zA-Z_][\w.]*)\s*(?:\|\s*tojson\s*)?\}\}")
_IF = re.compile(r"\{%\s*if\s+([\w.]+)\s*%\}(.*?)\{%\s*endif\s*%\}", re.S)
_FOR = re.compile(
    r"\{%\s*for\s+(\w+)\s+in\s+([\w.]+)\s*%\}(.*?)\{%\s*endfor\s*%\}", re.S)


def _lookup(context: dict[str, Any], dotted: str) -> Any:
    node: Any = context
    for part in dotted.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        else:
            node = getattr(node, part, None)
        if node is None:
            return None
    return node


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str, ensure_ascii=False)
    return str(value)


def _render_basic(template: str, context: dict[str, Any]) -> str:
    """Support {{ var }}, {% if %} and {% for %} without Jinja2."""

    def do_for(m: re.Match[str]) -> str:
        item_name, iterable_name, body = m.group(1), m.group(2), m.group(3)
        items = _lookup(context, iterable_name) or []
        if isinstance(items, dict):
            items = [{"key": k, "value": v} for k, v in items.items()]
        return "".join(
            _render_basic(body, {**context, item_name: item}) for item in items)

    def do_if(m: re.Match[str]) -> str:
        return m.group(2) if _lookup(context, m.group(1)) else ""

    out = _FOR.sub(do_for, template)
    out = _IF.sub(do_if, out)
    return _VAR.sub(lambda m: _stringify(_lookup(context, m.group(1))), out)
