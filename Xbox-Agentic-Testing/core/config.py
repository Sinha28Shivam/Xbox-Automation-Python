"""
config.py - load YAML config with ${ENV:default} expansion.

WHY THIS EXISTS
---------------
The requirement for this framework was "nothing hardcoded". That is easy to say
and easy to quietly break: one `model="claude-..."` default in a constructor and
the config file becomes decorative.

So every tunable value in the system comes through here, and code reads it with
`cfg.get("a.b.c", fallback)`. If you find a literal device name, model id, path
or threshold anywhere else in the codebase, that is a bug.

ENV EXPANSION
-------------
Any string may contain ${VAR} or ${VAR:default}. Expansion happens once at load
time, so the rest of the code never touches os.environ:

    model: "${ANTHROPIC_MODEL:claude-sonnet-4-5-20250929}"

Precedence is  environment variable  >  the default in the file.

TYPE COERCION
-------------
Environment variables are always strings, but `max_steps: "${MAX_STEPS:40}"`
should still be an int. `Config.get` coerces to the type of the fallback you
pass, so `cfg.get("runtime.max_steps", 40)` returns an int either way. This
avoids a whole family of "40" != 40 bugs that only appear once someone actually
sets the env var.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypeVar

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit("PyYAML is required:  pip install pyyaml")

T = TypeVar("T")

# ${VAR} or ${VAR:default}. The default may contain anything except '}'.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", "none", "null", ""}


def expand_env(value: Any) -> Any:
    """Recursively expand ${VAR:default} in strings, lists and dicts."""
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var) or (default if default is not None else "")
        return _ENV_PATTERN.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def coerce(value: Any, like: T) -> T:
    """Coerce `value` to the type of `like`.

    Needed because env vars arrive as strings. Returns `value` unchanged if it
    cannot be coerced - a wrong-typed value is better than a crash inside a
    config lookup, and the caller's own validation will catch it.
    """
    if like is None or value is None:
        return value
    target = type(like)
    if isinstance(value, target):
        return value
    try:
        if target is bool:
            if isinstance(value, str):
                low = value.strip().lower()
                if low in _TRUE:
                    return True            # type: ignore[return-value]
                if low in _FALSE:
                    return False           # type: ignore[return-value]
            return bool(value)             # type: ignore[return-value]
        if target is int:
            return int(float(value))       # type: ignore[return-value]
        if target is float:
            return float(value)            # type: ignore[return-value]
        if target is str:
            return str(value)              # type: ignore[return-value]
    except (TypeError, ValueError):
        pass
    return value


class Config:
    """A dict with dotted-path lookup and type-aware defaults."""

    def __init__(self, data: dict[str, Any] | None = None,
                 source: Path | None = None, base: Path | None = None):
        self.data: dict[str, Any] = data or {}
        self.source = source
        # Directory that relative paths in this config resolve against.
        # Defaults to the config file's own folder, but the runner overrides it
        # with the PROJECT ROOT - because paths in settings.yaml are written
        # relative to the project ("./scenarios", "../Xbox-Automation-Python"),
        # not relative to the config/ folder the file happens to live in.
        self._base = base

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str, base: Path | None = None) -> "Config":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(expand_env(raw), source=path, base=base)

    @classmethod
    def load_all(cls, directory: Path | str, names: dict[str, str],
                 base: Path | None = None) -> dict[str, "Config"]:
        """Load several named config files from one directory.

        `names` maps a short key -> filename, e.g. {"settings": "settings.yaml"}.
        A missing file yields an empty Config rather than an error, so an
        optional config (say a per-site override) can simply be absent.
        """
        directory = Path(directory)
        out: dict[str, Config] = {}
        for key, filename in names.items():
            p = directory / filename
            out[key] = (cls.load(p, base=base) if p.is_file()
                        else cls({}, source=p, base=base))
        return out

    @property
    def base_dir(self) -> Path:
        """Directory relative paths resolve against."""
        if self._base is not None:
            return self._base
        return self.source.parent if self.source else Path.cwd()

    # -- access ------------------------------------------------------------
    def get(self, dotted: str, fallback: T = None) -> T:      # type: ignore[assignment]
        """Fetch by dotted path, coerced to the type of `fallback`.

            cfg.get("runtime.max_steps", 40)   -> int
            cfg.get("runtime.dry_run", False)  -> bool
        """
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return fallback
            node = node[part]
        if node is None:
            return fallback
        return coerce(node, fallback)          # type: ignore[return-value]

    def section(self, dotted: str) -> dict[str, Any]:
        """A sub-dict, or {} if absent. Never returns None."""
        node = self.get(dotted, None)
        return node if isinstance(node, dict) else {}

    def list_of(self, dotted: str) -> list[Any]:
        node = self.get(dotted, None)
        if node is None:
            return []
        return node if isinstance(node, list) else [node]

    def require(self, dotted: str) -> Any:
        """Fetch a value that has no sensible default.

        Fails loudly and points at the file, because a silent None here shows
        up much later as a confusing error inside an agent.
        """
        node = self.get(dotted, None)
        if node is None:
            raise KeyError(
                f"Required config key '{dotted}' is missing from "
                f"{self.source or '<inline>'}")
        return node

    def resolve_path(self, dotted: str, fallback: str = "",
                     base: Path | None = None) -> Path:
        """A path value resolved against `base` (default: `base_dir`).

        Relative paths in config are NEVER relative to the shell's working
        directory - otherwise the same config would behave differently
        depending on where you happened to run the CLI from, which is a
        genuinely nasty class of bug to track down.
        """
        raw = self.get(dotted, fallback)
        p = Path(str(raw)).expanduser()
        if p.is_absolute():
            return p
        return ((base or self.base_dir) / p).resolve()

    # -- merging -----------------------------------------------------------
    def merged_with(self, other: "Config | dict[str, Any]") -> "Config":
        """Deep-merge another config over this one (other wins)."""
        payload = other.data if isinstance(other, Config) else other
        return Config(_deep_merge(self.data, payload),
                      source=self.source, base=self._base)

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, None) is not None

    def __repr__(self) -> str:
        return f"Config(source={self.source}, keys={list(self.data)})"


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_dotenv_if_present(path: Path | str) -> None:
    """Load a .env file without requiring python-dotenv.

    Existing environment variables always win, matching the note in .env.example
    that a key already set in the shell takes precedence.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ and val:
            os.environ[key] = val
