"""
adapters.py - bridge to the existing hardware layer.

WHY NOT JUST COPY THE CODE
--------------------------
Xbox-Automation-Python already contains hardware-verified modules: GimxSession
owns the serial port and the Guide-button handshake, ConsolePad sends
config-driven input, ScreenCapture knows the capture card's real warm-up
behaviour. Every one of those encodes a measurement that took someone real time
to establish.

Copying them here would fork that knowledge and guarantee the two copies drift.
Instead we import the ORIGINAL files at runtime, by path, from the location
given in settings.yaml. The hardware layer stays the single source of truth and
this project stays a pure orchestration layer on top of it.

WHY IMPORT BY PATH
------------------
The directories are named with hyphens (`gimx-session`, `test-controller`), so
they are not importable packages - `import gimx-session` is a syntax error.
importlib.util.spec_from_file_location loads them anyway, without requiring the
other repo to be restructured or pip-installed.

DEGRADED MODE
-------------
An import failure never crashes startup. `HardwareBridge` records the error and
reports `available=False`; the health agent then turns that into a BLOCKED
verdict with a readable message. Crashing on import would produce a stack trace
instead of a diagnosis - and the whole point of the health agent is to diagnose.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any

from config import Config


class AdapterLoadError(RuntimeError):
    pass


def load_module_from_path(path: Path, module_name: str) -> Any:
    """Import a .py file as a module, whatever its folder is called."""
    path = Path(path)
    if not path.is_file():
        raise AdapterLoadError(f"Module file not found: {path}")

    # Its own folder must be importable too: test_controller.py does
    # `from gimx_session import ...` relative to the sibling directory.
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AdapterLoadError(f"Cannot build an import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    # Register before exec so a self-referential import inside the module
    # resolves instead of re-entering and looping.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise AdapterLoadError(f"Error while importing {path}: {exc}") from exc
    return module


class Adapter:
    """One lazily-imported hardware module plus its load status."""

    def __init__(self, name: str, path: Path, module_name: str,
                 exports: list[str]):
        self.name = name
        self.path = path
        self.module_name = module_name
        self.exports = exports
        self.module: Any | None = None
        self.error: str | None = None
        self._loaded = False

    def load(self) -> bool:
        """Import on first use. Errors are captured, never raised."""
        if self._loaded:
            return self.module is not None
        self._loaded = True
        try:
            self.module = load_module_from_path(self.path, self.module_name)
        except (AdapterLoadError, ImportError, SystemExit) as exc:
            # SystemExit is caught deliberately: the legacy modules call
            # sys.exit() when a dependency such as PyYAML or cv2 is missing.
            # In a CLI that is friendly; inside a framework it would kill the
            # whole run, so we convert it into a health finding instead.
            self.error = str(exc) or exc.__class__.__name__
            self.module = None
            return False

        missing = [e for e in self.exports if not hasattr(self.module, e)]
        if missing:
            self.error = (
                f"{self.path.name} loaded but is missing: {', '.join(missing)}. "
                f"The hardware layer may have been refactored - update the "
                f"'exports' list in settings.yaml.")
        return True

    @property
    def available(self) -> bool:
        self.load()
        return self.module is not None and self.error is None

    def get(self, attr: str) -> Any:
        """Fetch an export, or raise with a message that says what to fix."""
        if not self.load() or self.module is None:
            raise AdapterLoadError(
                f"Adapter '{self.name}' is unavailable: {self.error}")
        if not hasattr(self.module, attr):
            raise AdapterLoadError(
                f"'{attr}' not found in adapter '{self.name}' ({self.path})")
        return getattr(self.module, attr)

    def status(self) -> dict[str, Any]:
        self.load()
        return {
            "name": self.name,
            "path": str(self.path),
            "available": self.available,
            "error": self.error,
        }


class HardwareBridge:
    """Single access point to GIMX, the pad and the capture card.

    Owns the long-lived objects. ScreenCapture in particular must be opened
    ONCE and kept: the first frame costs ~700 ms (device open) while later ones
    cost ~1 ms, and only one process may hold the device at a time. Re-opening
    it per tool call would make every verification unusably slow.
    """

    def __init__(self, settings: Config):
        self.settings = settings
        self._lock = threading.Lock()

        automation_root = settings.resolve_path(
            "paths.automation_root", "../Xbox-Automation-Python")
        self.automation_root = automation_root
        self.controls_path = settings.resolve_path(
            "paths.controls_config",
            "../Xbox-Automation-Python/config/controls.yaml")

        self.adapters: dict[str, Adapter] = {}
        for name, spec in (settings.section("adapters") or {}).items():
            self.adapters[name] = Adapter(
                name=name,
                path=automation_root / str(spec.get("file", "")),
                module_name=str(spec.get("module_name", name)),
                exports=list(spec.get("exports") or []),
            )

        self.dry_run = settings.get("runtime.dry_run", False)
        self._pad: Any | None = None
        self._capture: Any | None = None
        self._controls: Any | None = None

    # -- adapters ----------------------------------------------------------
    def adapter(self, name: str) -> Adapter:
        if name not in self.adapters:
            known = ", ".join(self.adapters) or "none"
            raise AdapterLoadError(
                f"No adapter '{name}' configured. Known: {known}")
        return self.adapters[name]

    def status(self) -> dict[str, Any]:
        return {
            "automation_root": str(self.automation_root),
            "controls_config": str(self.controls_path),
            "controls_exists": self.controls_path.is_file(),
            "dry_run": self.dry_run,
            "adapters": {n: a.status() for n, a in self.adapters.items()},
        }

    # -- controls config ---------------------------------------------------
    @property
    def controls(self) -> Any:
        """The ControlConfig object: every button, timing and console profile.

        Agents introspect this to discover what controls exist, which is what
        keeps control names out of prompts and out of the code.
        """
        if self._controls is None:
            cls = self.adapter("pad").get("ControlConfig")
            self._controls = cls(self.controls_path)
        return self._controls

    def controls_summary(self) -> dict[str, Any]:
        """A compact description of the control surface, for prompts.

        Deliberately dynamic: whatever is in controls.yaml is what the agents
        are told exists. Add a button to the YAML and the agents can use it
        with no code change.
        """
        try:
            c = self.controls
        except (AdapterLoadError, Exception) as exc:
            return {"error": str(exc)}
        return {
            "buttons": {
                name: {
                    "aliases": spec.get("aliases", []),
                    "description": spec.get("description", ""),
                }
                for name, spec in c.buttons.items()
            },
            "triggers": {
                name: {
                    "range": [spec.get("min", 0), spec.get("max", 255)],
                    "description": spec.get("description", ""),
                }
                for name, spec in c.triggers.items()
            },
            "sticks": {
                name: list((spec.get("directions") or {}).keys())
                for name, spec in c.sticks.items()
            },
            "macros": {
                name: spec.get("description", "")
                for name, spec in c.macros.items()
            },
            "special_actions": {
                name: {
                    "description": spec.get("description", ""),
                    # Surfaced so the planner can prefer verified sequences and
                    # the report can flag when an unverified one was relied on.
                    "verified": bool(spec.get("verified", False)),
                }
                for name, spec in c.special.items()
            },
            "consoles": {
                name: {
                    "type": spec.get("gimx_type"),
                    "default": bool(spec.get("default")),
                }
                for name, spec in c.consoles.items()
            },
            "timing": dict(c.timing),
        }

    # -- pad ---------------------------------------------------------------
    def pad(self, console: str | None = None) -> Any:
        """The ConsolePad. Honours runtime.dry_run so plans can be rehearsed."""
        with self._lock:
            if self._pad is None:
                cls = self.adapter("pad").get("ConsolePad")
                self._pad = cls(self.controls, console, self.dry_run)
            return self._pad

    # -- capture -----------------------------------------------------------
    def capture(self) -> Any:
        """The ScreenCapture handle, opened once and reused."""
        with self._lock:
            if self._capture is None:
                cls = self.adapter("capture").get("ScreenCapture")
                self._capture = cls(config_path=self.controls_path)
                self._capture.open()
            return self._capture

    def capture_functions(self) -> dict[str, Any]:
        """Frame maths helpers from capture.py, reused rather than reimplemented.

        The thresholds in there were derived from measurements on this exact
        card; a second implementation here would be a guess.
        """
        a = self.adapter("capture")
        return {
            "preflight": a.get("preflight"),
            "frame_stats": a.get("frame_stats"),
            "is_blank": a.get("is_blank"),
            "difference": a.get("difference"),
        }

    # -- gimx --------------------------------------------------------------
    def gimx_functions(self) -> dict[str, Any]:
        a = self.adapter("gimx")
        return {
            "session_is_up": a.get("session_is_up"),
            "running_gimx_pids": a.get("running_gimx_pids"),
            "stop_all": a.get("stop_all"),
            "GimxSession": a.get("GimxSession"),
        }

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """Release the capture device.

        Important: while we hold it, nothing else can open it - including
        RECentral, which a human may want for an eyeball check after the run.
        """
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.close()
                except Exception:
                    pass
                self._capture = None

    def __enter__(self) -> "HardwareBridge":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
