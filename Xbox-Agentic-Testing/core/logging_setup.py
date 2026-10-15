"""
logging_setup.py - one place that configures logging for every entry point.

Before this module, settings.yaml had a `logging:` section (level, log_file)
that nothing read, so LOG_LEVEL/LOG_FILE did nothing and a run left no log
file: the live terminal was the only record.

Design
------
* Python `logging` is configured once per process from settings (level,
  persistent log file) with an optional --log-level override.
* Each run additionally gets `<run_dir>/logs/run.log` (attach_run_log).
* The framework has ~300 operator-facing print() calls whose exact layout is
  the operator's live view. Rewriting them all at once would risk changing it,
  so stdout is TEED instead: everything printed still appears on the terminal
  unchanged, and the same lines are written, timestamped, to the log files
  via the "console" logger. New code should use logging.getLogger(__name__);
  prints can migrate gradually without losing anything in the meantime.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_CONSOLE_LOGGER = "console"
_configured = False
_run_handler: logging.Handler | None = None


class _StdoutTee(io.TextIOBase):
    """Writes through to the real stream and mirrors complete lines to a logger."""

    def __init__(self, stream: TextIO, logger: logging.Logger):
        self._stream = stream
        self._logger = logger
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        written = self._stream.write(text)
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self._logger.info(line.rstrip())
        return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._stream, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._stream.fileno()


def configure(settings: Any = None, level: str | None = None,
              base_dir: Path | None = None) -> None:
    """Configure root logging once. Safe to call again (no duplicate handlers)."""
    global _configured
    level_name = (level
                  or (settings.get("logging.level", "INFO") if settings is not None else None)
                  or "INFO")
    numeric = getattr(logging, str(level_name).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)

    if _configured:
        return
    _configured = True

    # Framework log records go to stderr only at WARNING+, so the terminal
    # keeps its current print()-based layout; everything goes to the files.
    stderr = logging.StreamHandler(sys.__stderr__)
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    stderr.addFilter(lambda record: record.name != _CONSOLE_LOGGER)
    root.addHandler(stderr)

    log_file = settings.get("logging.log_file", "") if settings is not None else ""
    if log_file:
        path = Path(str(log_file))
        if not path.is_absolute() and base_dir is not None:
            path = (base_dir / path).resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(handler)
        except OSError as exc:
            logging.getLogger(__name__).warning("Cannot open log file %s: %s", path, exc)

    console_logger = logging.getLogger(_CONSOLE_LOGGER)
    console_logger.propagate = True
    if not isinstance(sys.stdout, _StdoutTee):
        sys.stdout = _StdoutTee(sys.stdout, console_logger)

    for noisy in ("httpx", "httpcore", "anthropic", "urllib3", "langsmith"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))


def attach_run_log(run_dir: Path | str) -> Path | None:
    """Start writing <run_dir>/logs/run.log; replaces the previous run's handler."""
    global _run_handler
    detach_run_log()
    path = Path(run_dir) / "logs" / "run.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        return None
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.getLogger().addHandler(handler)
    _run_handler = handler
    return path


def detach_run_log() -> None:
    global _run_handler
    if _run_handler is not None:
        logging.getLogger().removeHandler(_run_handler)
        _run_handler.close()
        _run_handler = None
