"""Persistent gameplay-session state.

A scenario run is intentionally bounded, while a level/campaign may span many
runs. This small JSON-backed state object keeps progress, checkpoints and the
last verified screen independent from an individual agent invocation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GameSession:
    game_name: str
    mode: str = "level"
    level: str = ""
    checkpoint: str = ""
    objective: str = ""
    status: str = "running"
    completed_levels: list[str] = field(default_factory=list)
    attempts: int = 0
    last_verified_frame: str = ""
    last_verdict: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.time()
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GameSession":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def checkpoint_now(self, *, level: str | None = None,
                       checkpoint: str | None = None,
                       frame: str | None = None,
                       verdict: str | None = None) -> None:
        if level is not None:
            self.level = level
        if checkpoint is not None:
            self.checkpoint = checkpoint
        if frame is not None:
            self.last_verified_frame = frame
        if verdict is not None:
            self.last_verdict = verdict
        self.attempts += 1
        self.updated_at = time.time()

    def mark_level_complete(self, level: str) -> None:
        if level not in self.completed_levels:
            self.completed_levels.append(level)
        self.level = level
        self.checkpoint = "complete"
        self.status = "running"
        self.updated_at = time.time()

    def mark_complete(self) -> None:
        self.status = "complete"
        self.updated_at = time.time()

    def mark_blocked(self) -> None:
        self.status = "blocked"
        self.updated_at = time.time()
