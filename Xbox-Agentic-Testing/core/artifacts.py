"""
artifacts.py - per-run evidence store.

Every run gets its own directory:

    artifacts/runs/<run_id>/
        frames/      before/after screenshots, named by step
        logs/        raw tool output, GIMX lines
        reports/     json / markdown / junit
        state.json   the final state digest

WHY THIS MATTERS
----------------
The verdict is only as trustworthy as the evidence behind it. Frames are saved
BEFORE the verifier ever looks at them, so a human can open the folder and check
the machine's reasoning against the same pictures it used. A framework that
reports "pass" with nothing to inspect is asking to be believed on faith, which
is exactly the failure mode docs 07 warns about.

Filenames are ordered and descriptive (`step-003_after_press-a.png`) so the
sequence reads correctly in a file listing without opening anything.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str, max_length: int = 40) -> str:
    """Filesystem-safe fragment of arbitrary text."""
    cleaned = _SAFE.sub("-", str(text).strip()).strip("-").lower()
    return (cleaned[:max_length] or "item").rstrip("-")


def new_run_id(prefix: str = "run") -> str:
    """Sortable, unique-per-second run id."""
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


class ArtifactStore:
    """Owns one run's output directory."""

    def __init__(self, root: Path | str, run_id: str,
                 frame_format: str = "png", enabled: bool = True):
        self.run_id = run_id
        self.enabled = enabled
        self.frame_format = frame_format.lstrip(".") or "png"

        self.run_dir = Path(root) / "runs" / run_id
        self.frames_dir = self.run_dir / "frames"
        self.logs_dir = self.run_dir / "logs"
        self.reports_dir = self.run_dir / "reports"

        if self.enabled:
            for d in (self.frames_dir, self.logs_dir, self.reports_dir):
                d.mkdir(parents=True, exist_ok=True)

        self.files: list[str] = []

    # -- frames ------------------------------------------------------------
    def save_frame(self, frame: Any, label: str,
                   step: int | None = None) -> str | None:
        """Write a numpy frame to disk and return its path.

        Returns None (never raises) when saving is off or the frame is absent,
        because a screenshot failing to save must not abort a hardware run
        mid-sequence - the step result simply carries no frame path.
        """
        if not self.enabled or frame is None:
            return None
        try:
            import cv2
        except ImportError:
            return None

        prefix = f"step-{step:03d}_" if step is not None else ""
        name = f"{prefix}{slug(label)}.{self.frame_format}"
        path = self.frames_dir / name
        try:
            ok = bool(cv2.imwrite(str(path), frame))
        except Exception:
            return None
        if not ok:
            return None
        self.files.append(str(path))
        return str(path)

    # -- text / json -------------------------------------------------------
    def save_text(self, name: str, content: str,
                  subdir: str = "logs") -> str | None:
        if not self.enabled:
            return None
        path = self._dir(subdir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.files.append(str(path))
        return str(path)

    def save_json(self, name: str, payload: Any,
                  subdir: str = "reports") -> str | None:
        if not self.enabled:
            return None
        # default=str keeps enums, datetimes and stray objects from blowing up
        # serialisation - a partially readable artifact beats an exception.
        text = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
        return self.save_text(name, text, subdir=subdir)

    def append_log(self, name: str, line: str) -> None:
        if not self.enabled:
            return
        path = self.logs_dir / name
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line.rstrip() + "\n")
        if str(path) not in self.files:
            self.files.append(str(path))

    # -- helpers -----------------------------------------------------------
    def _dir(self, subdir: str) -> Path:
        return {
            "frames": self.frames_dir,
            "logs": self.logs_dir,
            "reports": self.reports_dir,
        }.get(subdir, self.run_dir)

    def list_frames(self) -> list[str]:
        if not self.enabled or not self.frames_dir.is_dir():
            return []
        return sorted(str(p) for p in self.frames_dir.iterdir() if p.is_file())

    def relative(self, path: str | Path) -> str:
        """Path relative to the run dir. For display, not for links."""
        try:
            return str(Path(path).relative_to(self.run_dir).as_posix())
        except ValueError:
            return str(path)

    def link_from_reports(self, path: str | Path) -> str:
        """A link that works from inside a file in `reports/`.

        Reports live in `reports/` and frames in `frames/`, so a link relative
        to the RUN directory ("frames/x.png") resolves to
        "reports/frames/x.png" and shows a broken image. It needs "../frames/".

        This bug is easy to miss because the report renders fine structurally -
        the images are simply absent, and a report whose evidence does not load
        is a report nobody can check.
        """
        try:
            return str(Path(path).relative_to(self.reports_dir).as_posix())
        except ValueError:
            pass
        try:
            inside_run = Path(path).relative_to(self.run_dir)
            return str(Path("..").joinpath(inside_run).as_posix())
        except ValueError:
            # Outside the run directory entirely - an absolute path is the only
            # thing that can work.
            return Path(path).as_posix()
