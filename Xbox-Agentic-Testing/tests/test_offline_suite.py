"""Run every offline regression script as a pytest test.

The scripts in tests/offline/ assert at module level and monkeypatch modules
(navigation_tools, vision_tools) globally, so importing them into one pytest
process would let one test's fakes leak into the next. Each script therefore
runs in its own interpreter; a non-zero exit is a failure, and its output is
shown in the pytest report.

None of these touch hardware, an LLM, or the network.

    python -m pytest tests            # whole suite
    python tests/offline/test_x.py    # one script directly, as before
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

OFFLINE = Path(__file__).resolve().parent / "offline"
PROJECT = OFFLINE.parent.parent
SCRIPTS = sorted(OFFLINE.glob("test_*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.stem for s in SCRIPTS])
def test_offline_script(script: Path) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "LEARNING_ENABLED": "true"}
    # Under `pytest --cov`, run the script under coverage too (parallel data
    # file) so modules only exercised by offline scripts are not reported 0%.
    command = [sys.executable, str(script)]
    if os.environ.get("COV_CORE_SOURCE") is not None or "--cov" in " ".join(sys.argv):
        command = [sys.executable, "-m", "coverage", "run", "--parallel-mode",
                   "--rcfile", str(PROJECT / "pyproject.toml"), str(script)]
    proc = subprocess.run(
        command, cwd=PROJECT, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-40:])
        pytest.fail(f"{script.name} exited {proc.returncode}\n{tail}", pytrace=False)


def test_suite_is_not_empty() -> None:
    assert len(SCRIPTS) >= 9, f"expected the offline regression scripts in {OFFLINE}"
