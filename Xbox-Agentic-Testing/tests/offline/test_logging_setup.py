"""Logging: prints still reach the terminal unchanged AND land in run.log."""

import io
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import logging_setup


class _Settings:
    def __init__(self, log_file):
        self.values = {"logging.level": "INFO", "logging.log_file": str(log_file)}

    def get(self, key, default=None):
        return self.values.get(key, default)


tmp = Path(tempfile.mkdtemp())
real_stdout = sys.stdout
captured = io.StringIO()
sys.stdout = captured                      # stands in for the terminal

logging_setup.configure(_Settings(tmp / "agentic.log"), base_dir=tmp)
logging_setup.configure(_Settings(tmp / "agentic.log"), base_dir=tmp)   # idempotent
run_log = logging_setup.attach_run_log(tmp / "run-1")

print("[Step 1/2] Action: press_button")
print("  Result:    OK (0.20s)")
logging.getLogger("executor").warning("capture lost, retrying")
logging_setup.detach_run_log()
print("after detach")

sys.stdout.flush()
terminal = captured.getvalue()
sys.stdout = real_stdout

# 1. Terminal output is byte-for-byte what was printed.
assert terminal == "[Step 1/2] Action: press_button\n  Result:    OK (0.20s)\nafter detach\n", terminal

# 2. run.log has the prints and the logger record, timestamped.
text = run_log.read_text(encoding="utf-8")
assert "console: [Step 1/2] Action: press_button" in text
assert "WARNING executor: capture lost, retrying" in text
assert "after detach" not in text, "run log must stop at detach"

# 3. Persistent log file gets everything; handlers not duplicated.
persistent = (tmp / "agentic.log").read_text(encoding="utf-8")
assert persistent.count("Action: press_button") == 1
assert "after detach" in persistent

print("logging setup tests passed")
