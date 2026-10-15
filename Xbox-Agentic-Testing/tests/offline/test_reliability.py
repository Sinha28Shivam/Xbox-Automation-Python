"""Reliability: OCR-tolerant matching, config-driven screen signatures, RCA
product-defect evidence rule. OCR strings are real misreads from this rig."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.executor_agent import ExecutorAgent, _screen_signatures
from agents.rca_agent import RCAAgent
from core import text_match as tm
from core.schemas import ExecutionResult, FailureClass, RootCauseAnalysis, StepResult

# ---- text_match --------------------------------------------------------------
# Screen identification tolerates OCR noise...
assert tm.contains("Minecraft Dundecns", "minecraft dungeons")
assert tm.contains("Tel Go to official club", "Go to official club")
assert tm.contains("SPONSORED", "sponsored")
# ...but short words need an exact match.
assert not tm.contains("Status", "Stats")
assert not tm.contains("Start a party", "Stats")
# Input decisions use the exact level: case/punctuation only.
assert tm.exact_contains("Game activity", "Game Activity")
assert tm.exact_contains("@ Go to official club", "Go to official club")
assert not tm.exact_contains("Achievement tracker", "Achievements"), \
    "fuzzy scored this 0.96 - it must never stop navigation on the wrong row"
assert tm.exact_matched_lines("Achievements\nMore achievements\nStats", "Stats") == ["Stats"]

# ---- screen signatures come from YAML -------------------------------------
sigs = _screen_signatures()
assert "minecraft dungeons" in sigs["in_game"] and "curse of brotherhood" in sigs["in_game"]
assert "sponsored" in sigs["dashboard"] and "buy $" in sigs["store"]


def _guard(ocr):
    return ExecutorAgent._guard_ingame_screen([StepResult(index=0, action="x", ocr_text=ocr)])


# Dashboard without a game -> blocked.
assert not _guard("Game on\nExclusive deals\nSPONSORED")["ok"]
# Dungeons running with OCR misread of its name -> still recognised, not blocked.
assert _guard("SPONSORED\nMinecraft Dundecns\nMission select")["ok"]
# Store wall -> blocked.
assert not _guard("Join Game Pass\nBuy $69.99")["ok"]

# ---- RCA: product defect needs console evidence ----------------------------
def _rca(execution):
    rca = RootCauseAnalysis(scenario_id="s", failure_class=FailureClass.PRODUCT_DEFECT,
                            primary_cause="console misbehaved", product_bug_suspected=True)
    agent = RCAAgent.__new__(RCAAgent)
    return agent._apply_guardrails(rca, {"execution": execution},
                                   {"gimx": {"reachable": True}, "capture": {"has_signal": True}})


def _press(i, delta, ocr=None):
    return StepResult(index=i, action="press_button", dispatched=True,
                      screen_delta=delta, ocr_text=ocr)


# Responsive console, no error dialog -> demoted to automation_defect.
out = _rca(ExecutionResult(scenario_id="s", steps=[_press(0, 30.0), _press(1, 12.0)]))
assert out.failure_class == FailureClass.AUTOMATION_DEFECT and not out.product_bug_suspected

# Error dialog on screen -> product defect kept.
out = _rca(ExecutionResult(scenario_id="s", steps=[
    _press(0, 30.0, "Something went wrong\nError code 0x87e10bc6")]))
assert out.failure_class == FailureClass.PRODUCT_DEFECT

# Console stopped responding after working -> product defect kept.
out = _rca(ExecutionResult(scenario_id="s", steps=[
    _press(0, 30.0), _press(1, 0.0), _press(2, 0.01), _press(3, 0.0)]))
assert out.failure_class == FailureClass.PRODUCT_DEFECT

# Never responded at all is a rig problem, not "stopped responding".
out = _rca(ExecutionResult(scenario_id="s", steps=[_press(i, 0.0) for i in range(4)]))
assert out.failure_class != FailureClass.PRODUCT_DEFECT

print("reliability tests passed")
