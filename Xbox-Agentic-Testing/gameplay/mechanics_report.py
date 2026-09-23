"""
mechanics_report.py - the game-mechanics report for a gameplay session.

WHAT THIS ANSWERS
-----------------
Not "did the run finish" but "which game mechanics were PROVEN to work". For
Sea of Sand that is six specific capabilities - move, open the marker, draw a
pillar, erase a pillar, jump, and get on top of a pillar - each with its own
verdict and its own screenshots.

FOUR VERDICTS, AND WHY NOT TWO
------------------------------
A pass/fail column would be dishonest here, because "was not attempted" and
"was attempted and did not work" are completely different findings:

    PASS          dispatched AND the screen proves it happened
    FAIL          attempted, but the evidence says it did not work
    NOT TESTED    never attempted in this session - proves nothing either way
    INCONCLUSIVE  the action log and the vision model disagree

Printing FAIL for a mechanic the run never tried would be a false negative,
and it is the same mistake as reporting a broken rig as a product failure.
In a short session `destroy_pillars` and `reach_pillars` genuinely may never
come up, so they must be reportable as untested.

TWO INDEPENDENT PASSES
----------------------
1. DETERMINISTIC: read trace.json - which macros were dispatched, and did the
   measured frame delta clear this rig's ambient noise floor? Cheap, exact,
   and it cannot hallucinate. But a dispatched action is not a working
   mechanic: the pad can fire while nothing happens on screen.
2. AI DIAGNOSTICS: show the before/after frame pair to the vision model and
   ask whether it can SEE the mechanic happening. This catches the
   dispatched-but-ineffective case that the log alone calls a success.

They are reconciled, not averaged. If the log says PASS and the model cannot
see it, the answer is INCONCLUSIVE - because we genuinely do not know.

USAGE
    python -m gameplay.mechanics_report artifacts/gameplay/play-20260907-193052
"""
from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "core"), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===========================================================================
# The mechanics under test
# ===========================================================================
@dataclass
class Mechanic:
    """One capability, and what would count as proof of it."""

    key: str
    title: str
    # Macros from GameplayAction that exercise this mechanic.
    actions: tuple[str, ...]
    # What the vision model is asked to look for in a before/after pair.
    question: str
    needs_drawing: bool = False
    needs_elevation: bool = False


MECHANICS: tuple[Mechanic, ...] = (
    Mechanic(
        key="move_left_right",
        title="Character is able to move left right",
        actions=("move",),
        question=(
            "Has Max (the small boy in a blue hoodie with orange hair) "
            "CHANGED POSITION horizontally between these two frames? Look at "
            "his position relative to fixed scenery such as rocks, trees and "
            "ledges. Camera drift alone does not count - his position in the "
            "world must have changed."),
    ),
    Mechanic(
        key="use_magic_marker",
        title="Character is able to use Magic marker",
        actions=("magic_marker", "destroy_drawing"),
        question=(
            "Is the MAGIC MARKER OPEN in the second frame? Signs: a round "
            "pale cursor/reticle on screen, a desaturated or slowed scene, an "
            "ink gauge (a pale capsule holding orange fluid), or glowing "
            "orange nodes becoming prominent. The marker being open is the "
            "mechanic - a drawing does not have to exist yet."),
    ),
    Mechanic(
        key="draw_pillars",
        title="Character is able to draw pillers using Magic Marker",
        actions=("magic_marker",),
        needs_drawing=True,
        question=(
            "Has a NEW EARTH PILLAR (a brown/orange column of earth rising "
            "from the ground) APPEARED in the second frame that was not in "
            "the first? A player-drawn structure often carries a small blue "
            "'X' badge meaning erasable. Answer no if the terrain is "
            "unchanged."),
    ),
    Mechanic(
        key="destroy_pillars",
        title="Character is able to destroy Pillers using Magic marker",
        actions=("destroy_drawing",),
        question=(
            "Has an EARTH PILLAR that was present in the first frame been "
            "REMOVED in the second? You should see the column gone and the "
            "terrain behind it restored."),
    ),
    Mechanic(
        key="jump",
        title="Character is able to Jump",
        actions=("jump", "running_jump", "edge_jump_grab"),
        question=(
            "Is Max AIRBORNE or at a different HEIGHT in the second frame "
            "compared to the first? Look for his feet off the ground, a jump "
            "pose, or him standing on something higher than before."),
    ),
    Mechanic(
        key="reach_pillars",
        title="Character is able to reach the pillers",
        actions=("climb_or_pull_up", "edge_jump_grab", "running_jump", "jump"),
        needs_elevation=True,
        question=(
            "Is Max ON TOP OF, or gripping, a drawn EARTH PILLAR or a raised "
            "ledge in the second frame, when he was on lower ground in the "
            "first? Standing next to a pillar does NOT count - he must be on "
            "it or hanging from it."),
    ),
)

Verdict = Literal["PASS", "FAIL", "NOT TESTED", "INCONCLUSIVE"]


class PairJudgement(BaseModel):
    """The vision model's reading of one before/after frame pair."""

    happened: bool = Field(
        description="True only if the described mechanic is VISIBLE in these "
                    "two frames. If you cannot tell, answer false and say so "
                    "in evidence.")
    evidence: str = Field(
        description="What you actually saw that supports the answer, quoting "
                    "concrete visual detail (positions, new structures, "
                    "on-screen UI). Say plainly when the frames are "
                    "ambiguous.")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence in this judgement.")


# ===========================================================================
# Deterministic pass: what does the action log prove on its own?
# ===========================================================================
# This rig's noise floor, from autonomous_player._delta_verdict: an IDLE
# screen already measures 2-3 because foliage sways and dust drifts, and Max
# merely walking only reaches about 4. So a delta below this is NOT evidence
# of anything, however plausible the number looks.
AMBIENT_DELTA = 4.5
# A drawing appearing or a camera pan moves a large share of the frame. The
# measured figure for a real pillar was ~1.18M px against 47K for a failure.
STRUCTURAL_DELTA = 12.0


def load_session(session_dir: Path) -> dict[str, Any]:
    """Read session.json and trace.json. Either may be missing."""
    session_dir = Path(session_dir)
    out: dict[str, Any] = {"session_dir": session_dir, "steps": [],
                           "session": {}, "has_trace": False}

    sfile = session_dir / "session.json"
    if sfile.is_file():
        try:
            out["session"] = json.loads(sfile.read_text(encoding="utf-8"))
        except Exception:
            pass

    tfile = session_dir / "trace.json"
    if tfile.is_file():
        try:
            trace = json.loads(tfile.read_text(encoding="utf-8"))
            out["steps"] = list(trace.get("steps") or [])
            out["level"] = trace.get("level", "")
            out["game_name"] = trace.get("game_name", "")
            out["launch_trace"] = trace.get("launch") or {}
            out["has_trace"] = True
        except Exception:
            pass

    if not out.get("game_name"):
        out["game_name"] = out["session"].get(
            "game_name", "Max: The Curse of Brotherhood")
    if not out.get("level"):
        out["level"] = (out["session"].get("metadata", {})
                        .get("launch_target_level", ""))
    return out


def _step_actions(step: dict[str, Any]) -> list[str]:
    return [str(a.get("action", "")) for a in (step.get("actions") or [])]


def _mentions_drawing(step: dict[str, Any]) -> bool:
    """Did the model report a drawing/pillar in the scene it described?"""
    blob = " ".join([
        str(step.get("interactive_elements", "")),
        str(step.get("tactical_reasoning", "")),
        str(step.get("player_location", "")),
    ]).lower()
    return any(w in blob for w in ("pillar", "piller", "drawn", "drawing",
                                   "structure i drew", "my pillar"))


def _mentions_elevation(step: dict[str, Any]) -> bool:
    """Did the model describe Max being on top of something?"""
    blob = " ".join([
        str(step.get("player_location", "")),
        str(step.get("tactical_reasoning", "")),
    ]).lower()
    return any(w in blob for w in (
        "on top", "standing on the pillar", "on the pillar", "climbed",
        "hanging", "gripping", "upper platform", "higher ground",
        "reached the ledge", "on the ledge"))


def candidates(mech: Mechanic, steps: list[dict[str, Any]],
               limit: int) -> list[dict[str, Any]]:
    """Steps that exercised this mechanic, best evidence first.

    Ordered by measured delta so the strongest candidate is judged first -
    the AI pass is capped, and spending it on the most promising pair gives
    the mechanic its fairest hearing.
    """
    hits = [s for s in steps
            if any(a in mech.actions for a in _step_actions(s))
            and s.get("scene_state") == "in_gameplay"]

    if mech.needs_drawing:
        # A marker stroke only proves DRAWING if something appeared. Prefer
        # steps whose delta is large enough for a new structure.
        hits = [s for s in hits
                if (s.get("delta") or 0) >= STRUCTURAL_DELTA
                or _mentions_drawing(s)]
    if mech.needs_elevation:
        hits = [s for s in hits if _mentions_elevation(s)]

    hits.sort(key=lambda s: float(s.get("delta") or 0.0), reverse=True)
    return hits[:limit]


def deterministic_verdict(mech: Mechanic,
                          hits: list[dict[str, Any]]) -> tuple[Verdict, str]:
    """Classify from the action log alone. Never uses the model."""
    if not hits:
        return ("NOT TESTED",
                f"No '{'/'.join(mech.actions)}' macro was dispatched during "
                f"gameplay in this session.")

    moved = [s for s in hits if float(s.get("delta") or 0.0) >= AMBIENT_DELTA]
    if not moved:
        best = max(float(s.get("delta") or 0.0) for s in hits)
        return ("FAIL",
                f"Dispatched {len(hits)} time(s), but the largest screen "
                f"change was {best:.1f}, at or below this rig's ambient noise "
                f"floor of {AMBIENT_DELTA} - so nothing measurably happened.")

    best = max(float(s.get("delta") or 0.0) for s in moved)
    return ("PASS",
            f"Dispatched {len(hits)} time(s); {len(moved)} produced a screen "
            f"change above the {AMBIENT_DELTA} noise floor (best "
            f"{best:.1f}).")


# ===========================================================================
# AI diagnostics: can the model SEE the mechanic in the frames?
# ===========================================================================
JUDGE_PROMPT = """\
You are auditing an automated test of "Max: The Curse of Brotherhood". Two
frames are attached, captured from a real Xbox One a moment apart:

  FRAME 1 = BEFORE the controller action
  FRAME 2 = AFTER the controller action

The action dispatched was: {action}

THE QUESTION
{question}

HOW TO ANSWER HONESTLY
  * Answer about THESE TWO IMAGES only. Do not assume the action worked just
    because it was dispatched - proving that is the entire point of this
    audit, and a dispatched button that changed nothing is exactly the bug
    we are looking for.
  * If the frames are too similar, too dark, or too ambiguous to tell, answer
    happened=false and say WHY in evidence. "I cannot tell" is a useful,
    respectable answer; a confident guess is not.
  * Quote concrete visual detail in evidence - where Max is relative to
    scenery, what appeared or vanished, what UI is on screen.
"""


def _encode(path: str | Path, width: int = 1280) -> str | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode("ascii") if ok else None


class Judge:
    """Wraps the vision model. Absent model = no diagnostics, not a crash."""

    def __init__(self, settings: Any, llm_factory: Any = None):
        self.settings = settings
        self._factory = llm_factory
        self._runnable: Any = None
        self.available = True
        self.error = ""

    def _build(self) -> Any:
        from llm import LLMFactory, structured

        factory = self._factory or LLMFactory(self.settings)
        provider = factory.default_provider
        if not factory.supports_vision(provider):
            raise RuntimeError(
                f"provider '{provider}' is not marked supports_vision")
        return structured(factory.build(provider=provider), PairJudgement)

    def judge(self, mech: Mechanic, step: dict[str, Any]) -> dict[str, Any]:
        """Judge one step's frame pair. Returns a plain dict for the report."""
        before, after = step.get("frame"), step.get("frame_after")
        if not before or not after:
            return {"ok": False,
                    "reason": "no before/after frame pair was saved"}

        b64_before, b64_after = _encode(before), _encode(after)
        if not b64_before or not b64_after:
            return {"ok": False, "reason": "frames could not be read"}

        if self._runnable is None:
            try:
                self._runnable = self._build()
            except Exception as exc:
                self.available = False
                self.error = str(exc)
                return {"ok": False,
                        "reason": f"vision model unavailable: {exc}"}

        actions = ", ".join(
            f"{a.get('action')}({a.get('direction', 'none')})"
            for a in (step.get("actions") or [])) or "unknown"
        prompt = JUDGE_PROMPT.format(action=actions, question=mech.question)

        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "text", "text": "FRAME 1 (before):"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64_before}"}},
            {"type": "text", "text": "FRAME 2 (after):"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64_after}"}},
        ])

        try:
            res: PairJudgement = self._runnable.invoke([msg])
        except Exception as exc:
            return {"ok": False, "reason": f"model error: {exc}"}

        return {"ok": True, "step": step.get("step"),
                "happened": bool(res.happened), "evidence": res.evidence,
                "confidence": round(float(res.confidence), 2),
                "frame_before": before, "frame_after": after}


def reconcile(log_verdict: Verdict,
              judgements: list[dict[str, Any]]) -> tuple[Verdict, str]:
    """Combine the two passes. Disagreement means INCONCLUSIVE, not a guess.

    The log knows what was DISPATCHED; the model knows what is VISIBLE. When
    they conflict, asserting either one would be inventing certainty we do
    not have - and a false PASS is the failure mode this framework exists to
    prevent.
    """
    usable = [j for j in judgements if j.get("ok")]
    if not usable:
        why = (judgements[0].get("reason", "no diagnostics")
               if judgements else "no candidate steps to judge")
        return log_verdict, f"AI diagnostics unavailable ({why})."

    seen = [j for j in usable if j.get("happened")]

    if log_verdict == "NOT TESTED":
        # Nothing was dispatched, so there is nothing for vision to confirm.
        return log_verdict, "Not attempted, so no visual audit was performed."

    if log_verdict == "PASS":
        if seen:
            best = max(seen, key=lambda j: j.get("confidence", 0.0))
            return ("PASS",
                    f"AI diagnostics CONFIRM this on step {best['step']} "
                    f"(confidence {best['confidence']:.2f}): "
                    f"{best['evidence']}")
        worst = usable[0]
        return ("INCONCLUSIVE",
                f"The action was dispatched and the screen changed, but AI "
                f"diagnostics could not SEE the mechanic in {len(usable)} "
                f"frame pair(s). Example (step {worst['step']}): "
                f"{worst['evidence']}")

    # log said FAIL
    if seen:
        best = max(seen, key=lambda j: j.get("confidence", 0.0))
        return ("INCONCLUSIVE",
                f"The measured screen change was below the noise floor, yet "
                f"AI diagnostics report seeing it on step {best['step']}: "
                f"{best['evidence']}")
    return ("FAIL",
            f"AI diagnostics AGREE it did not happen. Example: "
            f"{usable[0]['evidence']}")


# ===========================================================================
# Analysis
# ===========================================================================
def analyse(session_dir: Path, settings: Any = None, llm_factory: Any = None,
            max_pairs: int = 2, use_ai: bool = True) -> dict[str, Any]:
    """Run both passes over a session and return the full report payload."""
    data = load_session(session_dir)
    steps = data["steps"]
    judge = Judge(settings, llm_factory) if (use_ai and settings) else None

    results: list[dict[str, Any]] = []
    for mech in MECHANICS:
        hits = candidates(mech, steps, limit=max_pairs)
        log_verdict, log_note = deterministic_verdict(mech, hits)

        judgements: list[dict[str, Any]] = []
        # No point spending vision calls on something never attempted.
        if judge is not None and log_verdict != "NOT TESTED":
            for step in hits:
                if not judge.available:
                    break
                print(f"  [ai] {mech.key}: judging step {step.get('step')} ...",
                      flush=True)
                judgements.append(judge.judge(mech, step))

        verdict, ai_note = reconcile(log_verdict, judgements)
        results.append({
            "key": mech.key,
            "title": mech.title,
            "verdict": verdict,
            "log_verdict": log_verdict,
            "log_note": log_note,
            "ai_note": ai_note,
            "steps": [s.get("step") for s in hits],
            "evidence": [j for j in judgements if j.get("ok")],
            "candidates": [
                {"step": s.get("step"),
                 "actions": _step_actions(s),
                 "delta": s.get("delta"),
                 "delta_verdict": s.get("delta_verdict"),
                 "reasoning": s.get("tactical_reasoning", ""),
                 "frame": s.get("frame"),
                 "frame_after": s.get("frame_after")}
                for s in hits],
        })

    meta = data["session"].get("metadata", {})
    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": data["session"].get("session_id",
                                          Path(session_dir).name),
        "session_dir": str(session_dir),
        "game_name": data["game_name"],
        "level": data["level"] or "Sea Of Sand",
        "chapter": "Chapter 1",
        "has_trace": data["has_trace"],
        "launch": {
            "from_dashboard": bool(meta.get("launched_from_dashboard")),
            "mode": meta.get("launch_mode", ""),
            "target_level": meta.get("launch_target_level", ""),
            **(data.get("launch_trace") or {}),
        },
        "totals": {
            "steps": meta.get("total_steps", len(steps)),
            "deaths": meta.get("deaths", 0),
            "duration_seconds": meta.get("duration_seconds", 0.0),
        },
        "counts": counts,
        "mechanics": results,
    }


# ===========================================================================
# Markdown
# ===========================================================================
_MARK = {"PASS": "PASS", "FAIL": "FAILED",
         "NOT TESTED": "NOT TESTED", "INCONCLUSIVE": "INCONCLUSIVE"}


def _rel(path: str | None, base: Path) -> str:
    """A link that works from inside the report's own reports/ directory.

    Gameplay frames sit in <session>/frames, so "../frames/x.jpg". Launch
    frames are written by the ArtifactStore into artifacts/runs/<id>/frames
    instead, which is OUTSIDE the session dir - a report that linked those
    absolutely would break the moment the folder was copied or shared, so
    they get a relative path computed from the reports dir.
    """
    if not path:
        return ""
    try:
        return Path("..").joinpath(
            Path(path).resolve().relative_to(base.resolve())).as_posix()
    except ValueError:
        pass
    try:
        import os
        return Path(os.path.relpath(Path(path).resolve(),
                                    (base / "reports").resolve())).as_posix()
    except Exception:
        return Path(path).as_posix()


def markdown(r: dict[str, Any]) -> str:
    base = Path(r["session_dir"])
    out: list[str] = []
    a = out.append

    a(f"# Game Mechanics of {r['level']} {r['chapter'].lower()} "
      f"({r['game_name']})")
    a("")
    t = r["totals"]
    a(f"Session `{r['session_id']}` - {t['steps']} gameplay steps, "
      f"{t['deaths']} death(s), {float(t['duration_seconds']):.0f}s.")
    if r["launch"]["from_dashboard"]:
        a(f"Launched from the Xbox dashboard into "
          f"**{r['launch']['target_level']}** "
          f"({r['launch']['mode']}-guided menu navigation).")
    a("")

    # ---- the six lines, exactly as requested -------------------------------
    for i, m in enumerate(r["mechanics"], 1):
        a(f"{i}. {m['title']}. **{_MARK[m['verdict']]}**")
    a("")

    c = r["counts"]
    a(f"**{c.get('PASS', 0)} passed, {c.get('FAIL', 0)} failed, "
      f"{c.get('NOT TESTED', 0)} not tested, "
      f"{c.get('INCONCLUSIVE', 0)} inconclusive.**")
    a("")

    if not r["has_trace"]:
        a("> **No `trace.json` in this session.** It predates per-step trace "
          "recording, so the action log could not be read and every verdict "
          "below rests on frames alone. Re-run `play` to get full fidelity.")
        a("")

    a("## Verdict meanings")
    a("")
    a("| verdict | means |")
    a("|---|---|")
    a("| PASS | The macro was dispatched AND the evidence shows it worked. |")
    a("| FAILED | It was attempted, but the evidence shows it did not work. |")
    a("| NOT TESTED | Never attempted this session - proves nothing either "
      "way. This is deliberately NOT reported as a failure. |")
    a("| INCONCLUSIVE | The action log and the AI audit disagree. |")
    a("")

    # ---- launch journey ---------------------------------------------------
    launch = r["launch"]
    lsteps = launch.get("steps") or []
    if lsteps:
        ok = launch.get("ok")
        a("## Launch: dashboard to gameplay")
        a("")
        a(f"**{'REACHED GAMEPLAY' if ok else 'DID NOT COMPLETE'}** in "
          f"{launch.get('cycles', len(lsteps))} menu screens. "
          f"Target level: **{launch.get('target', launch.get('target_level'))}**.")
        a("")
        if not launch.get("target_confirmed", True):
            a("> The target level was never explicitly confirmed on a level "
              "picker, so which level actually started is UNVERIFIED - "
              "gameplay beginning is not by itself proof of the right level.")
            a("")
        a(f"- {launch.get('reason', '')}")
        a("")
        a("| # | screen | focused | input | note |")
        a("|---|---|---|---|---|")
        for s in lsteps:
            rep = s.get("dispatched_repeat", 1) or 1
            act = s.get("dispatched", s.get("action", ""))
            act_txt = f"`{act}`" + (f" x{rep}" if rep > 1 else "")
            note = "vetoed confirm" if s.get("veto") else ""
            a(f"| {s.get('cycle')} | {s.get('screen', '')} | "
              f"{s.get('focused_item') or '-'} | {act_txt} | {note} |")
        a("")
        for s in lsteps:
            img = _rel(s.get("frame"), base)
            if not img:
                continue
            a(f"**Screen {s.get('cycle')} - {s.get('screen', '')}**"
              + (f" - focused: {s['focused_item']}"
                 if s.get("focused_item") else ""))
            a("")
            a(f"![launch screen {s.get('cycle')}]({img})")
            a("")
            if s.get("focus_evidence"):
                a(f"> Focus evidence: {s['focus_evidence']}")
                a("")
            if s.get("veto"):
                a(f"> {s['veto']}")
                a("")
        a("")

    # ---- per-mechanic detail ---------------------------------------------
    a("## Detail")
    a("")
    for i, m in enumerate(r["mechanics"], 1):
        a(f"### {i}. {m['title']} - {_MARK[m['verdict']]}")
        a("")
        a(f"- **Deterministic check:** {m['log_note']}")
        a(f"- **AI diagnostics:** {m['ai_note']}")
        if m["steps"]:
            a(f"- **Steps exercising this:** "
              f"{', '.join(str(s) for s in m['steps'])}")
        a("")

        if m["verdict"] == "NOT TESTED":
            a("> **What this does not prove:** the mechanic was never "
              "exercised, so this report says nothing about whether it "
              "works. It is not evidence of a defect.")
            a("")

        for cand in m["candidates"]:
            before = _rel(cand.get("frame"), base)
            after = _rel(cand.get("frame_after"), base)
            if not before:
                continue
            a(f"**Step {cand['step']}** - `{', '.join(cand['actions'])}`, "
              f"delta {cand.get('delta')} ({cand.get('delta_verdict', 'n/a')})")
            a("")
            a("| before | after |")
            a("|---|---|")
            a(f"| ![step {cand['step']} before]({before}) | "
              + (f"![step {cand['step']} after]({after}) |" if after
                 else "_no after-frame saved_ |"))
            a("")
            if cand.get("reasoning"):
                a(f"> Agent reasoning at the time: {cand['reasoning']}")
                a("")

        for ev in m["evidence"]:
            a(f"- AI on step {ev['step']}: "
              f"**{'confirmed' if ev['happened'] else 'not visible'}** "
              f"(confidence {ev['confidence']:.2f}) - {ev['evidence']}")
        if m["evidence"]:
            a("")

    a("---")
    a("")
    a(f"Generated {r['generated_at']} from `{r['session_dir']}`.")
    a("")
    a("Every verdict above is derived from saved frames and the dispatched "
      "action log. A PASS means the mechanic was observed working at least "
      "once - it is not a guarantee of reliability across every attempt.")
    return "\n".join(out)


# ===========================================================================
# HTML
# ===========================================================================
_CSS = """
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
 background:#0f1115;color:#e6e6e6}
.wrap{max-width:1100px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 6px;line-height:1.3}
h2{font-size:19px;margin:34px 0 12px;border-bottom:1px solid #262b36;
 padding-bottom:6px}
h3{font-size:16px;margin:26px 0 10px}
.sub{color:#9aa4b2;margin:0 0 22px}
ol.summary{padding-left:24px;margin:0 0 20px}
ol.summary li{margin:7px 0;font-size:16px}
.tag{display:inline-block;padding:2px 9px;border-radius:11px;font-size:12px;
 font-weight:700;letter-spacing:.4px;margin-left:8px;vertical-align:1px}
.PASS{background:#123d20;color:#5ddc7f;border:1px solid #1e5c31}
.FAILED{background:#42151a;color:#ff8b8b;border:1px solid #6b2028}
.NOTTESTED{background:#2b2f3a;color:#a9b3c4;border:1px solid #3b4252}
.INCONCLUSIVE{background:#42361a;color:#f0c674;border:1px solid #6b5620}
.card{background:#161a22;border:1px solid #262b36;border-radius:9px;
 padding:16px 18px;margin:14px 0}
.kv{color:#9aa4b2;font-size:14px;margin:5px 0}
.kv b{color:#e6e6e6;font-weight:600}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
.pair figure{margin:0}
.pair img{width:100%;border:1px solid #262b36;border-radius:6px;display:block}
.pair figcaption{color:#7f8896;font-size:12px;margin-top:5px}
.quote{border-left:3px solid #3b4252;padding:7px 13px;margin:11px 0;
 color:#b9c1cd;background:#12151c;font-size:14px}
.warn{border-left:3px solid #6b5620;background:#1d1a12;padding:11px 14px;
 margin:12px 0;color:#f0c674}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid #262b36;padding:7px 10px;text-align:left}
th{background:#171b24;color:#9aa4b2}
footer{margin-top:44px;color:#7f8896;font-size:13px;border-top:1px solid
 #262b36;padding-top:16px}
"""


def html(r: dict[str, Any]) -> str:
    from html import escape as e

    base = Path(r["session_dir"])
    t, c = r["totals"], r["counts"]
    o: list[str] = []
    a = o.append

    a(f"<!doctype html><html><head><meta charset='utf-8'>"
      f"<title>Game Mechanics - {e(r['level'])}</title>"
      f"<style>{_CSS}</style></head><body><div class='wrap'>")
    a(f"<h1>Game Mechanics of {e(r['level'])} {e(r['chapter'].lower())} "
      f"({e(r['game_name'])})</h1>")
    a(f"<p class='sub'>Session <code>{e(r['session_id'])}</code> &middot; "
      f"{t['steps']} steps &middot; {t['deaths']} death(s) &middot; "
      f"{float(t['duration_seconds']):.0f}s")
    if r["launch"]["from_dashboard"]:
        a(f" &middot; launched from the dashboard into "
          f"<b>{e(r['launch']['target_level'])}</b> "
          f"({e(r['launch']['mode'])}-guided menus)")
    a("</p>")

    a("<ol class='summary'>")
    for m in r["mechanics"]:
        cls = _MARK[m["verdict"]].replace(" ", "")
        a(f"<li>{e(m['title'])}."
          f"<span class='tag {cls}'>{_MARK[m['verdict']]}</span></li>")
    a("</ol>")
    a(f"<p class='sub'><b>{c.get('PASS', 0)} passed, {c.get('FAIL', 0)} "
      f"failed, {c.get('NOT TESTED', 0)} not tested, "
      f"{c.get('INCONCLUSIVE', 0)} inconclusive.</b></p>")

    if not r["has_trace"]:
        a("<div class='warn'><b>No trace.json in this session.</b> It "
          "predates per-step trace recording, so the action log could not be "
          "read and the verdicts below rest on frames alone.</div>")

    a("<h2>Verdict meanings</h2><table><tr><th>verdict</th><th>means</th></tr>"
      "<tr><td>PASS</td><td>Dispatched and the evidence shows it worked.</td>"
      "</tr><tr><td>FAILED</td><td>Attempted, but the evidence shows it did "
      "not work.</td></tr><tr><td>NOT TESTED</td><td>Never attempted this "
      "session. Proves nothing either way, and is deliberately not reported "
      "as a failure.</td></tr><tr><td>INCONCLUSIVE</td><td>The action log "
      "and the AI audit disagree.</td></tr></table>")

    launch = r["launch"]
    lsteps = launch.get("steps") or []
    if lsteps:
        lok = launch.get("ok")
        a("<h2>Launch: dashboard to gameplay</h2>")
        a(f"<p class='sub'><b>{'REACHED GAMEPLAY' if lok else 'DID NOT COMPLETE'}"
          f"</b> in {launch.get('cycles', len(lsteps))} menu screens &middot; "
          f"target <b>{e(str(launch.get('target', launch.get('target_level', ''))))}"
          f"</b><br>{e(str(launch.get('reason', '')))}</p>")
        if not launch.get("target_confirmed", True):
            a("<div class='warn'>The target level was never explicitly "
              "confirmed on a level picker, so which level actually started "
              "is UNVERIFIED. Gameplay beginning is not by itself proof that "
              "the right level started.</div>")

        a("<table><tr><th>#</th><th>screen</th><th>focused</th>"
          "<th>input</th><th>note</th></tr>")
        for s in lsteps:
            rep = s.get("dispatched_repeat", 1) or 1
            act = s.get("dispatched", s.get("action", ""))
            act_txt = f"<code>{e(str(act))}</code>" + (f" &times;{rep}"
                                                       if rep > 1 else "")
            a(f"<tr><td>{s.get('cycle')}</td><td>{e(s.get('screen', ''))}</td>"
              f"<td>{e(s.get('focused_item') or '-')}</td><td>{act_txt}</td>"
              f"<td>{'vetoed confirm' if s.get('veto') else ''}</td></tr>")
        a("</table>")

        a("<div class='pair'>")
        for s in lsteps:
            img = _rel(s.get("frame"), base)
            if not img:
                continue
            cap = f"{s.get('cycle')}. {e(s.get('screen', ''))}"
            if s.get("focused_item"):
                cap += f" &middot; {e(s['focused_item'])}"
            a(f"<figure><img src='{e(img)}' alt='launch screen'>"
              f"<figcaption>{cap}</figcaption></figure>")
        a("</div>")

    a("<h2>Detail</h2>")
    for i, m in enumerate(r["mechanics"], 1):
        cls = _MARK[m["verdict"]].replace(" ", "")
        a(f"<div class='card'><h3>{i}. {e(m['title'])}"
          f"<span class='tag {cls}'>{_MARK[m['verdict']]}</span></h3>")
        a(f"<div class='kv'><b>Deterministic check:</b> "
          f"{e(m['log_note'])}</div>")
        a(f"<div class='kv'><b>AI diagnostics:</b> {e(m['ai_note'])}</div>")
        if m["steps"]:
            a(f"<div class='kv'><b>Steps:</b> "
              f"{', '.join(str(s) for s in m['steps'])}</div>")

        if m["verdict"] == "NOT TESTED":
            a("<div class='warn'><b>What this does not prove:</b> the "
              "mechanic was never exercised, so this report says nothing "
              "about whether it works. It is not evidence of a defect.</div>")

        for cand in m["candidates"]:
            before = _rel(cand.get("frame"), base)
            after = _rel(cand.get("frame_after"), base)
            if not before:
                continue
            a(f"<div class='kv'><b>Step {cand['step']}</b> &middot; "
              f"<code>{e(', '.join(cand['actions']))}</code> &middot; delta "
              f"{cand.get('delta')} ({e(str(cand.get('delta_verdict', '')))})"
              f"</div>")
            a("<div class='pair'>")
            a(f"<figure><img src='{e(before)}' alt='before'>"
              f"<figcaption>before</figcaption></figure>")
            if after:
                a(f"<figure><img src='{e(after)}' alt='after'>"
                  f"<figcaption>after</figcaption></figure>")
            a("</div>")
            if cand.get("reasoning"):
                a(f"<div class='quote'>Agent reasoning at the time: "
                  f"{e(cand['reasoning'])}</div>")

        for ev in m["evidence"]:
            verdict = "confirmed" if ev["happened"] else "not visible"
            a(f"<div class='kv'>AI on step {ev['step']}: <b>{verdict}</b> "
              f"(confidence {ev['confidence']:.2f}) - "
              f"{e(ev['evidence'])}</div>")
        a("</div>")

    a(f"<footer>Generated {e(r['generated_at'])} from "
      f"<code>{e(r['session_dir'])}</code>.<br>Every verdict is derived from "
      f"saved frames and the dispatched action log. A PASS means the "
      f"mechanic was observed working at least once; it is not a guarantee "
      f"of reliability across every attempt.</footer>")
    a("</div></body></html>")
    return "".join(o)


# ===========================================================================
# Writing
# ===========================================================================
def build_report(session_dir: Path | str, settings: Any = None,
                 llm_factory: Any = None, use_ai: bool = True,
                 max_pairs: int | None = None) -> dict[str, str]:
    """Analyse a session and write mechanics.{md,html,json}. Returns paths."""
    session_dir = Path(session_dir)
    if max_pairs is None:
        max_pairs = int(settings.get("gameplay.report.max_pairs", 2)
                        if settings else 2)

    payload = analyse(session_dir, settings=settings, llm_factory=llm_factory,
                      max_pairs=max_pairs, use_ai=use_ai)

    out_dir = session_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # Each format is written independently: losing the Markdown because the
    # HTML writer choked would be a poor trade.
    for fmt, name, render in (
        ("markdown", "mechanics.md", lambda: markdown(payload)),
        ("html", "mechanics.html", lambda: html(payload)),
        ("json", "mechanics.json",
         lambda: json.dumps(payload, indent=2, default=str)),
    ):
        try:
            path = out_dir / name
            path.write_text(render(), encoding="utf-8")
            written[fmt] = str(path)
        except Exception as exc:
            print(f"  [report] {fmt} failed: {exc}", flush=True)

    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build the game-mechanics report for a gameplay session.")
    ap.add_argument("session", type=Path,
                    help="a session dir under artifacts/gameplay/")
    ap.add_argument("--no-ai", action="store_true",
                    help="skip AI diagnostics; use the action log only")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="frame pairs to judge per mechanic (default 2)")
    args = ap.parse_args(argv)

    if not args.session.is_dir():
        print(f"ERROR: not a directory: {args.session}")
        return 2

    settings = None
    if not args.no_ai:
        try:
            from config import Config, load_dotenv_if_present
            load_dotenv_if_present(_ROOT / ".env")
            load_dotenv_if_present(_ROOT.parent / ".env")
            settings = Config.load(_ROOT / "config" / "settings.yaml",
                                   base=_ROOT)
        except Exception as exc:
            print(f"WARNING: settings could not be loaded ({exc}); "
                  f"continuing without AI diagnostics.")

    written = build_report(args.session, settings=settings,
                           use_ai=not args.no_ai, max_pairs=args.max_pairs)
    if not written:
        print("No report was written.")
        return 1
    for fmt, path in written.items():
        print(f"wrote {fmt:9s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
