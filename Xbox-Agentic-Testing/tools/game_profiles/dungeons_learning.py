"""Learning rules for Minecraft Dungeons gameplay (core/gameplay_lessons.py).

Every lesson comes from the saved per-cycle log, never from the model's own
claims, and every runtime change it makes is bounded and visible:

    death            -> drink earlier (threshold 70% -> 80/85%, drop 20 -> 12/10)
                        + a prompt hint about attackers off screen
    not_attempted    -> a chest/item/villager was seen close but never pressed:
                        the interaction agent gets priority over moving on
    failed           -> an agent acted but no later frame showed the result:
                        prompt hint only (no knob is proven to fix it)

Changes that fire at runtime are tagged in the move purpose ("[learned]") and
counted (learn_early_potions, learn_interact_overrides), so the report shows
the lesson was actually enforced, not only loaded.
"""
from __future__ import annotations

from typing import Any

GAME = "dungeons"
MIN_JUDGED_CYCLES = 10   # shorter runs neither prove a fix nor age a lesson

_INTERACT_SCENARIOS = {
    "Looting chests": ["loot"],
    "Interacting with items": ["loot", "door_or_lever"],
    "Interacting with villagers": ["villager"],
    "Saving villagers": ["villager"],
}


def _in_mission(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in payload.get("cycles") or [] if c.get("scene_state") == "in_mission"]


def _status(report: dict[str, Any], name: str) -> str:
    return next((s["status"] for s in report.get("scenarios") or []
                 if s.get("scenario") == name), "NOT ENCOUNTERED")


def _scenario(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next((s for s in report.get("scenarios") or [] if s.get("scenario") == name), {})


def death_cycles(payload: dict[str, Any]) -> list[int]:
    """First cycle of each downing (a revive countdown spans several cycles)."""
    out, prev = [], None
    for c in payload.get("cycles") or []:
        dead = c.get("scene_state") == "death"
        if dead and prev != "death":
            out.append(int(c["cycle"]))
        prev = c.get("scene_state")
    return out


def _death_finding(payload: dict[str, Any]) -> dict[str, Any] | None:
    deaths = death_cycles(payload)
    if not deaths:
        return None
    cycles = payload.get("cycles") or []
    first = deaths[0]
    before = [c for c in cycles if c["cycle"] < first
              and c.get("scene_state") == "in_mission"][-3:]
    hp = " -> ".join(f"{c.get('health_percent')}% (c{c['cycle']})" for c in before) or "unknown"
    unseen = [c["cycle"] for c in before if "unseen attacker" in str(c.get("route", ""))
              or c.get("blind_attacker")]
    drank = [c["cycle"] for c in before
             if any(m.get("macro") == "potion" for m in c.get("moves") or [])]
    cause = (f"Hero was downed at cycle {first}"
             + (f" ({len(deaths)} downings this run: {', '.join(f'c{d}' for d in deaths)})"
                if len(deaths) > 1 else "")
             + f". Health before: {hp}.")
    if unseen:
        cause += (f" Hit by an attacker nobody had located at "
                  f"{', '.join(f'c{n}' for n in unseen)}.")
    if not drank:
        cause += (" No potion was drunk: health never read at or below 70% and "
                  "never dropped 20% in one cycle before the hero went down.")
    return {"rule": "death", "scenario": "Survival",
            "signature": f"{GAME}|death",
            "primary_cause": cause,
            "evidence": {"death_cycles": deaths,
                         "cycles_before": [c["cycle"] for c in before],
                         "unseen_attacker_cycles": unseen, "potion_cycles": drank},
            "data": {"deaths": len(deaths), "unseen_attacker": bool(unseen)}}


def _not_attempted(payload: dict[str, Any], report: dict[str, Any],
                   name: str) -> dict[str, Any] | None:
    s = _scenario(report, name)
    if s.get("status") != "OBSERVED":
        return None
    by_n = {c["cycle"]: c for c in payload.get("cycles") or []}
    cats = _INTERACT_SCENARIOS[name]
    close = []
    for n in s.get("seen_cycles") or []:
        c = by_n.get(n) or {}
        if name == "Saving villagers" and not c.get("villager_in_danger"):
            # A calm villager is not a rescue (G006 in dungeons-20261016-133419
            # was a villager the hero had just talked to).
            continue
        near = [e for e in c.get("entities") or [] if e.get("category") in cats
                and e.get("distance_estimate") in ("very_close", "close")]
        if near:
            close.append((n, near[0].get("kind"), near[0].get("distance_estimate"),
                          c.get("agent"), c.get("route")))
    if not close:
        return None   # only seen far away: walking past it is not a mistake
    shown = "; ".join(f"c{n} {kind} {dist}, {agent} drove ({route})"
                      for n, kind, dist, agent, route in close[:4])
    return {"rule": "not_attempted", "scenario": name,
            "signature": f"{GAME}|not_attempted|{name}".lower(),
            "primary_cause": (f"'{name}': the target was close {len(close)} time(s) but "
                              f"no agent pressed A - another agent kept control. {shown}."),
            "evidence": {"close_cycles": [n for n, *_ in close]},
            "data": {"categories": cats}}


def _failed(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    s = _scenario(report, name)
    if s.get("status") != "FAIL":
        return None
    acted = s.get("attempted_cycles") or []
    return {"rule": "failed", "scenario": name,
            "signature": f"{GAME}|failed|{name}".lower(),
            "primary_cause": (f"'{name}': agents acted at "
                              f"{', '.join(f'c{n}' for n in acted[:8])} but no later frame "
                              f"showed the result ({'; '.join(s.get('notes') or [])})."),
            "evidence": {"attempted_cycles": acted},
            "data": {}}


class DungeonsRules:
    game = GAME

    def detect(self, payload: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
        found: list[dict[str, Any] | None] = [_death_finding(payload)]
        found += [_not_attempted(payload, report, n) for n in _INTERACT_SCENARIOS]
        found += [_failed(report, s["scenario"]) for s in report.get("scenarios") or []]
        return [f for f in found if f]

    def assess(self, lesson: dict[str, Any], payload: dict[str, Any],
               report: dict[str, Any]) -> str:
        """Judge an open lesson that did NOT recur this run."""
        if len(_in_mission(payload)) < MIN_JUDGED_CYCLES:
            return "untested"
        rule, name = lesson.get("failure_class"), str(lesson.get("scenario") or "")
        if rule == "death":
            # Surviving a run with no fighting proves nothing.
            return "quiet" if _status(report, "Fighting mobs") != "NOT ENCOUNTERED" \
                else "untested"
        status = _status(report, name)
        if status == "PASS":
            lesson["_fix_note"] = f"'{name}' passed with a result shown on a later frame"
            return "fixed"
        if rule == "not_attempted" and status == "FAIL":
            lesson["_fix_note"] = f"'{name}' is now attempted (its result is tracked separately)"
            return "fixed"
        return "untested"

    def adjustments(self, lessons: list[dict[str, Any]]) -> dict[str, Any]:
        counters: dict[str, Any] = {}
        applied: list[dict[str, Any]] = []
        for l in lessons:
            rule = l.get("failure_class")
            if rule == "death":
                # Strong only after 2+ deaths in the latest run that recurred;
                # 85% / -10% drank at 90% and broke off a sustain for a 10%
                # chip in dungeons-20261016-133419 (a deathless run).
                strong = int((l.get("data") or {}).get("deaths", 1)) >= 2 \
                    and int(l.get("quiet_runs", 0)) == 0
                counters["_learn_potion_threshold"] = 85 if strong else 80
                counters["_learn_potion_drop"] = 15 if strong else 18
                applied.append({"id": l.get("id"), "rule": rule,
                                "change": (f"drink at <= {counters['_learn_potion_threshold']}% "
                                           f"health or a {counters['_learn_potion_drop']}% drop "
                                           "(default 70% / 20%)")})
            elif rule == "not_attempted":
                cats = list((l.get("data") or {}).get("categories") or [])
                merged = sorted(set(counters.get("_learn_interact_priority", [])) | set(cats))
                counters["_learn_interact_priority"] = merged
                applied.append({"id": l.get("id"), "rule": rule,
                                "change": (f"interact with close {'/'.join(cats)} before "
                                           "the navigator moves on")})
            else:
                applied.append({"id": l.get("id"), "rule": rule, "change": "prompt hint only"})
        return {"counters": counters, "applied": applied}

    def prompt_text(self, lessons: list[dict[str, Any]]) -> str:
        if not lessons:
            return ""
        lines = ["LEARNED FROM EARLIER RUNS (verified from saved frames):"]
        for l in lessons[:5]:
            rule = l.get("failure_class")
            if rule == "death":
                tip = ("watch the screen edges for ranged mobs; if health drops while "
                       "nobody is visible, say so in reasoning and move away")
            elif rule == "not_attempted":
                what = "/".join((l.get("data") or {}).get("categories") or ["target"])
                tip = (f"when a {what.replace('_', ' ')} is close, command the "
                       "interaction agent before moving on")
            else:
                tip = "check the next frame actually shows the result of the action"
            lines.append(f"- [{l.get('id')}] {str(l.get('primary_cause', ''))[:160]} -> {tip}")
        return "\n".join(lines) + "\n"


RULES = DungeonsRules()
