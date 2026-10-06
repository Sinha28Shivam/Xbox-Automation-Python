"""Scenario report for a Minecraft Dungeons gameplay run.

Built only from the saved per-cycle log (dungeons-cycles.json): every claim
links to the cycle numbers and frames it came from. Statuses: NOT ENCOUNTERED
(never on screen, excluded from the verdict), OBSERVED (seen, no agent acted),
FAIL (acted, no result on a later frame), PASS (result shown on a later frame).

usage: python tools/dungeons_scenario_report.py [path/to/dungeons-cycles.json]
"""
from __future__ import annotations

import difflib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

_RESCUE_WORDS = ("rescue", "free", "save", "villager")
_VILLAGE_WORDS = ("village", "farm", "haven", "town", "market")


def _moves(c: dict[str, Any]) -> list[dict[str, Any]]:
    return c.get("moves") or []


def _did(c: dict[str, Any], *macros: str) -> bool:
    return any(m.get("macro") in macros for m in _moves(c))


def _ents(c: dict[str, Any], cat: str, near: bool = False) -> list[dict[str, Any]]:
    return [e for e in c.get("entities") or [] if e.get("category") == cat
            and (not near or e.get("distance_estimate") in ("very_close", "close"))]


def _pressed_at(c: dict[str, Any], cat: str, word: str = "") -> bool:
    """A was pressed AT this kind of target (from the interaction agent's purpose)."""
    for m in _moves(c):
        p = str(m.get("purpose", "")).lower()
        if m.get("macro") == "interact" and p.startswith(f"interact with {cat}") and word in p:
            return True
    return False


def _text(c: dict[str, Any]) -> str:
    return f"{c.get('objective', '')} {c.get('loot_or_door', '')}".lower()


PASS, FAIL, OBSERVED, NOT_ENCOUNTERED = "PASS", "FAIL", "OBSERVED", "NOT ENCOUNTERED"
# Objective texts this similar are the same stage (OCR-ish misreads like
# "ALLBGERS" vs "ILLAGERS").
_SAME_OBJECTIVE = 0.75


def _status(seen: list[int], attempted: list[int], evidence: list[int]) -> str:
    if evidence:
        return PASS
    if attempted:
        return FAIL
    return OBSERVED if seen else NOT_ENCOUNTERED


def _norm_objective(text: str) -> str:
    t = re.sub(r"\(.*?\)|objective marker.*|marker .*", " ", text.lower())
    t = re.sub(r"the [a-z]+ from the rift|[^a-z ]", " ", t)
    return " ".join(t.split())


def objective_stages(cycles: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """(first cycle, objective) for each distinct objective stage, in order."""
    stages: list[tuple[int, str]] = []
    for c in cycles:
        o = _norm_objective(str(c.get("objective") or ""))
        if len(o) < 6:
            continue
        if any(difflib.SequenceMatcher(None, o, s).ratio() >= _SAME_OBJECTIVE
               for _, s in stages[-1:]):
            continue
        if stages:
            # Ignore a one-cycle flicker: require the next reading to agree.
            idx = cycles.index(c)
            nxt = [_norm_objective(str(x.get("objective") or "")) for x in cycles[idx + 1:idx + 3]]
            if nxt and not any(difflib.SequenceMatcher(None, o, n).ratio() >= _SAME_OBJECTIVE
                               for n in nxt if len(n) >= 6):
                continue
        stages.append((c["cycle"], o))
    return stages


def verdict(scenarios: list[dict[str, Any]]) -> str:
    met = [s["status"] for s in scenarios if s["status"] != NOT_ENCOUNTERED]
    if not met:
        return "NO SCENARIOS ENCOUNTERED"
    if FAIL in met:
        return FAIL
    return PASS if all(s == PASS for s in met) else "INCOMPLETE"


def _scenario(name: str, seen: list[int], attempted: list[int],
              evidence: list[int], notes: list[str]) -> dict[str, Any]:
    return {"scenario": name, "status": _status(seen, attempted, evidence),
            "seen_cycles": seen, "attempted_cycles": attempted,
            "evidence_cycles": evidence, "notes": notes}


def build(payload: dict[str, Any]) -> dict[str, Any]:
    cycles = [c for c in payload.get("cycles", []) if c.get("scene_state") == "in_mission"]
    by_n = {c["cycle"]: c for c in cycles}
    nxt = {a["cycle"]: b for a, b in zip(cycles, cycles[1:])}

    # Saving villagers: villager in danger -> rescue moves -> danger cleared.
    seen = [c["cycle"] for c in cycles if c.get("villager_in_danger")
            or c.get("objective_kind") == "rescue"
            or any(w in _text(c) for w in _RESCUE_WORDS[:3])]
    tried = [c["cycle"] for c in cycles
             if any(str(m.get("purpose", "")).startswith("Rescue") for m in _moves(c))]
    done = [n for n in tried if n in nxt and not nxt[n].get("villager_in_danger")
            and by_n[n].get("villager_in_danger")]
    rescue = _scenario("Saving villagers", seen, tried, done,
                       ["evidence = villager no longer in danger on the next frame"])

    # Fighting mobs: combat moves, then fewer enemies near the hero.
    seen = [c["cycle"] for c in cycles if (c.get("enemy_count") or 0) > 0
            or _ents(c, "enemy")]
    tried = [c["cycle"] for c in cycles if _did(c, "fight", "melee", "ranged")]
    done = [n for n in tried if n in nxt
            and (nxt[n].get("enemy_count") or 0) < (by_n[n].get("enemy_count") or 0)]
    fight = _scenario("Fighting mobs", seen, tried, done,
                      ["evidence = nearby enemy count fell after the attack"])

    # Exploring villages: village structures/objective seen while the map moved.
    # Objective text alone ("Find Honeycomb Farm") is not a sighting.
    seen = [c["cycle"] for c in cycles if _ents(c, "building")
            or any(w in str(c.get("loot_or_door", "")).lower() for w in _VILLAGE_WORDS)]
    moved = [n for n in seen if (by_n[n].get("map_shift") or 0) >= 8]
    explore = _scenario("Exploring villages", seen, moved, moved,
                        ["evidence = hero moved on the minimap while a village was in view"])

    # Interacting with items: A pressed next to a gate/lever/loot/prompt.
    def near_item(c: dict[str, Any]) -> bool:
        return bool(c.get("interact_prompt_visible") or _ents(c, "door_or_lever", True)
                    or _ents(c, "loot", True) or _ents(c, "portal", True))
    # Next logged cycle of any kind: a teleport's result is a loading screen.
    all_c = payload.get("cycles", [])
    nxt_any = {a["cycle"]: b for a, b in zip(all_c, all_c[1:])}
    seen = [c["cycle"] for c in cycles if near_item(c)]
    tried = [c["cycle"] for c in cycles if _did(c, "interact") and near_item(c)
             and not _pressed_at(c, "villager")]
    done = [n for n in tried if n in nxt_any and (
        by_n[n].get("interact_prompt_visible") and not nxt_any[n].get("interact_prompt_visible")
        or nxt_any[n].get("scene_state") == "loading_or_cutscene"
        or (nxt_any[n].get("delta") or 0) >= 10)]
    items = _scenario("Interacting with items", seen, tried, done,
                      ["PASS = after A, the prompt went away, a loading screen "
                       "followed (teleport), or the screen changed a lot"])

    # Looting chests: interact with a chest close by, chest gone/opened after.
    def chest(c: dict[str, Any], near: bool = False) -> list[dict[str, Any]]:
        return [e for e in _ents(c, "loot", near) if "chest" in str(e.get("kind", "")).lower()]
    seen = [c["cycle"] for c in cycles if chest(c) or "chest" in _text(c)]
    tried = [c["cycle"] for c in cycles if _pressed_at(c, "loot", "chest")]
    done = [n for n in tried if n in nxt and (
        not chest(nxt[n], near=True)
        or any("open" in str(e.get("kind", "")).lower() for e in chest(nxt[n])))]
    loot = _scenario("Looting chests", seen, tried, done,
                     ["PASS = A pressed at a chest, then the chest is gone or "
                      "shown open on the next frame"])

    # Interacting with villagers (not a rescue): A aimed at a calm villager.
    seen = [c["cycle"] for c in cycles if _ents(c, "villager")]
    tried = [c["cycle"] for c in cycles if _pressed_at(c, "villager")
             and not c.get("villager_in_danger")]
    done = [n for n in tried if by_n[n].get("interact_prompt_visible")
            or (n in nxt and nxt[n].get("interact_prompt_visible"))]
    talk = _scenario("Interacting with villagers", seen, tried, done,
                     ["PASS = A pressed at a villager while a prompt/speech "
                      "bubble showed (that frame or the next)"])

    # Navigation: walking the minimap bearing, and freeing the hero from walls.
    nav_c = [c["cycle"] for c in cycles if c.get("agent") == "navigator"
             or any(str(m.get("purpose", "")).startswith(("Minimap", "Navigate"))
                    for m in _moves(c))]
    walked = [n for n in nav_c if n in nxt and (nxt[n].get("map_shift") or 0) >= 8]
    nav = _scenario("Following the minimap", nav_c, nav_c, walked,
                    [f"map travel {payload.get('map_travel_px', 0)} px, "
                     f"heading corrections {payload.get('map_steers', 0)}",
                     "PASS = minimap scrolled after a navigator move"])
    stuck = [c["cycle"] for c in cycles if c.get("hero_occluded")]
    esc = [c["cycle"] for c in cycles if any(
        str(m.get("purpose", "")).startswith(("Blocked", "Detour")) for m in _moves(c))]
    freed = [n for n in esc if n in nxt and (nxt[n].get("map_shift") or 0) >= 8]
    wall = _scenario("Escaping walls", stuck + [n for n in esc if n not in stuck], esc, freed,
                     [f"wall escapes {payload.get('wall_escapes', 0)}",
                      "PASS = minimap moved again after an escape"])

    # Objective reached: the objective banner moves to a new stage.
    stages = objective_stages(cycles)
    obj = _scenario("Objective reached", [stages[0][0]] if stages else [],
                    [n for n, _ in stages[:1]], [n for n, _ in stages[1:]],
                    ["stages: " + " -> ".join(f"{s} (c{n})" for n, s in stages)
                     if stages else "no objective text read",
                     "PASS = objective changed to a new stage"])
    if not stages[1:] and stages:
        obj["status"] = OBSERVED

    survival = {"deaths": payload.get("deaths", 0), "potions": payload.get("potions", 0),
                "potions_confirmed": payload.get("potions_confirmed", 0),
                "min_health": payload.get("min_health")}
    agents: dict[str, list[int]] = {}
    messages = []
    for c in payload.get("cycles", []):
        agents.setdefault(c.get("agent") or "model", []).append(c["cycle"])
        cmd = c.get("command") or {}
        messages.append({"cycle": c["cycle"], "command_to": cmd.get("to"),
                         "target": cmd.get("target"), "direction": cmd.get("direction"),
                         "reason": cmd.get("reason"), "agent": c.get("agent"),
                         "route": c.get("route"),
                         "buttons": ", ".join(str(m.get("macro")) + (f"({m['direction']})" if m.get("direction") else "")
                                              for m in _moves(c))})
    scenarios = [rescue, fight, explore, items, loot, talk, nav, wall, obj]
    return {"verdict": verdict(scenarios),
            "agents": agents,
            "messages": messages,
            "run": {"cycles_run": payload.get("cycles_run"),
                    "duration_seconds": payload.get("duration_seconds"),
                    "stop_reason": payload.get("stop_reason"),
                    "mission_complete": payload.get("mission_complete", False),
                    "objective_reached": obj["status"] == PASS},
            "survival": survival,
            "scenarios": scenarios,
            "frames": {c["cycle"]: c.get("frame_before") for c in payload.get("cycles", [])}}


def _html(report: dict[str, Any], reports_dir: Path) -> str:
    colors = {PASS: "#2e7d32", FAIL: "#c62828", OBSERVED: "#1565c0",
              NOT_ENCOUNTERED: "#757575", "INCOMPLETE": "#f9a825",
              "NO SCENARIOS ENCOUNTERED": "#757575"}

    def link(n: int) -> str:
        p = report["frames"].get(n)
        if not p:
            return str(n)
        try:
            rel = Path(p).resolve().relative_to(reports_dir.parent.resolve())
            href = "../" + rel.as_posix()
        except ValueError:
            href = Path(p).as_uri()
        return f'<a href="{html.escape(href)}">{n}</a>'

    def cyc(ns: list[int]) -> str:
        return ", ".join(link(n) for n in ns[:25]) + (" ..." if len(ns) > 25 else "") or "-"

    rows = "".join(
        f"<tr><td>{html.escape(s['scenario'])}</td>"
        f"<td style='color:#fff;background:{colors[s['status']]}'>{s['status']}</td>"
        f"<td>{cyc(s['seen_cycles'])}</td><td>{cyc(s['attempted_cycles'])}</td>"
        f"<td>{cyc(s['evidence_cycles'])}</td>"
        f"<td>{html.escape('; '.join(s['notes']))}</td></tr>"
        for s in report["scenarios"])
    run, sv = report["run"], report["survival"]
    arows = "".join(f"<tr><td>{html.escape(a)}</td><td>{len(ns)}</td><td>{cyc(ns)}</td></tr>"
                    for a, ns in report.get("agents", {}).items())
    mrows = "".join(
        f"<tr><td>{link(m['cycle'])}</td>"
        f"<td>{html.escape(str(m['command_to'] or '-'))}: {html.escape(str(m['target'] or ''))} "
        f"@{html.escape(str(m['direction'] or ''))}</td>"
        f"<td>{html.escape(str(m['reason'] or ''))}</td>"
        f"<td>{html.escape(str(m['agent'] or '-'))}</td>"
        f"<td>{html.escape(str(m['route'] or ''))}</td>"
        f"<td>{html.escape(m['buttons'])}</td></tr>"
        for m in report.get("messages", []))
    counts = {k: sum(s["status"] == k for s in report["scenarios"])
              for k in (PASS, FAIL, OBSERVED, NOT_ENCOUNTERED)}
    v = report["verdict"]
    return (
        "<html><head><meta charset='utf-8'><title>Dungeons scenario report</title>"
        "<style>body{font-family:sans-serif}td,th{border:1px solid #ccc;padding:4px;"
        "vertical-align:top}table{border-collapse:collapse}</style></head><body>"
        "<h2>Minecraft Dungeons - scenario report</h2>"
        f"<h3>Overall: <span style='color:#fff;background:{colors.get(v, '#757575')};"
        f"padding:2px 8px'>{html.escape(v)}</span></h3>"
        f"<p>{counts[PASS]} PASS, {counts[FAIL]} FAIL, {counts[OBSERVED]} OBSERVED, "
        f"{counts[NOT_ENCOUNTERED]} NOT ENCOUNTERED (excluded). "
        f"Objective reached: {run.get('objective_reached')}. "
        f"Mission complete: {run['mission_complete']}.</p>"
        f"<p>Cycles {run['cycles_run']} in {run['duration_seconds']}s. "
        f"Stop: {html.escape(str(run['stop_reason']))}.</p>"
        f"<p>Deaths {sv['deaths']}, potions {sv['potions']} "
        f"(confirmed {sv['potions_confirmed']}), lowest health {sv['min_health']}%.</p>"
        "<table><tr><th>Scenario</th><th>Status</th><th>Seen (cycles)</th>"
        f"<th>Acted</th><th>Result shown</th><th>Rule</th></tr>{rows}</table>"
        "<h3>Which agent drove each cycle</h3><table><tr><th>Agent</th>"
        f"<th>Cycles</th><th>Cycle numbers</th></tr>{arows}</table>"
        "<h3>Agent messages (Observer command -> acting agent)</h3>"
        "<table><tr><th>Cycle</th><th>Observer command</th><th>Reason</th>"
        f"<th>Acted</th><th>Route</th><th>Buttons</th></tr>{mrows}</table>"
        "<p>Cycle numbers link to the frame the agent saw. FAIL means the agent "
        "acted but no later frame showed the result; only PASS is backed by a "
        "change on a later frame. NOT ENCOUNTERED scenarios never appeared on "
        "screen and do not affect the overall result.</p>"
        "</body></html>")


def write(payload: dict[str, Any], reports_dir: Path) -> dict[str, Any]:
    report = build(payload)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "dungeons-scenarios.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    (reports_dir / "dungeons-scenarios.html").write_text(
        _html(report, reports_dir), encoding="utf-8")
    return report


def print_summary(report: dict[str, Any]) -> None:
    print(f"\nSCENARIO REPORT - overall {report['verdict']}", flush=True)
    for s in report["scenarios"]:
        print(f"  {s['scenario']:30s} {s['status']:16s} seen={len(s['seen_cycles'])} "
              f"acted={len(s['attempted_cycles'])} shown={len(s['evidence_cycles'])}",
              flush=True)
    print(f"  Objective reached: {report['run']['objective_reached']}, "
          f"mission complete: {report['run']['mission_complete']}", flush=True)
    print(f"  Survival: {report['survival']}", flush=True)
    print("  Agents  : " + ", ".join(f"{a}={len(ns)}" for a, ns in report.get("agents", {}).items()),
          flush=True)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent
        / "artifacts/runs/dungeons-fast/reports/dungeons-cycles.json")
    rep = write(json.loads(src.read_text(encoding="utf-8")), src.parent)
    print_summary(rep)
    print(f"  Written: {src.parent / 'dungeons-scenarios.html'}")
