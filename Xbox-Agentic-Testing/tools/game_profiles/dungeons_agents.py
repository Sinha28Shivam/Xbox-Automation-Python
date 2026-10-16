"""Multi-agent decision layer for Minecraft Dungeons.

Observer (the one vision-model call + pixel sensors) reads the screen and
sends a command {to, target, direction, reason}. The supervisor delivers it
to that action agent, which builds the button presses. Safety overrides win
over the command: low health -> health agent, enemy in reach -> combat.
If the commanded agent refuses: interaction -> combat -> navigator.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from . import dungeons_profile as dp
from .dungeons_profile import (AgentCommand, DungeonsEntity, DungeonsFrameDecision,
                               DungeonsMove)

_MAX_INTERACT_TRIES = 3      # A presses on one target before giving up
_MAX_APPROACH = 4            # consecutive approach cycles before giving up
_MAX_SIDESTEPS = 2           # stalled-approach sidesteps before giving up
_INTERACT_COOLDOWN = 4       # cycles the interaction agent stands down after giving up


class Agent:
    name = "agent"

    def propose(self, d: DungeonsFrameDecision, c: dict[str, Any],
                cmd: AgentCommand | None = None) -> list[DungeonsMove] | None:
        raise NotImplementedError


class HealthAgent(Agent):
    """Watches the heart gauge and drinks a potion (LT)."""
    name = "health"

    def propose(self, d, c, cmd=None):
        return dp._potion_check(d, c)


class CombatAgent(Agent):
    """Senses enemies and attacks at once. urgent=True only acts on an enemy
    in reach or one attacking a villager; urgent=False handles ranged."""

    def __init__(self, urgent: bool) -> None:
        self.urgent = urgent
        self.name = "combat"

    def propose(self, d, c, cmd=None):
        threat = dp._villager_threat_dir(d)
        if self.urgent:
            if threat:
                c["rescue_attacks"] += 1
                return [DungeonsMove(action="fight", direction=threat,
                                     purpose="Rescue: kill the enemy on the villager.")]
            if d.enemy_adjacent and d.enemy_direction != "none":
                return dp._combat_moves(d, c)
            return None
        # Only fight what perception confirmed (dp._combat_state). The old
        # fallback to the Observer's command direction fired crossbow volleys
        # at empty ground whenever the model said "combat" without an enemy.
        if d.enemy_direction != "none":
            return dp._combat_moves(d, c)
        return None


_COIN_WORDS = ("coin", "emerald", "gold", "gem")


def _is_coins(e: DungeonsEntity) -> bool:
    kind = e.kind.lower()
    return e.category == "loot" and "chest" not in kind and any(w in kind for w in _COIN_WORDS)


class InteractionAgent(Agent):
    """Villagers, chests, pickups, gates and levers: walk up and press A."""
    name = "interaction"

    _CATS = ("villager", "loot", "door_or_lever")

    @classmethod
    def _target(cls, d: DungeonsFrameDecision,
                cmd: AgentCommand | None) -> DungeonsEntity | None:
        def near(cat: str) -> list[DungeonsEntity]:
            es = [e for e in dp._entities(d, cat)
                  if e.distance_estimate in ("very_close", "close")]
            return sorted(es, key=lambda e: e.distance_estimate != "very_close")
        cands = [e for cat in cls._CATS for e in near(cat)]
        if cmd and cmd.target:
            # The Observer named the target: prefer the matching entity.
            want = cmd.target.lower()
            named = [e for e in cands + dp._entities(d, "portal")
                     if e.kind.lower() in want or want in e.kind.lower()
                     or e.category in want]
            if named:
                return named[0]
        ordered = (near("villager") if d.villager_in_danger else []) + \
            near("loot") + near("door_or_lever") + near("villager")
        return ordered[0] if ordered else None

    def _give_up(self, c: dict[str, Any]) -> None:
        c["_interact_run"] = c["_approach_run"] = c["_sidesteps"] = 0
        c["_interact_cooldown"] = _INTERACT_COOLDOWN
        c["interaction_give_ups"] += 1

    def propose(self, d, c, cmd=None):
        if d.enemy_adjacent:
            return None
        prompt = d.interact_prompt_visible
        if c["_interact_cooldown"] > 0 and not prompt:
            c["_interact_cooldown"] -= 1
            return None
        t = self._target(d, cmd)
        if t is not None and t.category == "portal" and not (cmd and cmd.target):
            t = None
        rescue = bool(t and t.category == "villager" and d.villager_in_danger)
        if t is not None and _is_coins(t) and not prompt:
            # Spilled coins/emeralds are collected by walking over them.
            c["_interact_run"] = 0
            c["coin_pickups"] = c.get("coin_pickups", 0) + 1
            return [DungeonsMove(action="move", direction=t.direction,
                                 duration=0.6 if t.distance_estimate == "very_close" else 1.2,
                                 purpose=f"Collect loot '{t.kind}' by walking over it.")]
        if prompt or (t and t.distance_estimate == "very_close"):
            c["_approach_run"] = c["_sidesteps"] = 0
            c["_interact_run"] += 1
            if c["_interact_run"] > _MAX_INTERACT_TRIES:
                self._give_up(c)
                return None
            if rescue:
                c["rescue_interacts"] += 1
            what = f"{t.category} '{t.kind}'" if t else "on-screen prompt"
            c["_interact_target"] = t.category if t else "prompt"
            return [DungeonsMove(action="interact", direction=t.direction if t else "none",
                                 purpose=f"Interact with {what}.")]
        c["_interact_run"] = 0
        heading = t.direction if t else (cmd.direction if cmd else "none")
        if heading == "none":
            c["_approach_run"] = 0
            return None
        label = f"{t.category} '{t.kind}'" if t else (cmd.target or "target")
        stalled = (c["_approach_run"] > 0 and (d._nav.get("shift") or 0) < dp._MAP_STILL_PX)
        c["_approach_run"] += 1
        if stalled:
            if c["_sidesteps"] >= _MAX_SIDESTEPS:
                self._give_up(c)
                return None
            side = 45 if c["_sidesteps"] % 2 == 0 else -45
            c["_sidesteps"] += 1
            heading = dp._rotate(heading, side)
            purpose = f"Approach {label}: stalled, sidestep {heading}."
        elif c["_approach_run"] > _MAX_APPROACH:
            self._give_up(c)
            return None
        else:
            purpose = f"Approach {label}."
        c["rescue_approaches" if rescue else "interaction_approaches"] += 1
        return [DungeonsMove(action="move", direction=heading, duration=1.2,
                             purpose=purpose)]


class NavigatorAgent(Agent):
    """Minimap following and wall escape. Always returns a move."""
    name = "navigator"

    def propose(self, d, c, cmd=None):
        moves = dp._nav_moves(d, c)
        if moves:
            return moves
        bearing = d._nav.get("bearing")
        hint = cmd.direction if cmd and cmd.to == "navigator" else "none"
        # Map wins when the Observer's hint points well away from the marker.
        if bearing and hint != "none" and dp._angle_diff(hint, bearing) > 90:
            hint = "none"
        heading = hint if hint != "none" else (bearing or d.objective_bearing)
        if not heading or heading == "none":
            heading = d.progress_direction if d.progress_direction != "none" else "up"
        src = "Observer hint" if heading == hint else "minimap"
        return [DungeonsMove(action="move", direction=heading, duration=2.5,
                             purpose=f"Navigate {heading} ({src}).")]


class Supervisor:
    def __init__(self) -> None:
        self.health = HealthAgent()
        self.urgent = CombatAgent(urgent=True)
        self.by_name: dict[str, Agent] = {
            "health": self.health, "combat": CombatAgent(urgent=False),
            "interaction": InteractionAgent(), "navigator": NavigatorAgent()}
        self.fallback = ("interaction", "combat", "navigator")

    @staticmethod
    def _credit(d: DungeonsFrameDecision, c: dict[str, Any], name: str,
                route: str) -> None:
        d._agent, d._route = name, route
        c[f"agent_{name}"] = c.get(f"agent_{name}", 0) + 1
        print(f"  Agent     : {name} ({route})", flush=True)

    def _learned_interaction(self, d: DungeonsFrameDecision, c: dict[str, Any],
                             cmd: AgentCommand) -> list[DungeonsMove] | None:
        """A not_attempted lesson: a close target of a learned category goes to
        the interaction agent before any other agent can walk past it.

        Runs after the health and unseen-attacker overrides. A HELD engagement
        (no enemy on this frame, only a remembered one) does not block it;
        a real hit or a visible enemy does."""
        cats = set(c.get("_learn_interact_priority") or [])
        if not cats or cmd.to == "interaction" or d.enemy_adjacent \
                or d._nav.get("blind_attacker"):
            return None
        if d.enemy_direction != "none" and not d._nav.get("held"):
            return None
        near = [e for e in d.entities if e.category in cats
                and e.distance_estimate in ("very_close", "close")]
        if not near and not (d.interact_prompt_visible and cats & {"loot", "door_or_lever"}):
            return None
        target = near[0] if near else None
        steer = AgentCommand(to="interaction", target=target.kind if target else "",
                             direction=target.direction if target else "none",
                             reason="learned: close target was skipped in an earlier run")
        moves = self.by_name["interaction"].propose(d, c, steer)
        if not moves:
            return None
        for m in moves:
            m.purpose = f"{m.purpose} [learned]"
        c["learn_interact_overrides"] = c.get("learn_interact_overrides", 0) + 1
        self._credit(d, c, "interaction", "learned: interact before moving on")
        return moves

    def forced_moves(self, d: DungeonsFrameDecision,
                     c: dict[str, Any]) -> list[DungeonsMove] | None:
        if d.scene_state == "mission_complete" and d._nav.get("hud"):
            d.scene_state = "in_mission"  # completion claim was rejected
        if d.scene_state != "in_mission":
            self._credit(d, c, "scene", d.scene_state)
            return dp._forced_moves(d, c)
        c["_downed"] = False
        dp._update_nav(d, c)
        dp._perceive(d, c)
        if dp._entities(d, "building"):
            c["village_sightings"] += 1
        cmd = d.command
        print(f"  Observer  : -> {cmd.to}: {cmd.target or '-'} @{cmd.direction} "
              f"- {cmd.reason}", flush=True)
        moves = self.health.propose(d, c, cmd)
        if moves:
            self._credit(d, c, "health", f"override: health {d.health_percent}%")
            return moves
        moves = dp._blind_defence(d, c)
        if moves:
            self._credit(d, c, "combat", f"override: hit by unseen attacker "
                                         f"(-{d._nav.get('damage')}%)")
            return moves
        moves = self._learned_interaction(d, c, cmd)
        if moves:
            return moves
        if d._nav.get("blind_attacker") or d._nav.get("held"):
            moves = self.by_name["combat"].propose(d, c, None)
            if moves:
                why = ("hit, attacking last-seen enemy" if d._nav.get("blind_attacker")
                       else "holding engagement")
                self._credit(d, c, "combat", f"override: {why}")
                return moves
        moves = self.urgent.propose(d, c, cmd)
        if moves:
            self._credit(d, c, "combat", "override: enemy in reach")
            return moves
        to = cmd.to
        if to == "combat" and d.enemy_direction == "none":
            # The Observer asked for a fight that perception did not confirm.
            c["combat_commands_dropped"] = c.get("combat_commands_dropped", 0) + 1
            to = "navigator"
        elif to == "navigator" and d.enemy_direction != "none":
            # A confirmed enemy outranks walking past it.
            to = "combat"
        order = [to] + [n for n in self.fallback if n != to]
        for i, name in enumerate(order):
            if name == "health":
                continue
            moves = self.by_name[name].propose(d, c, cmd if i == 0 else None)
            if moves:
                route = ("command" if to == cmd.to else f"redirected from {cmd.to}") \
                    if i == 0 else f"fallback from {to}"
                self._credit(d, c, name, route)
                return moves
        return None  # unreachable: navigator always moves


class ReporterAgent:
    """Writes the scenario report at the end of every run."""
    name = "reporter"

    def run(self, result: dict[str, Any], reports_dir: Path) -> dict[str, Any]:
        import tools.dungeons_scenario_report as rep
        report = rep.write(result, reports_dir)
        rep.print_summary(report)
        print(f"  Report: {reports_dir / 'dungeons-scenarios.html'}", flush=True)
        return report


_AGENT_NAMES = ("scene", "health", "combat", "interaction", "navigator")


def _initial_counters() -> dict[str, Any]:
    c = dp._initial_counters()
    c.update({f"agent_{n}": 0 for n in _AGENT_NAMES})
    c.update({"_interact_run": 0, "_approach_run": 0, "_interact_cooldown": 0,
              "_sidesteps": 0, "interaction_approaches": 0,
              "interaction_give_ups": 0, "village_sightings": 0,
              "combat_commands_dropped": 0})
    return c


def _summarize(c: dict[str, Any]) -> dict[str, Any]:
    out = dp._summarize(c)
    out.update({k: c.get(k, 0) for k in ("interaction_approaches", "interaction_give_ups",
                                         "village_sightings", "combat_commands_dropped")})
    out["agent_cycles"] = {n: c.get(f"agent_{n}", 0) for n in _AGENT_NAMES}
    return out


def _cycle_extra_fields(d: DungeonsFrameDecision) -> dict[str, Any]:
    return {**dp._cycle_extra_fields(d), "agent": d._agent, "route": d._route,
            "command": d.command.model_dump()}


SUPERVISOR = Supervisor()
PROFILE = replace(dp.PROFILE, forced_moves=SUPERVISOR.forced_moves,
                  initial_counters=_initial_counters, summarize=_summarize,
                  cycle_extra_fields=_cycle_extra_fields)
