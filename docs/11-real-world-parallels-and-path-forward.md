# 11 — Real-world parallels, where this project stands, and what's next

An honest map: which real industries build this exact shape of software, where
this project currently sits against that map, what's missing to call it
"complete," and which direction to take from here — with reasoning, not just
options.

---

## 1. Real-world scenarios that build this exact shape of software

All of these share the same skeleton this project already has:
**ACT on real/simulated hardware → OBSERVE the result → DECIDE what it means →
report evidence.** That skeleton is not specific to games or to Xbox — it is
how you automate anything you cannot query directly and must instead watch.

| Scenario | Who does this for real | Action layer | Perception layer | Decision layer |
|---|---|---|---|---|
| **Console/game QA farms** | Sony, Microsoft, King, Unity's own test infra | Robot fingers or emulated controllers pressing real hardware | Camera or framebuffer capture | Scripted assertions, increasingly vision-model judges |
| **RPA with computer-vision** | UiPath, Automation Anywhere | Simulated mouse/keyboard on a real OS | Screen capture, OCR, image matching (no DOM/API access) | Rule engine or LLM judging "did the expected screen appear" |
| **Hardware-in-the-loop (HIL) testing** | Automotive ECU labs, avionics certification | Signal generators/actuators driving a real ECU | Oscilloscopes, CAN bus sniffers, cameras | Pass/fail against a spec, with "inconclusive" as a first-class outcome |
| **Compliance/certification labs** | USB-IF, HDMI Forum, FCC test houses | Automated button/signal sequences on a real device under test | Logic analyzers, protocol analyzers | Deterministic verdicts — a false pass is a liability, not just a bug |
| **Warehouse/manufacturing robotics** | Amazon robotics, pick-and-place lines | Real robotic arms | Vision-guided grasp verification | Closed-loop retry, not open-loop "fire and hope" |
| **Game-playing AI research** | Voyager (NVIDIA), VPT/STEVE-1 (OpenAI), DreamerV3 & SIMA (DeepMind) | Emulated controller / keyboard-mouse | Raw pixels | LLM-written code, learned policy, or world-model planning |

**The unifying lesson from every column above, and the one this repo's own
`docs/07-lessons-learned.md` already learned the hard way:** the ACT layer
lying to you ("the write succeeded") is the single most common and most
expensive failure mode in this entire category of software. Every serious
system in that table treats "I sent the command" and "the command had the
intended effect" as two separate, independently-verified facts. That is
exactly the rule this framework already enforces structurally (`PASS` with no
observational evidence gets downgraded to `INCONCLUSIVE` by a pydantic
validator, not by a prompt asking nicely).

---

## 2. Where this project actually stands against that map

Split honestly into the two halves this repo already contains, because they
are at very different maturity levels.

### 2a. The scenario-testing pipeline (`Xbox-Agentic-Testing`'s main workflow)

This is the **HIL/compliance-lab column** of the table above, and it is the
most mature part of the project.

| Capability | Status | Real-world equivalent it matches |
|---|---|---|
| Real hardware action (button presses land on a real console) | ✅ Verified | HIL actuation |
| Perception (capture card, not a webcam pretending to be one) | ✅ Verified, with a guard against the exact webcam-substitution failure documented in the README | Camera/logic-analyzer capture |
| "Command accepted" ≠ "command worked," enforced structurally | ✅ Verified (pydantic validator + smoke test 7) | The one universal lesson from every column above |
| Deterministic vs. LLM-judged split, by construction (executor can't judge, verifier can't act) | ✅ Verified (smoke test 3) | Separation of actuation and QA sign-off in every cert lab |
| First-class `BLOCKED` verdict, distinct from `FAIL` | ✅ Implemented, own exit code | "Rig fault" vs. "device under test fault" — the same distinction every cert lab makes |
| Config-driven agent/graph topology (no hand-wired control flow) | ✅ Implemented | Configurable test-station tooling |
| Regression suite for the framework itself | ✅ `smoke_test.py`, `test_fixes.py` | Standard test-infra hygiene |

**This half is close to "complete" for a single-rig, single-operator tool.**
What it is missing to be *industrial*-grade (not "wrong," just "not built
yet"):
- No parallelization — one capture device, one run at a time, by design.
- No device farm / fleet management (every real QA-farm column above runs
  many rigs at once).
- `game_launch_wait: 30.0` and similar timing constants are still
  placeholders, not measurements, per the README's own "Honest limitations."
- Authentication (holding Guide for 2s) is still a human-in-the-loop step;
  every real cert lab in the table above has fully unattended actuation.

### 2b. The autonomous-gameplay half (Minecraft/Max profiles, `combo_dispatch`)

This is the **game-playing-AI-research column**, and it is genuinely at
**prototype stage**, not "complete" — and that is not a criticism, it is
exactly where a project this new should be.

| Capability | Status | Closest research equivalent |
|---|---|---|
| Vision-LLM-per-frame decide/act loop | ✅ Working, hardware-verified | The reasoning-loop half of Voyager |
| Simultaneous multi-control dispatch (move+look+attack in one call) | ✅ Working, hardware-verified | Basic to any real-time control policy — this project only just added it |
| Structured move vocabulary per game, via a shared engine | ✅ Working for 2 games (Max, Minecraft) | Voyager's fixed action primitives (before it adds code-gen) |
| Coordinate/location memory across runs | ✅ Working, coordinate-keyed | A weak analogue of a world model |
| Reusable **skill/macro** memory (verified move *sequences*, not just places) | ✅ Built + wired into `gameplay_engine.py` (lookup, replay, auto-report via the loop's own measured delta) — offline-verified AND hardware+LLM-verified on 2026-09-28 (real GIMX dispatch, real Claude scene classification, real measured delta, real success_rate update). Still OFF by default per profile (`use_skill_memory=False`) — the live test used a temporary `dataclasses.replace()` copy, no profile file was changed | Voyager's skill library — its single biggest capability multiplier |
| Self-directed goal setting / curriculum | ✅ Built + wired into `gameplay_engine.py` (`CurriculumProposal`, `_propose_next_goal()`, `use_curriculum`/`curriculum_interval`) — offline-verified (schema construction, and a broken-context call proven to fall back to the unchanged goal rather than raise) AND hardware+LLM-verified on 2026-09-28 in a Creative-mode world (no death risk): a real 3-cycle run changed goal at cycle 2 from "Find a tree and inspect the area" to "Approach and punch the tree visible on the right side to collect wood logs", visibly changed cycle 3's own reasoning, and the final payload's `goal` field reflected the change. Still OFF by default (`use_curriculum=False`) — the live test used a temporary `dataclasses.replace()` copy, no profile file was changed | Voyager's automatic curriculum |
| Learned policy / pretraining on demonstration data | ❌ Not attempted, not needed at this scale | VPT/STEVE-1, DreamerV3, SIMA |
| Success metric / benchmark to know if changes actually help | ✅ Built (`tools/gameplay_benchmark.py`: `run_benchmark`/`compare_benchmark_runs`) — offline-verified (profile resolution, no-history failure path, `best_cycles_to_terminal`/`terminal_rate` scoring) AND hardware+LLM-verified on 2026-09-28: two real runs of the same named scenario recorded and compared, correctly showing `best_cycles_to_terminal: None` and `terminal_rate: 0.0` since neither real run reached the terminal state | Any RL/agent research loop needs this to iterate at all |
| Regression suite for gameplay quality specifically | ❌ Not built (the scenario side has one; gameplay does not) | Standard for any agent that changes behavior over time |

---

## 3. What's actually left to build, concretely

In priority order, cheapest/highest-leverage first:

1. **A skill/macro library** — DONE: `tools/skill_memory.py` (the journal,
   same persistence pattern as `location_memory.py`) plus the wiring into
   `gameplay_engine.py` (`GameProfile.use_skill_memory`/`skill_success_
   threshold`/`skill_min_attempts`, a `_skill_lookup()` helper, and an
   automatic `report_skill_outcome` call using the loop's own measured
   delta as the success signal). Verified OFFLINE with no hardware: move
   reconstruction from a stored dict, and all four lookup branches (no
   skill yet / not enough attempts yet / high win-rate replay / demoted
   after failures). HARDWARE+LLM-VERIFIED (2026-09-28): seeded a real
   'in_gameplay' skill (look right), ran one real `run_gameplay_loop` cycle
   against the live console with a temporary `use_skill_memory=True` test
   profile - the real vision model classified the scene and proposed its
   own move (walk forward, mine a log), but the loop correctly discarded
   that and replayed the seeded skill instead (logged: "replaying saved
   'in_gameplay' sequence instead of asking the model to re-decide it"),
   dispatched it for real (`right_stick right @0.60 for 0.40s ok`, measured
   delta 0.141), and auto-reported the outcome, updating success_rate from
   1.0 to 0.5. No profile has `use_skill_memory=True` in its committed
   defaults - today's live gameplay behavior is unchanged unless a profile
   opts in.
2. **A self-directed curriculum wrapper** — DONE: `CurriculumProposal` +
   `_propose_next_goal()` in `gameplay_engine.py`, gated by
   `GameProfile.use_curriculum`/`curriculum_interval` (default off, checked
   every N cycles rather than every cycle to avoid doubling the LLM
   round-trip cost). Verified OFFLINE: schema construction, and a
   deliberately broken context proven to fall back to the unchanged goal
   rather than raise. HARDWARE+LLM-VERIFIED (2026-09-28, Creative-mode
   world - chosen specifically to remove death risk during testing): a
   real 3-cycle run started with "Find a tree and inspect the area", the
   curriculum call fired at cycle 2 and changed the goal to "Approach and
   punch the tree visible on the right side to collect wood logs", cycle
   3's own reasoning visibly picked up the new goal's specifics, and the
   run's final `goal` field reflected the change. No profile has
   `use_curriculum=True` in its committed defaults.
3. **A gameplay regression/benchmark suite** — DONE: `tools/gameplay_
   benchmark.py` (`run_benchmark` wraps `run_gameplay_loop` and scores/
   records the result under a named scenario; `compare_benchmark_runs`
   lists a scenario's full history plus `best_cycles_to_terminal` and
   `terminal_rate`, same persisted-journal pattern as `location_memory.py`
   and `skill_memory.py`). Verified OFFLINE: profile-key resolution
   (known and unknown keys), the no-history failure path, and the scoring
   math against seeded fake records (confirmed `best_cycles_to_terminal`
   correctly ignores runs that never reached terminal rather than treating
   `None` as 0). HARDWARE+LLM-VERIFIED (2026-09-28): ran the SAME named
   scenario twice for real (real GIMX dispatch, real Claude scene
   classification, real deltas of 0.5436 and 0.5693 mean), recorded both,
   and compared them - correctly reported `total_runs: 2`,
   `terminal_rate: 0.0`, `best_cycles_to_terminal: None` since neither real
   run's goal ever asked to reach a success state, proving the "None is
   worse than any finite number" rule holds with genuine data, not just
   the offline fake records.
4. **Close the two explicitly-flagged unverified gaps** — DONE, both halves:
   (a) Real melee-hit confirmation: added `HitCheck` + `_check_hit()` in
   `combat_tools.py` - a second, focused vision call over the after-attack
   frame ONLY on cycles where an attack was actually dispatched, judging a
   real visual reaction (damage flash/knockback) rather than inferring a
   hit from pixel delta (which an attack's own swing animation already
   moves regardless of whether it connected). Tallied as `hits_confirmed`
   in the run payload. HARDWARE+LLM-VERIFIED (2026-09-28): ran the real
   check against a real live frame with NO mob present and got
   `hit_confirmed: False` with honest evidence ("No mob is visible..."),
   proving it does not hallucinate a hit - a real mob encounter to see a
   TRUE case is still needed (Creative-mode testing avoids risk but rarely
   spawns hostiles; this remains the one still-open half of this item).
   (b) Continuous-aim primitive: added `ComboComponent(kind="axis")` to
   `combo_dispatch.py` - arbitrary -1..1 x/y (not a named direction), the
   actual capability the Magic Marker needed. HARDWARE-VERIFIED: dispatched
   a real axis=(0.3,-0.7) combo and measured a real delta of 0.061 (camera
   genuinely moved). Deliberately did NOT migrate the Magic Marker's full
   gesture onto `dispatch_combo` - documented in `combo_dispatch.py` why:
   the marker's real sequence polls an ink gauge mid-hold via
   `_draw_until_empty` and keeps going until the gauge empties, not until a
   timer does; `dispatch_combo`'s assert-hold-release shape has no
   mid-hold-callback concept, and growing it into one for a single
   consumer would be a rewrite nothing else currently needs.
5. **LLM round-trip latency** — PARTIALLY DONE via Anthropic prompt
   caching: `gameplay_engine.decide()` now accepts a `cacheable_prefix`
   (a game's `system_prompt`, byte-identical every cycle) and marks it
   with Anthropic's `cache_control` so repeated cycles reuse the
   provider's cached read instead of reprocessing the same few thousand
   tokens every call. Gated to only engage when the provider is actually
   `anthropic` AND the prefix clears the real ~1024-token minimum (below
   that, Anthropic silently does not cache at all - claiming a benefit
   there would be dishonest) - every other case (short prompt, other
   provider, or no prefix given) falls back to today's unchanged single-
   block prompt. Verified OFFLINE: all four branches (long-prefix+
   anthropic splits and marks the block; short-prefix stays single-block;
   non-anthropic provider stays single-block; no-prefix-arg call stays
   single-block) using a fake decider with no real LLM call.
   HARDWARE+LLM-VERIFIED (2026-09-28): 4 real `decide()` calls against the
   live game with the SAME real system_prompt each time - cycle 1 (cold,
   writes the cache) took 3.971s; cycles 2-4 (reading from cache) took
   3.003s, 3.062s, 2.797s - a real, consistent ~25% latency drop after the
   first cycle, not a one-off measurement. Checked `combat_tools.py`'s and
   `exploration_tools.py`'s own system prompts too (649 and 430 tokens) -
   both are BELOW the real caching minimum, so wiring caching into them
   today would be a documented no-op, not a real improvement; left
   unchanged rather than adding dead code. Model selection (a
   smaller/faster model for routine cycles) and streaming remain open -
   only the caching half of this item is done.
6. Only if/when the above is exhausted and more autonomy is genuinely
   needed: consider a learned-policy layer (VPT-style pretraining or
   DreamerV3-style world-model planning) as a *replacement* for the
   per-frame LLM loop on the specific sub-skills that turn out to be
   bottlenecks (e.g., precise mining aim). This is listed last deliberately
   — see the recommendation below for why.

---

## 4. Which approach is better from here — and why

**Recommendation: keep the LLM-per-frame decision loop as the core, and
spend the next effort on the skill library + curriculum (items 1–2 above),
not on switching to a learned policy (VPT/DreamerV3-style).**

Reasoning, weighed directly against the alternatives in the table:

- **This project's entire culture is auditable evidence** — the README's
  central rule ("the command was accepted is never evidence anything
  happened"), the pydantic-enforced downgrade to `INCONCLUSIVE`, frames saved
  before the verifier sees them so a human can check the machine's own
  reasoning. A learned policy (VPT, DreamerV3, SIMA) is a black box: it
  cannot explain *why* it pressed a button, which is the opposite of what
  every other part of this codebase was built to guarantee. Switching to one
  would throw away the project's actual competitive advantage.
- **Data and compute cost don't fit a single-rig project.** VPT trained on
  70,000 hours of labeled YouTube footage; DreamerV3 needs a full RL training
  loop with a learned world model. This project has one Xbox, one capture
  card, and no GPU training cluster — that path is not "harder," it is a
  different project with a different budget.
- **Voyager's actual lesson generalizes cheaply, everything else in it
  doesn't.** Voyager's code-generation and its use of a general-purpose LLM
  are close to what's already built here (a vision LLM reasoning per frame).
  Its *specific* leverage — skill library + self-curriculum — is a
  straightforward addition on top of infrastructure that already exists
  (`location_memory.py` is proof the pattern works in this codebase), not a
  new paradigm.
- **The scenario-testing half should NOT be touched to chase this.** It is
  already the most mature, most-verified part of the project and matches
  industrial HIL/cert-lab practice closely. Effort there should stay on
  hardening (parallelization, replacing placeholder timing constants,
  reducing the manual-auth step), not on gameplay-style autonomy — they are
  different problems with different correctness bars (`BLOCKED`/`FAIL`
  precision vs. "did it eventually get the diamond").

**In short:** you are not behind because you're missing a fundamentally
different technique — you are behind because the one technique you've
correctly chosen (LLM-per-frame + verified hardware dispatch) is still
missing its memory. Add the skill library next; everything else on the list
compounds on top of that far more cheaply than any learned-policy rewrite
would.
