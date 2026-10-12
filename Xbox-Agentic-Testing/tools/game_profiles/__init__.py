"""game_profiles - per-game GameProfile definitions for gameplay_engine.

Each module here supplies the genuinely game-specific pieces (move
vocabulary, frame schema, system prompt, dispatch, stuck-counter grouping)
that gameplay_engine.run_gameplay_loop() is parameterized over. Adding a new
game means adding one module here, a tool module under tools/ that exposes it
(see tools/dungeons_gameplay.py), and listing that tool module in
tools/registry.py - nothing in gameplay_engine.py should need to change.
"""
