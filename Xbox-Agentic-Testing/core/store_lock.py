"""
store_lock.py - cross-process locking for the JSON stores under artifacts/.

Lessons, shared lessons, cached routes and gameplay memories are each one JSON
file that every run reads, modifies and rewrites. Two processes doing that at
once (a suite plus a manual run, or two rigs sharing a drive) can each load the
old file, change it, and save - the second save silently discards the first
one's change. Writes were already atomic (tmp + replace), which prevents a
torn file but not this lost update.

`locked(path)` holds an OS-level lock on `<path>.lock` for the whole
read-modify-write. It is re-entrant within a thread, so a locked function may
call another one that locks the same store.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

# Long enough for any real read-modify-write (milliseconds), short enough that
# a stale lock never stalls a hardware run: on timeout the write proceeds
# unlocked and logs it, which is no worse than the behaviour before locking.
LOCK_TIMEOUT_SECONDS = float(os.environ.get("XBOX_AGENTIC_LOCK_TIMEOUT", "10"))

_locks: dict[str, FileLock] = {}
_guard = threading.Lock()


def _lock_for(path: Path) -> FileLock:
    key = str(path.resolve())
    with _guard:
        lock = _locks.get(key)
        if lock is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # thread_local=False + is_singleton semantics via the cache above
            # give re-entrancy across the whole process.
            lock = FileLock(key + ".lock", thread_local=False)
            _locks[key] = lock
        return lock


@contextmanager
def locked(path: Path | str) -> Iterator[None]:
    lock = _lock_for(Path(path))
    try:
        lock.acquire(timeout=LOCK_TIMEOUT_SECONDS)
    except Timeout:
        import logging
        logging.getLogger("store_lock").warning(
            "Could not lock %s within %.0fs; writing without the lock", path,
            LOCK_TIMEOUT_SECONDS)
        yield
        return
    try:
        yield
    finally:
        lock.release()
