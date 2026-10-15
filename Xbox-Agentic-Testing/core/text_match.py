"""
text_match.py - OCR-tolerant text matching.

Console OCR is noisy in predictable ways, all seen on this rig:

    "Minecraft Dungeons II" -> "Dundecns Il"
    "Game Activity"         -> "Game activity"
    "Go to official club"   -> "@ Go to official club" / "Tel Go to official club"

Two levels, for two different risks:

  exact_contains()  normalised substring (case, punctuation, spacing only).
                    For decisions that SEND INPUT, e.g. guided navigation
                    deciding to stop and press A. Measured: fuzzy scored
                    "Achievement tracker" vs "Achievements" at 0.96, which
                    would have pressed A on the wrong row.
  contains()        additionally tolerates OCR letter errors (difflib ratio).
                    For identifying WHICH screen is showing, where a false
                    match only makes a guard more permissive.

Short targets get no fuzz at all: "Stats" must not match "Status" or "Start".
Heavy misreads ("Dundecns Il" for "Dungeons II", ratio 0.73) are not matched;
list the longer, more readable phrase ("minecraft dungeons") instead.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_NON_WORD = re.compile(r"[^a-z0-9&$]+")
DEFAULT_THRESHOLD = 0.82


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).lower()
    return " ".join(_NON_WORD.sub(" ", text).split())


def _threshold_for(needle: str, threshold: float) -> float:
    # Short targets get no fuzz: one wrong letter in "stats" is a different word.
    if len(needle) <= 5:
        return 1.0
    if len(needle) <= 8:
        return max(threshold, 0.88)
    return threshold


def line_similarity(line: str, target: str) -> float:
    """Best similarity between `target` and any same-length word window of `line`."""
    hay, needle = normalise(line), normalise(target)
    if not needle or not hay:
        return 0.0
    if needle in hay:
        return 1.0
    words, n = hay.split(), len(needle.split())
    best = 0.0
    for size in {n, n + 1, max(1, n - 1)}:
        for i in range(max(1, len(words) - size + 1)):
            window = " ".join(words[i:i + size])
            best = max(best, SequenceMatcher(None, window, needle).ratio())
    return best


def matched_lines(text: str, target: str,
                  threshold: float = DEFAULT_THRESHOLD) -> list[str]:
    """Screen lines that contain `target`, allowing for OCR noise."""
    limit = _threshold_for(normalise(target), threshold)
    return [line.strip() for line in str(text or "").splitlines()
            if line.strip() and line_similarity(line, target) >= limit]


def exact_contains(text: str, target: str) -> bool:
    needle = normalise(target)
    return bool(needle) and needle in normalise(text)


def exact_matched_lines(text: str, target: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines()
            if line.strip() and exact_contains(line, target)]


def contains(text: str, target: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    if normalise(target) and normalise(target) in normalise(text):
        return True
    return bool(matched_lines(text, target, threshold))


def first_match(text: str, targets: list[str],
                threshold: float = DEFAULT_THRESHOLD) -> str | None:
    return next((t for t in targets if contains(text, t, threshold)), None)
