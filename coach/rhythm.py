"""Stress-timing and phrase-final tone.

English keeps roughly even beats between stressed words and squeezes the
small ones. The phrase tune lives on the last content word (the nucleus),
not on whatever token happens to be last, and not on a rule that every
question must rise.

Both scores prefer a reference read (your saved voice, else a model read).
Without one, they only flag the obvious: small words as long as big ones,
and a statement that rises on its nucleus.
"""

from __future__ import annotations

FUNCTION = {
    "the", "a", "an", "of", "to", "and", "or", "but", "in", "on", "at", "for",
    "from", "with", "by", "as", "is", "was", "were", "be", "been", "am", "are",
    "it", "its", "his", "her", "their", "this", "that", "these", "those",
    "my", "your", "our", "we", "you", "he", "she", "they", "i", "do", "did",
    "does", "have", "has", "had", "will", "would", "can", "could", "should",
    "not", "so", "if", "than", "then", "there", "here", "into", "onto", "over",
    "him", "them", "us", "me", "must", "may", "might", "shall",
    "mine", "yours", "ours",
}

# Wh-questions typically end falling; yes/no questions may rise.
_WH = {"who", "what", "where", "when", "why", "how"}

_STRIP = ".,!?;:\"'“”‘’()"
_ARROW = {"up": "↗", "down": "↘", "flat": "→", "none": "·"}


def clean(word: str) -> str:
    return word.lower().strip(_STRIP)


def is_function(word: str) -> bool:
    return clean(word) in FUNCTION


def content_indexes(words: list[str]) -> list[int]:
    return [i for i, w in enumerate(words) if clean(w) and not is_function(w)]


def nucleus(words: list[str]) -> int | None:
    """Where English puts the phrase tune.

    A word written in ALL-CAPS marks contrastive focus and wins outright
    (even a function word — “I said give it to HER” is stress on a pronoun).
    Otherwise: the last content word.
    """
    for i, w in enumerate(words):
        token = w.strip(_STRIP)
        if len(token) > 1 and token.isalpha() and token.isupper():
            return i
    found = content_indexes(words)
    if found:
        return found[-1]
    return len(words) - 1 if words else None


def durations(ts: dict[int, dict]) -> dict[int, float]:
    out = {}
    for i, t in ts.items():
        out[i] = max(0.0, float(t["end"]) - float(t["start"]))
    return out


def npvi(seq: list[float]) -> float | None:
    """Normalized pairwise variability. Higher = more long/short alternation."""
    pairs = [(a, b) for a, b in zip(seq, seq[1:]) if a + b > 0.02]
    if len(pairs) < 2:
        return None
    total = sum(abs(a - b) / ((a + b) / 2) for a, b in pairs)
    return 100.0 * total / len(pairs)


def _ordered(words: list[str], durs: dict[int, float]) -> list[float]:
    return [durs[i] for i in range(len(words)) if i in durs and durs[i] > 0]


def rhythm_line(sentence: str, user_ts: dict[int, dict],
                ref_ts: dict[int, dict] | None = None) -> str | None:
    """Stress-timing vs the reference, or vs a content/function duration ratio."""
    words = sentence.split()
    if len(words) < 3 or not user_ts:
        return None
    ud = durations(user_ts)
    user_pvi = npvi(_ordered(words, ud))
    content = [ud[i] for i in content_indexes(words) if i in ud and ud[i] > 0]
    function = [ud[i] for i, w in enumerate(words)
                if is_function(w) and i in ud and ud[i] > 0]
    long_small = []
    if content:
        beat = sorted(content)[len(content) // 2]
        long_small = [clean(words[i]) for i, w in enumerate(words)
                      if is_function(w) and i in ud and ud[i] > 0.85 * beat]

    if ref_ts:
        rd = durations(ref_ts)
        ref_pvi = npvi(_ordered(words, rd))
        if user_pvi is None or ref_pvi is None:
            return None
        if user_pvi < ref_pvi * 0.75:
            names = ", ".join(long_small[:3])
            extra = f" — squeeze {names}" if names else " — squeeze the small words"
            return f"too even{extra}"
        if user_pvi > ref_pvi * 1.45:
            return "beats jumpy — keep big words big, small words quick"
        return "stress timing close to the reference"

    if not content or not function:
        return None
    ratio = (sum(content) / len(content)) / (sum(function) / len(function))
    if ratio < 1.25:
        names = ", ".join(long_small[:3])
        extra = f" — squeeze {names}" if names else " — squeeze the small words"
        return f"small words as long as big ones{extra}"
    return "content words longer than small words"


def _slope_tone(slopes: dict[int, float] | None, i: int) -> str | None:
    """Tail-slope (st/s over the nucleus's final voiced ~400ms) -> tone.
    Slope beats the arrow when present; None falls back to the arrow."""
    if not slopes or slopes.get(i) is None:
        return None
    s = float(slopes[i])
    if s >= 2.5:
        return "up"
    if s <= -2.5:
        return "down"
    return "flat"


def phrase_final(sentence: str, user_dir: dict[int, str],
                 ref_dir: dict[int, str] | None = None,
                 user_slopes: dict[int, float] | None = None,
                 ref_slopes: dict[int, float] | None = None) -> str | None:
    """Tune on the nucleus, compared to the reference when one exists.

    user_slopes / ref_slopes: word index -> tail slope in semitones/sec over
    the word's final voiced ~400ms (from melody.word_pitch). A slope is a
    better read of the phrase-final tune than the coarse arrow, so it wins
    when present; arrows remain the fallback.
    """
    words = sentence.split()
    i = nucleus(words)
    if i is None:
        return None
    word = clean(words[i]) or words[i]
    you = user_dir.get(i)
    ref = (ref_dir or {}).get(i)
    y_tone = _slope_tone(user_slopes, i) or (you if you and you != "none" else None)
    r_tone = _slope_tone(ref_slopes, i) or (ref if ref and ref != "none" else None)
    if r_tone:
        mark = _ARROW.get(r_tone, r_tone)
        if y_tone == r_tone:
            return f"ending on “{word}” matches the reference ({mark})"
        if not y_tone:
            return f"listen to “{word}” — the reference goes {mark}"
        return f"ending on “{word}”: you {y_tone}, reference {mark}"
    if sentence.rstrip().endswith("?"):
        first = clean(words[0]) if words else ""
        if first in _WH:
            return (f"“{word}” ends the question — wh-questions usually "
                    f"fall; let the voice drop")
        return f"“{word}” ends the question — yes/no questions may rise; try a gentle lift"
    if y_tone == "down":
        return f"fell on “{word}” — natural phrase ending"
    if y_tone == "up":
        return f"rose on “{word}” — statements usually fall there"
    if y_tone == "flat":
        return f"flat on “{word}” — let the voice drop on “{word}”"
    return None
