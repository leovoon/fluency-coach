"""Word prominence — which words carry the beat, you vs the reference.

A word is prominent when its energy stands above the utterance's own middle
(energy_db_rel, dB above this recording's median). Both sides use the same
within-utterance yardstick, so only the shape is compared: no absolute dB
ever reaches the learner.

Marks: 'match' — both agree; 'flat' — the reference is prominent and you
aren't (missed the beat); 'extra' — you are prominent where the reference
isn't (pushed a small word).
"""

from __future__ import annotations

# dB above the utterance's own median energy for a word to count as prominent.
PROMINENT_DB_REL = 1.0


def _flag(entry) -> bool | None:
    """True/False for prominence, None when the data is missing or unusable.

    Accepts a bare energy_db_rel number (what ui stores) or a word_pitch-style
    dict with an 'energy_db_rel' key.
    """
    if isinstance(entry, dict):
        entry = entry.get("energy_db_rel")
    if entry is None:
        return None
    try:
        return float(entry) >= PROMINENT_DB_REL
    except (TypeError, ValueError):
        return None


def prominence(ref_pitch: dict[int, dict], user_pitch: dict[int, dict],
               words: list[str]) -> dict[int, str]:
    """Per-word prominence status over the passage words.

    ref_pitch / user_pitch map passage word index -> word_pitch-style entry
    (needs 'energy_db_rel'). A word with no usable data on either side gets
    no mark — an unknown is not a judgment.
    """
    marks: dict[int, str] = {}
    if not ref_pitch or not user_pitch:
        return marks
    for i in range(len(words)):
        r = _flag(ref_pitch.get(i))
        u = _flag(user_pitch.get(i))
        if r is None or u is None:
            continue
        if r and not u:
            marks[i] = "flat"
        elif u and not r:
            marks[i] = "extra"
        else:
            marks[i] = "match"
    return marks


def summarise(marks: dict[int, str] | None, words: list[str]) -> str | None:
    """One compact line, at most 2 words named. None when nothing to fix."""
    flats = sorted(i for i, m in (marks or {}).items() if m == "flat")
    extras = sorted(i for i, m in (marks or {}).items() if m == "extra")
    if not flats and not extras:
        return None
    parts: list[str] = []
    named = 0
    for i in flats:
        if named >= 2:
            break
        parts.append(f"beat lands on “{_word(words, i)}” — you flattened it")
        named += 1
    for i in extras:
        if named >= 2:
            break
        parts.append(f"“{_word(words, i)}” got pushed — the reference keeps it small")
        named += 1
    line = "; ".join(parts)
    return line[0].upper() + line[1:] if line else None


def _word(words: list[str], i: int) -> str:
    if 0 <= i < len(words):
        return words[i]
    return "that word"


def choose_focus(flow_marks: dict | None, stress: dict | None) -> str:
    """One feedback dimension per attempt: pauses > stress > melody.

    flow_marks: flow.score's {boundary: {"kind", "ms", "ok"}}. Any pause or
    link you missed wins the pass — connected speech is the most audible habit.
    stress: prominence's {word: "match"/"flat"/"extra"}. A beat you flattened
    or a small word you pushed comes next. Melody is the default lens when
    nothing else failed. Defensive: None / missing keys mean no judgment.
    """
    if isinstance(flow_marks, dict):
        for m in flow_marks.values():
            if (isinstance(m, dict) and m.get("kind") in ("pause", "link")
                    and m.get("ok") is False):
                return "flow"
    if isinstance(stress, dict):
        for v in stress.values():
            if v in ("flat", "extra"):
                return "stress"
    return "melody"
