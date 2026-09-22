"""Flow layer: where to pause, where to link — labeled from text, scored from audio.

Two kinds of marks between words (the teaching layer, shown before you speak):
  |   pause here — chunk/breath boundary (before conjunctions, after commas)
  ‿   link here — consonant-final word meets vowel-initial word: say them as one

After you speak, the parakeet-redux word timestamps give the real gap at each
boundary, so every mark turns green (you did it) or amber (you didn't). Big
gaps at unlabeled boundaries are hesitations — a different problem from
phrasing, and worth seeing separately.

Connected speech is where "sounding native" mostly lives: not the sounds, the
joins. Reducing the gap to ~zero at link points is the single most audible
habit.
"""

from __future__ import annotations

import re

PAUSE_MS = 150   # gap at a | boundary counts as having taken the pause
LINK_MS = 70     # gap at a ‿ boundary counts as connected
HESITATION_MS = 250  # gap at an unlabeled boundary = hesitation
PAUSE_FLOOR_MS = 80  # any gap >= this at a | boundary = you breathed there

# Reference-derived labeling: a boundary is a pause/link if the reference read
# actually pauses/links there. Rules become the fallback, not the source of truth.
# Parakeet word timestamps are quantized to 80ms frames, so measured gaps land on
# 0/80/160/240... TTS word spacing alone reaches 160ms mid-phrase with no breath
# intended, so a pause label needs a 3-frame gap: >= 200ms rounds up to 240.
REF_PAUSE_MS = 200
REF_LINK_MS = 45

_PAUSE_WORD = {"and", "but", "or", "so", "because", "when", "while", "if",
               "since", "although", "though", "before", "after", "until",
               "unless", "which", "who", "that"}
_VOWELS = set("aeiouɑæɛɪʊɔʌə")
_STRIP = ".,!?;:\"'“”‘’()"


def _clean(word: str) -> str:
    return word.lower().strip(_STRIP)


def _phoneme_hints() -> bool:
    """IPA link detection only when the tier asked for phonemes.

    expected_phones pulls the wav2vec2 vocab. lite/standard leave phonemes
    off so the first paint must stay on letter fallback and never download it.
    """
    try:
        from .config import load
        return bool(load().phonemes)
    except Exception:
        return False


def _last_sound(word: str, use_phones: bool) -> str:
    """Last IPA sound of the word, "" if phonemes are off or unknown."""
    if not use_phones:
        return ""
    try:
        from .phonemes import expected_phones
        phones = expected_phones(_clean(word))
        if phones:
            return phones[-1][-1]
    except Exception:
        pass
    return ""


def _ends_consonant(word: str, use_phones: bool = False) -> bool:
    w = _clean(word)
    if not w:
        return False
    sound = _last_sound(word, use_phones)
    if sound:
        return sound not in _VOWELS
    return w[-1] not in "aeiou"  # letter fallback when phonemes are off


def _starts_vowel(word: str, use_phones: bool = False) -> bool:
    w = _clean(word)
    if not w:
        return False
    if use_phones:
        try:
            from .phonemes import expected_phones
            phones = expected_phones(w)
            if phones:
                return phones[0][0] in _VOWELS
        except Exception:
            pass
    return w[0] in "aeiou"


def boundaries(sentence: str) -> dict[int, str]:
    """Labeled boundaries: {i: kind} where i = between word i and word i+1.

    A boundary can't be both — pause wins over link.
    """
    words = sentence.split()
    use_phones = _phoneme_hints()
    marks: dict[int, str] = {}
    for i in range(len(words) - 1):
        cur, nxt = words[i], words[i + 1]
        if cur[-1:] in ",;:" or _clean(nxt) in _PAUSE_WORD:
            marks[i] = "pause"
        elif _ends_consonant(cur, use_phones) and _starts_vowel(nxt, use_phones):
            marks[i] = "link"
    return marks


def score(words_ts: list[dict], sentence: str, spoken_j: dict[int, int],
          ref_gaps: dict[int, float] | None = None,
          ref_is_human: bool = False,
          labels: dict[int, str] | None = None) -> dict:
    """Score every boundary from measured gaps.

    words_ts: parakeet word timestamps ([{"word","start","end"}, ...]).
    spoken_j: passage word index -> transcript token index (diff.WordStatus.spoken_j).
    labels: explicit per-boundary labels ("pause"/"link"/"none"), e.g. decided
    by Jev. An explicit label wins over ref/rule inference; boundaries with no
    entry fall back to the measured reference, then rules. "none" means the
    judge considered the boundary and rejected a mark — no punctuation
    force-back happens there.
    ref_gaps: the reference read's measured gap per boundary (ms). When present,
    labels come from that read. Without ref_gaps, rule-based labels are used.
    ref_is_human: the gaps came from a saved human reference, not TTS.
    Punctuation forces a pause only then, or when there is no measured read.
    A TTS read that skipped a comma is not forced back into a pause.
    Breath scoring is place, not length: any real pause at the boundary or one
    word either side counts. Duration is not compared.
    Returns {"marks": {i: {"kind", "ms", "ok"}}, "hesitations": [ms...],
             "summary": str}.
    """
    words = sentence.split()
    rule = boundaries(sentence)
    out: dict[int, dict] = {}
    hesitations: list[float] = []

    def gap(i: int) -> float | None:
        a, b = spoken_j.get(i), spoken_j.get(i + 1)
        if a is None or b is None or b <= a or b >= len(words_ts):
            return None
        return max(0.0, (words_ts[b]["start"] - words_ts[a]["end"])) * 1000.0

    def _gap_or_zero(i: int) -> float:
        g = gap(i)
        return g if g is not None else 0.0

    for i in range(len(words) - 1):
        ms = gap(i)
        ref = (ref_gaps or {}).get(i)

        # Label: an explicit judgment wins; measure from the reference; rules
        # fill gaps only. "none" is an answer — no fallback inference runs.
        lab = (labels or {}).get(i)
        kind = lab if lab in ("pause", "link") else None
        punct = words[i][-1:] in ",;:"
        if lab is None:
            if ref is not None:
                if ref >= REF_PAUSE_MS:
                    kind = "pause"
                elif ref <= REF_LINK_MS:
                    kind = "link"
            if kind is None and i in rule:
                rule_kind = rule[i]
                # boundaries() marks every comma as a pause. That rule must not
                # sneak back in when a TTS read was measured and did not pause.
                if rule_kind == "pause" and punct and ref_gaps is not None and not ref_is_human:
                    rule_kind = None
                if rule_kind:
                    kind = rule_kind
            # Comma/semicolon/colon is a breath point for a human reference, and
            # for rule-only scoring. Do not force it onto a TTS read that ran through.
            if punct and (ref_is_human or ref_gaps is None):
                kind = "pause"

        ok = None
        if ms is not None:
            if kind == "pause":
                # Breath correctness is WHERE, not HOW LONG: any real pause at
                # this boundary or one word either side is natural phrasing.
                # (Breath duration varies too much between speakers to score.)
                ok = ms >= PAUSE_FLOOR_MS or _gap_or_zero(i - 1) >= PAUSE_FLOOR_MS \
                    or _gap_or_zero(i + 1) >= PAUSE_FLOOR_MS
            elif kind == "link":
                # Relative: within the reference's connection, capped at LINK_MS.
                threshold = min(ref + 60.0, LINK_MS) if ref is not None else LINK_MS
                ok = ms <= threshold
            elif ms >= max(HESITATION_MS, 2 * ref if ref else HESITATION_MS):
                ok = False
                hesitations.append(ms)
            else:
                ok = True
        out[i] = {"kind": kind, "ms": ms, "ok": ok}

    pauses = [v for v in out.values() if v["kind"] == "pause" and v["ok"] is not None]
    links = [v for v in out.values() if v["kind"] == "link" and v["ok"] is not None]
    parts = []
    if pauses:
        parts.append(f"pauses {sum(p['ok'] for p in pauses)}/{len(pauses)}")
    if links:
        parts.append(f"linked {sum(l['ok'] for l in links)}/{len(links)}")
    if hesitations:
        parts.append(f"{len(hesitations)} hesitation{'s' if len(hesitations) > 1 else ''}")
    return {"marks": out, "hesitations": hesitations,
            "summary": " · ".join(parts) if parts else ""}


def mark_char(kind: str | None) -> str:
    return "|" if kind == "pause" else ("‿" if kind == "link" else "·")


def ref_gaps_from_ts(ts: dict[int, dict], n_words: int) -> dict[int, float]:
    """Reference's measured gap (ms) at each passage boundary, from its word
    timestamps. This is what makes labels data-driven instead of rule-based."""
    out = {}
    for i in range(n_words - 1):
        a, b = ts.get(i), ts.get(i + 1)
        if a and b:
            out[i] = max(0.0, (b["start"] - a["end"])) * 1000.0
    return out


def link_groups(sentence: str, ref_gaps: dict[int, float]) -> list[tuple[int, int]]:
    """Maximal word spans (inclusive indices, >=2 words) whose internal
    boundaries are tight in the reference — the connected-speech chunks.
    Clicking any word in a span plays the whole span."""
    words = sentence.split()
    groups: list[tuple[int, int]] = []
    start = None
    for i in range(len(words) - 1):
        tight = ref_gaps.get(i, float("inf")) <= REF_LINK_MS
        if tight and start is None:
            start = i
        if not tight and start is not None:
            if i - start >= 1:
                groups.append((start, i))
            start = None
    if start is not None and (len(words) - 1) - start >= 1:
        groups.append((start, len(words) - 1))
    return groups


def pause_groups(sentence: str, marks: dict[int, str] | None = None) -> list[tuple[int, int]]:
    """Maximal word spans (inclusive indices) between pause marks — the phrase
    chunks a learner replays (their slice, then the reference's).

    marks is the labeled-boundary dict from boundaries()/marks_from_ref()
    ({boundary index: "pause" | "link"}); boundary i sits between word i and
    word i+1, and a "pause" there ends the current group. Link marks never
    split — a chunk runs through linked words. With no marks at all (or none
    labeled "pause") the sentence is one chunk, so replay always has something
    to play. Splitting can still leave a one-word chunk (a lone "However,"
    before the breath), which is a real phrase and is kept.

    Returns [] only when there is nothing chunkable: an empty sentence or a
    single word. Chunks describe WHERE the phrases split — place, never
    length. Clicking any word in a span plays the whole span.
    """
    words = sentence.split()
    n = len(words)
    if n < 2:
        return []
    pauses = {i for i, kind in (marks or {}).items() if kind == "pause"}
    if not pauses:
        return [(0, n - 1)]
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(n - 1):
        if i in pauses:
            groups.append((start, i))
            start = i + 1
    groups.append((start, n - 1))
    return groups


def marks_from_ref(sentence: str, ref_gaps: dict[int, float],
                    ref_is_human: bool = False) -> dict[int, str]:
    """Boundary labels measured from the reference read itself.

    Pauses: wherever that read breathed. Punctuation adds a pause only when
    the read is human (ref_is_human). A TTS read that did not pause is left
    alone — the model is not a native speaker to imitate on commas.
    Links: the reference confirms a tight join AND the junction is teachable
    (consonant→vowel). Marking every tight gap in fluent speech would just be
    noise — connected speech links nearly everywhere.
    """
    words = sentence.split()
    rule = boundaries(sentence)
    marks = {}
    for i, g in ref_gaps.items():
        punct = words[i][-1:] in ",;:"
        if g >= REF_PAUSE_MS or (punct and ref_is_human):
            if i == 0 and not punct:
                # A break between a sentence's first two words is hesitation in
                # the read, not phrasing — never a chunk boundary. (The rule
                # path can't produce it either, short of a leading 'However,'.)
                continue
            marks[i] = "pause"
        elif g <= REF_LINK_MS and rule.get(i) == "link":
            marks[i] = "link"
    return marks
