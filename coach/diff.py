"""Word-level alignment between the target passage and the ASR transcript.

Pure code — no model. Produces the skeleton the UI highlights and Jev judges:
matched words, substituted words, skipped words, inserted words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def normalize(word: str) -> str:
    """Lowercase, strip punctuation — what counts as 'the same word' locally."""
    m = _WORD_RE.search(word.lower())
    return m.group(0) if m else ""


@dataclass
class WordStatus:
    index: int          # index into the passage sentence's word list
    text: str           # original passage word
    norm: str           # normalized passage word
    spoken: str | None  # what the ASR heard, if anything
    status: str         # "match" | "sub" | "skip" | "insert-context"
    spoken_j: int | None = None  # token index in the transcript (for melody arrows)


@dataclass
class Alignment:
    words: list[WordStatus]
    spoken_words: list[str]

    @property
    def suspects(self) -> list[WordStatus]:
        """Words Jev should judge: everything not a clean match."""
        return [w for w in self.words if w.status != "match"]

    @property
    def clean_ratio(self) -> float:
        if not self.words:
            return 0.0
        return sum(1 for w in self.words if w.status == "match") / len(self.words)


def align(passage_sentence: str, transcript: str) -> Alignment:
    target = _WORD_RE.findall(passage_sentence.lower())
    spoken = _WORD_RE.findall(transcript.lower())
    words: list[WordStatus] = []

    sm = SequenceMatcher(a=target, b=spoken, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                norm = target[i1 + k]
                words.append(WordStatus(i1 + k, norm, norm, spoken[j1 + k], "match",
                                        spoken_j=j1 + k))
        elif tag == "replace":
            # Pair up what we can, in order; leftovers become skips/inserts.
            n_t, n_s = i2 - i1, j2 - j1
            pairs = min(n_t, n_s)
            for k in range(pairs):
                norm = target[i1 + k]
                words.append(WordStatus(i1 + k, norm, norm, spoken[j1 + k], "sub",
                                        spoken_j=j1 + k))
            for k in range(pairs, n_t):
                norm = target[i1 + k]
                words.append(WordStatus(i1 + k, norm, norm, None, "skip"))
        elif tag == "delete":
            for k in range(i1, i2):
                norm = target[k]
                words.append(WordStatus(k, norm, norm, None, "skip"))

    # Spoken words the passage doesn't have (insertions) — keep for Jev context.
    matched_spoken = {w.spoken for w in words if w.spoken is not None}
    extras = [s for s in spoken if s not in matched_spoken]
    return Alignment(words=words, spoken_words=extras)


def summarize(a: Alignment) -> str:
    n_match = sum(1 for w in a.words if w.status == "match")
    n_sub = sum(1 for w in a.words if w.status == "sub")
    n_skip = sum(1 for w in a.words if w.status == "skip")
    return f"{n_match}/{len(a.words)} words matched, {n_sub} substituted, {n_skip} skipped"
