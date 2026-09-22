"""Phoneme-level acoustic scoring (the layer that actually listens).

Pipeline per sentence:
  1. espeak (en-us) gives the expected IPA phones per passage word, segmented
     into the wav2vec2 model's own phone inventory (longest-match)
  2. wav2vec2-lv-60-espeak-cv-ft runs on the audio -> per-frame phone posteriors
  3. frame-level forced alignment: each expected phone owns a run of frames;
     blanks are free between runs; a phone scores by the geometric mean of the
     EXPECTED phone's posterior over its frames (GOP semantics — a
     confidently-wrong sound scores low, never high)
  4. per-word verdict: geometric mean of its phone posteriors

Honest scope: segment accuracy (which sound was off, how far). Melody and
stress (prosody) are NOT scored here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import librosa
import numpy as np

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Vowel families a learner cannot usefully be told apart: the model's /ɐ/ and
# espeak's /ə/ are the same weak vowel, and a diphthong often comes back as
# its own sequence (ɔɪ -> ɔːɹ). Flagging these teaches hyperarticulation.
_CLOSE_FAMILIES = [{"ə", "ɐ", "ʌ"}, {"ɪ", "i"}, {"ʊ", "u"},
                   {"ɔ", "o", "ɒ", "oʊ"}, {"ɑ", "a"}, {"ɛ", "e"}, {"ɝ", "ɚ"}]


def _close_enough(expected: str, got: str) -> bool:
    if expected == got:
        return True
    if expected in got or got in expected:
        return True
    if any(expected in fam and got in fam for fam in _CLOSE_FAMILIES):
        return True
    # Diphthong split: the model often returns ɔɪ as ɔː + ɹ — same nucleus,
    # lengthened. Not a learner error.
    if (len(expected) >= 2 and got.startswith(expected[0] + "ː")):
        return True
    return False


# Weak forms: natives reduce these. Citation-form GOP will always flag them;
# telling a learner to say full /eɪ/ in "a" teaches the wrong habit.
_WEAK = {"the", "a", "an", "of", "to", "and", "or", "that", "for", "from",
         "at", "in", "on", "as", "was", "is", "are", "his", "her", "their"}


@dataclass
class PhoneVerdict:
    expected: str           # IPA phone the passage wants
    got: str | None         # IPA phone the audio favored there
    prob: float | None      # P(expected | frames); None = not in model vocab

    @property
    def confused(self) -> bool:
        return (self.prob is not None and self.got is not None
                and self.got != self.expected and self.prob < 0.5)


@dataclass
class WordSound:
    word: str
    score: float                        # 0..1 geometric mean of phone posteriors
    phones: list[PhoneVerdict]

    @property
    def worst(self) -> PhoneVerdict | None:
        bad = [p for p in self.phones if p.prob is not None and p.prob < 0.5]
        return min(bad, key=lambda p: p.prob) if bad else None


@lru_cache(maxsize=1)
def _espeak():
    from phonemizer.backend import EspeakBackend
    return EspeakBackend("en-us", with_stress=False, language_switch="remove-flags")


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    return proc, model, torch


@lru_cache(maxsize=1)
def _vocab() -> frozenset[str]:
    """The model's phone inventory — used to segment espeak IPA strings."""
    proc, _, _ = _model()
    vocab = set(proc.tokenizer.get_vocab().keys())
    vocab -= {"<pad>", "<s>", "</s>", "<unk>", "|"}
    return frozenset(vocab)


@lru_cache(maxsize=None)
def _decompose(token: str) -> tuple[str, ...] | None:
    """Minimal-piece decomposition of a compound phone into shorter vocab
    tokens, or None if it cannot be decomposed. Used when the model gives a
    compound token no mass (e.g. it emits 'aɪ'+'ə' but never 'aɪə')."""
    vocab = _vocab()
    n = len(token)
    best = [None] * (n + 1)   # best[i] = fewest pieces covering token[:i]
    best[0] = ()
    for i in range(1, n + 1):
        cand = None
        for j in range(i):
            if best[j] is None or token[j:i] not in vocab:
                continue
            if i == n and j == 0:
                continue  # the full token as a single piece is not a decomposition
            if cand is None or len(best[j]) + 1 < len(cand):
                cand = best[j] + (token[j:i],)
        best[i] = cand
    return best[n] or None


def _split_phones(ipa: str) -> list[str]:
    """espeak emits a word as one unspaced IPA string; split into the model's
    phones by greedy longest-match against its vocab."""
    text = "".join(ch for ch in ipa if ch not in "ˈˌ")
    vocab = sorted(_vocab(), key=len, reverse=True)
    specials = {"<pad>", "<s>", "</s>", "<unk>", "|"}
    out: list[str] = []
    i = 0
    while i < len(text):
        for tok in vocab:
            if tok not in specials and text.startswith(tok, i):
                out.append(tok)
                i += len(tok)
                break
        else:
            i += 1  # unknown mark — skip
    return out


@lru_cache(maxsize=None)
def expected_phones(word: str) -> tuple[str, ...]:
    norm = _WORD_RE.search(word.lower())
    if not norm:
        return ()
    line = _espeak().phonemize([norm.group(0)])[0].strip()
    return tuple(_split_phones(line))


def _frame_logprobs(wav_path: str, proc, model, torch):
    """(logposteriors [T,V], blank_id, vocab_ids, inv) for 16k mono audio."""
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits[0]
    probs = logits.softmax(-1).numpy()
    vocab_ids = proc.tokenizer.get_vocab()
    blank_id = vocab_ids["<pad>"]
    inv = {v: k for k, v in vocab_ids.items()}
    return np.log(probs + 1e-9), blank_id, vocab_ids, inv


def _align_frames(expected, logp, blank_id, vocab_ids, inv):
    """CTC forced alignment: every frame is consumed by exactly one choice —
    a phone-continuation, a phone-start, or blank. The path pays lp_phone for
    frames it assigns to a phone, so each phone naturally anchors at its peak;
    silence is absorbed by blank at ~0 cost. Phone score = geometric mean of
    P(expected) over its assigned frames (>= 1: its start frame)."""
    T = logp.shape[0]
    NEG = -1e9
    lp_blank = logp[:, blank_id]

    scored = [(i, ph) for i, ph in enumerate(expected) if ph in vocab_ids]
    unscored = [i for i, ph in enumerate(expected) if ph not in vocab_ids]
    m = len(scored)
    verdicts: list[PhoneVerdict | None] = [None] * len(expected)
    for i in unscored:
        verdicts[i] = PhoneVerdict(expected[i], expected[i], None)
    if m == 0:
        return verdicts

    lp_ph = [logp[:, vocab_ids[ph]] for _, ph in scored]

    # dp[i][t]: best score, phones 0..i-1 fully placed, frames 0..t consumed
    dp = [[NEG] * T for _ in range(m + 1)]
    ch = [[None] * T for _ in range(m + 1)]   # 'blank' | 'cont' | 'start'
    dp[0][0] = 0.0
    ch[0][0] = 'start0'
    for t in range(1, T):
        dp[0][t] = dp[0][t - 1] + lp_blank[t]
        ch[0][t] = 'blank'
    for i in range(1, m + 1):
        col = lp_ph[i - 1]
        for t in range(1, T):
            b, c = 'blank', None
            v = dp[i][t - 1] + lp_blank[t]                     # frame t is blank
            s = dp[i - 1][t - 1] + col[t]                      # phone i-1 starts at t
            if s > v:
                v, b, c = s, 'start', t - 1
            k = dp[i][t - 1] + col[t]                          # phone i-1 continues
            if k > v:
                v, b, c = k, 'cont', None
            if dp[i][t] < v:
                dp[i][t] = v
                ch[i][t] = b if c is None else ('start', c)
            if b == 'start':
                ch[i][t] = ('start', c)
    if dp[m][T - 1] <= NEG / 2:
        for i, ph in scored:
            verdicts[i] = PhoneVerdict(ph, None, 0.0)
        return verdicts

    # traceback: collect assigned frames per phone
    frames_of: dict[int, list[int]] = {i: [] for i in range(m)}
    i, t = m, T - 1
    while t >= 0:
        tag = ch[i][t]
        if tag == 'blank':
            t -= 1
        elif tag == 'cont':
            frames_of[i - 1].append(t)
            t -= 1
        else:  # ('start', prev_state_t)
            frames_of[i - 1].append(t)
            t = tag[1]
            i -= 1
            if t < 0 or i == 0:
                # remaining frames 0..t belong to leading blanks / dp[0]
                break

    for i in range(m):
        fs = sorted(frames_of[i]) or [T - 1]
        phone = scored[i][1]
        col = lp_ph[i]
        p = math.exp(float(np.mean([col[f] for f in fs])))
        if p < 0.05:
            # compound the model never emits (e.g. 'aɪə' spoken as 'aɪ'+'ə'):
            # re-score its minimal vocab decomposition. Sub-phones live in the
            # gap between this phone's neighbors (coarticulation slack ±5).
            subs = _decompose(phone)
            if subs and all(s in vocab_ids for s in subs):
                prev_end = max((max(frames_of[i - 1]) if i > 0 and frames_of[i - 1] else fs[0] - 6), -1)
                next_start = min(frames_of[i + 1]) if i + 1 < m and frames_of[i + 1] else fs[-1] + 6
                lo = max(0, min(prev_end + 1, fs[0] - 5))
                hi = min(T, max(next_start, fs[-1] + 6))
                window = range(lo, hi)
                logs = []
                best_frame, best_val = fs[0], -1e9
                for sub in subs:
                    scol = logp[:, vocab_ids[sub]]
                    fmax = max(window, key=lambda f: scol[f])
                    logs.append(scol[fmax])
                    if scol[fmax] > best_val:
                        best_val, best_frame = scol[fmax], fmax
                p = math.exp(float(np.mean(logs)))
                got = inv[int(np.argmax(logp[best_frame]))]
                verdicts[scored[i][0]] = PhoneVerdict(phone, got, p)
                continue
        best_frame = max(fs, key=lambda f: col[f])
        got = inv[int(np.argmax(logp[best_frame]))]
        verdicts[scored[i][0]] = PhoneVerdict(phone, got, p)
    return verdicts


def score_words(wav_path: str, words: list[str]) -> list[WordSound]:
    """Score how each passage word actually sounded in the audio."""
    proc, model, torch = _model()
    logp, blank_id, vocab_ids, inv = _frame_logprobs(wav_path, proc, model, torch)

    all_phones: list[tuple[int, str]] = []
    for wi, w in enumerate(words):
        for ph in expected_phones(w):
            all_phones.append((wi, ph))
    exp_seq = [ph for _, ph in all_phones]

    verdicts = _align_frames(exp_seq, logp, blank_id, vocab_ids, inv)

    per_word: dict[int, list[PhoneVerdict]] = {}
    for (wi, _), v in zip(all_phones, verdicts):
        per_word.setdefault(wi, []).append(v)

    out = []
    for wi, w in enumerate(words):
        vs = per_word.get(wi, [])
        scored = [v.prob for v in vs if v.prob is not None]
        if not scored:
            continue
        score = math.exp(sum(math.log(max(p, 1e-6)) for p in scored) / len(scored))
        out.append(WordSound(word=w, score=score, phones=vs))
    return out


def notes(word_sounds: list[WordSound], max_notes: int = 5) -> list[str]:
    """Plain-language notes for the worst phones, junk filtered out.

    - <pad>/blank is a model artifact, never a sound: it becomes "dropped".
    - Function words are skipped: natives reduce them.
    - Near-identical vowels are not flagged.
    - Sorted worst first, capped.
    """
    cand = []
    for ws in word_sounds:
        if ws.word.lower().strip(".,!?;:'\"“”‘’()") in _WEAK:
            continue
        for p in ws.phones:
            if p.prob is None or p.prob >= 0.4 or p.got is None:
                continue
            if p.got in {"<pad>", "<s>", "</s>", "<unk>", "|"}:
                cand.append((p.prob, f"'{ws.word}': /{p.expected}/ dropped"))
            elif not _close_enough(p.expected, p.got):
                cand.append((p.prob, f"'{ws.word}': /{p.expected}/ sounded like /{p.got}/"))
    cand.sort(key=lambda c: c[0])
    return [text for _, text in cand[:max_notes]]


def render(word_sounds: list[WordSound]) -> str:
    """ANSI line for the terminal: color by sound quality, notes underneath."""
    GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
    parts = []
    for ws in word_sounds:
        color = GREEN if ws.score >= 0.7 else YELLOW if ws.score >= 0.4 else RED
        parts.append(f"{color}{ws.word}{RESET}")
    line = " ".join(parts)
    ns = notes(word_sounds)
    if ns:
        line += f"\n  {DIM}sound notes:{RESET} " + "; ".join(ns)
    return line
