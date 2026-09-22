"""Melody (intonation) feedback — did your voice go up or down, per word.

No IPA, no phonetic symbols. Arrows a learner can read at a glance:
  ↗ rose   ↘ fell   → flat   · (no pitch — usually an unstressed function word)

Pitch comes from librosa.pyin over the whole utterance; each ASR word is scored
inside its own time window using the word timestamps parakeet-redux returns
(models.transcribe_detailed). Direction = semitone change from the word's first
voiced third to its last voiced third.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

import librosa

from . import rhythm

UP, DOWN, FLAT, NONE = "up", "down", "flat", "none"
_ARROW = {UP: "↗", DOWN: "↘", FLAT: "→", NONE: "·"}
# Semitones of net movement before a word counts as rising/falling.
THRESHOLD_SEMITONES = 1.5
# Tail-slope window: fit the last ~400ms of voiced frames of the word.
TAIL_S = 0.4
# Wh-questions (typically fall) vs yes/no questions (may rise).
WH_STARTERS = ("who", "what", "when", "where", "why", "which", "how", "whose")


def _fold_to(hz: float, anchor: float) -> float:
    """Pull an octave jump back toward anchor. A true octave leap inside one
    word is rare; pyin doubling on a final syllable is common."""
    if hz <= 0 or anchor <= 0:
        return hz
    while hz / anchor > 1.5:
        hz /= 2.0
    while anchor / hz > 1.5:
        hz *= 2.0
    return hz


def _fold_series(hzs: list[float | None]) -> list[float | None]:
    """Walk the utterance so a lone +12 st spike cannot survive next to its neighbors."""
    out: list[float | None] = []
    prev = None
    for hz in hzs:
        if not hz or hz <= 0:
            out.append(None)
            continue
        if prev:
            hz = _fold_to(hz, prev)
        out.append(hz)
        prev = hz
    return out


def _direction(f0: np.ndarray) -> str:
    f0 = f0[np.isfinite(f0)]
    if len(f0) < 3:
        return NONE
    anchor = float(np.median(f0))
    f0 = np.array([_fold_to(float(x), anchor) for x in f0])
    k = max(1, len(f0) // 3)
    start, end = float(f0[:k].mean()), float(f0[-k:].mean())
    if start <= 0 or end <= 0:
        return NONE
    semitones = 12.0 * np.log2(end / start)
    if semitones > THRESHOLD_SEMITONES:
        return UP
    if semitones < -THRESHOLD_SEMITONES:
        return DOWN
    return FLAT


def _tail_slope(times: np.ndarray, st: np.ndarray) -> float | None:
    """Semitones/sec fitted over the last ~400ms of voiced frames (fallback:
    last voiced third when the word is shorter than the window). None when
    there are too few frames to fit."""
    if len(times) < 3:
        return None
    span = float(times[-1] - times[0])
    if span > TAIL_S:
        sel = times >= times[-1] - TAIL_S
        t, s = times[sel], st[sel]
        if len(t) < 3:
            return None
    else:
        k = max(1, len(times) // 3)
        t, s = times[-k:], st[-k:]
    if len(t) < 3 or float(np.ptp(t)) <= 0:
        return None
    return round(float(np.polyfit(t, s, 1)[0]), 2)


def word_pitch(wav_path, words: list[dict]) -> list[dict]:
    """Score each word {"word","start","end"} -> adds "direction", "st",
    "slope" and "energy_db_rel".

    "st" is where the voice landed on that word (last third of voiced frames),
    in semitones relative to this recording's own median. Octave jumps from
    the pitch tracker are folded back toward the previous word, so a falling
    final syllable is not drawn an octave high.

    "slope" is the tail tune: semitones/sec fitted over the last ~400ms of
    voiced frames of the word. "energy_db_rel" is loudness relative to the
    utterance itself: the word's peak RMS level minus the median of all the
    words' peaks. Both are within-utterance relative measures — mic-safe, no
    absolute numbers reach the learner.

    The two lines on the chart each use their own median. +12 on the reference
    and -8 on you are not "the reference is higher in Hz." Compare the shape.
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    f0, voiced, _ = librosa.pyin(
        audio, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=256)
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
    rms_times = librosa.times_like(rms, sr=sr, hop_length=256)
    out = []
    landings = []
    slopes = []
    peaks = []
    for w in words:
        mask = (times >= w["start"]) & (times <= w["end"]) & (voiced == True)
        f = f0[mask]
        t = times[mask]
        good = np.isfinite(f)
        f, t = f[good], t[good]
        landing = None
        slope = None
        if len(f):
            k = max(1, len(f) // 3)
            landing = float(np.median(f[-k:]))
            # Same semitone series, word-anchored, for the tail fit.
            med_f = float(np.median(f))
            st = 12.0 * np.log2(np.maximum(f, 1e-10) / med_f)
            slope = _tail_slope(t, st)
        landings.append(landing)
        slopes.append(slope)
        rmask = (rms_times >= w["start"]) & (rms_times <= w["end"])
        peak = float(np.percentile(db[rmask], 90)) if rmask.any() else None
        peaks.append(peak)
        out.append({**w, "st": None, "direction": _direction(f),
                    "slope": slope, "energy_db_rel": None})
    folded = _fold_series(landings)
    voiced_hz = [h for h in folded if h]
    med = float(np.median(voiced_hz)) if voiced_hz else 0.0
    real_peaks = [p for p in peaks if p is not None]
    med_peak = float(np.median(real_peaks)) if real_peaks else None
    for w, hz, peak in zip(out, folded, peaks):
        if hz and med > 0:
            w["st"] = round(12.0 * np.log2(hz / med), 2)
        if peak is not None and med_peak is not None:
            w["energy_db_rel"] = round(peak - med_peak, 2)
    return out


def verdict(word_pitch_result: list[dict], sentence: str) -> str | None:
    """Plain-language read on the phrase tune, judged on the nucleus (the last
    content word), not the last token. Yes/no questions may rise; wh-questions
    typically fall. None if nothing was voiced."""
    if not word_pitch_result:
        return None
    words = [w["word"] for w in word_pitch_result]
    i = rhythm.nucleus(words)
    if i is None:
        return None
    entry = word_pitch_result[i]
    d = entry["direction"]
    if d == NONE:
        return None
    if sentence.rstrip().endswith("?"):
        wh = sentence.strip().lower().startswith(WH_STARTERS)
        if wh:
            if d == DOWN:
                return "fell on “%s” — natural for a wh-question" % entry["word"]
            return "wh-questions usually fall — let the voice drop on “%s”" % entry["word"]
        if d == UP:
            return "rose on “%s” — natural for a yes/no question" % entry["word"]
        return "yes/no questions often rise — try lifting “%s”" % entry["word"]
    if d == DOWN:
        return "fell on “%s” — natural for a statement" % entry["word"]
    if d == FLAT:
        return "stayed level on “%s” — try letting the voice drop" % entry["word"]
    return "rose on “%s” — sounds like a question; let the voice fall" % entry["word"]


def render(word_pitch_result: list[dict], paint=None) -> str:
    """Transcript with an arrow after each word. paint(direction)->str wraps the arrow."""
    parts = []
    for w in word_pitch_result:
        arrow = _ARROW[w["direction"]]
        if paint:
            arrow = paint(w["direction"], arrow)
        parts.append(f"{w['word']} {arrow}")
    return " ".join(parts)


def reference_arrows(sentence: str, ref_wav=None, origin: str = "engine") -> dict:
    """Melody data from a reference wav, or a synthesized read if none is given.

    The wav is transcribed (parakeet word timestamps), pitch-scored (pyin),
    and mapped onto the passage words via the diff. Returns
    {"arrows": {idx: dir}, "st": {idx: semitones}, "ts": {idx: {start, end}},
     "slopes": {idx: tail semitones/sec}, "energy": {idx: relative dB},
     "source": origin}.
    origin is whatever the caller passes (default "engine"). A path is not
    proof of a human read — pass origin="human" only for a saved reference.
    A model rendering is a model read, not the correct native answer.
    """
    from pathlib import Path
    from . import diff as diffmod, models
    tmp = None
    source = origin
    try:
        if ref_wav is not None:
            wav = Path(ref_wav)
        else:
            wav = models.synthesize(sentence)
            tmp = wav
        asr = models.transcribe_detailed(wav)
        if not asr["words"]:
            return {"arrows": {}, "st": {}, "ts": {}, "slopes": {}, "energy": {},
                    "source": source}
        alignment = diffmod.align(sentence, asr["text"])
        wp = word_pitch(wav, asr["words"])
        arrows, st, ts, slopes, energy = {}, {}, {}, {}, {}
        for w in alignment.words:
            if w.spoken_j is not None and w.spoken_j < len(wp):
                entry = wp[w.spoken_j]
                arrows[w.index] = entry["direction"]
                st[w.index] = entry["st"]
                ts[w.index] = {"start": entry["start"], "end": entry["end"]}
                slopes[w.index] = entry["slope"]
                energy[w.index] = entry["energy_db_rel"]
        return {"arrows": arrows, "st": st, "ts": ts, "slopes": slopes,
                "energy": energy, "source": source}
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)
