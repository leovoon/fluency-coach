#!/usr/bin/env python3
"""Fluency coach — read a passage aloud, get corrected on the spot.

Loop per sentence:
  1. show the sentence
  2. press Enter, read aloud; capture stops on trailing silence
  3. parakeet-redux (moondream Photon, local) transcribes
  4. local word-diff highlights match / sub / skip instantly
  5. melody arrows show where your voice rose/fell per word
  6. one batched Jev call (Experiential Labs) judges suspect words + sentence
  7. TTS reads the correct sentence back (Kokoro by default)

Own-voice reference always wins over a model read. Feature flags come from
coach.yaml (or FLUENCY_CONFIG), then CLI. A flag is never a passage path.

usage:
  python -m coach.main [passage.txt]     # built-in passage if omitted
  python -m coach.main essay.txt         # your passage
  python -m coach.main --no-jev          # diff-only, no charged calls
  python -m coach.main --no-tts          # skip read-back
  python -m coach.main --no-phonemes     # skip acoustic scoring
  python -m coach.main --no-melody       # skip intonation arrows
  python -m coach.main --no-flow         # skip pause/link scoring
  python -m coach.main --tier lite       # lite | standard | pro
  python -m coach.main --device cpu      # auto | cpu | mps | cuda
  python -m coach.main --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from . import diff as diffmod
from . import jev, melody, models, phonemes, record
from .config import load, passage_arg

DEFAULT_PASSAGE = (
    "This app was built to help you read aloud with better melody and rhythm. "
    "It is written in Python, and the screen you see is a NiceGUI web page "
    "served to your browser. "
    "When you read a sentence, a Parakeet speech model listens and writes "
    "down exactly what you said. "
    "A PocketTTS model then reads it back in a cloned voice, running on the "
    "Apple Neural Engine, and every single word is spoken in that same voice. "
    "Your recording and the model read are compared word by word, so pauses, "
    "pitch, and stress become the marks you see on screen. "
    "Everything runs locally on this machine, and a browser extension can "
    "grab any article and turn it into a practice session. "
    "You are, right now, reading how the app you are using was built."
)

GREEN, RED, YELLOW, GRAY, BOLD, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[1m", "\033[2m", "\033[0m",
)


def load_passage(path: str | None) -> list[str]:
    import re
    text = Path(path).read_text() if path else DEFAULT_PASSAGE
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences


def show_sentence(sentence: str, alignment: diffmod.Alignment | None = None) -> None:
    words = sentence.split()
    if alignment is None:
        print(f"\n{BOLD}{' '.join(words)}{RESET}\n")
        return
    # Re-derive coloring by matching diff words back to the sentence's words.
    colored = []
    by_index = {w.index: w for w in alignment.words}
    norm_index = 0
    for raw in words:
        norm = diffmod.normalize(raw)
        status = None
        while norm_index < len(words):
            st = by_index.get(norm_index)
            if st and st.norm == norm:
                status = st.status
                break
            norm_index += 1
        norm_index += 1
        color = {"match": GREEN, "sub": RED, "skip": GRAY}.get(status, YELLOW)
        colored.append(f"{color}{raw}{RESET}")
    print("\n" + " ".join(colored) + RESET + "\n")


def suspect_payload(sentence: str, alignment: diffmod.Alignment) -> list[dict]:
    payload = []
    for w in alignment.suspects:
        if w.status == "match":
            continue
        payload.append({
            "index": w.index,
            "word": w.text,
            "status": w.status,
            "spoken": w.spoken,
        })
    return payload


def run_sentence(sentence: str, use_jev: bool, use_tts: bool, use_phonemes: bool,
                 use_melody: bool) -> None:
    show_sentence(sentence)
    input(f"{DIM}  press Enter, then read the sentence aloud…{RESET}")
    wav = None
    try:
        wav = record.record_utterance()

        t0 = time.time()
        asr = models.transcribe_detailed(wav)
        transcript = asr["text"]
        asr_ms = (time.time() - t0) * 1000
        print(f"\n{DIM}  you said ({asr_ms:.0f}ms ASR):{RESET} {transcript}")

        alignment = diffmod.align(sentence, transcript)
        print(f"  {DIM}local diff:{RESET} {diffmod.summarize(alignment)}")
        show_sentence(sentence, alignment)

        if alignment.spoken_words:
            print(f"  {YELLOW}extra words not in passage:{RESET} "
                  f"{' '.join(alignment.spoken_words)}")

        if use_melody and asr["words"]:
            try:
                t1 = time.time()
                wp = melody.word_pitch(wav, asr["words"])
                mel_ms = (time.time() - t1) * 1000
                arrows = melody.render(wp, paint=lambda d, a: f"{DIM}{a}{RESET}")
                print(f"  {DIM}melody ({mel_ms:.0f}ms):{RESET} {arrows}")
                v = melody.verdict(wp, sentence)
                if v:
                    print(f"  {DIM}tune:{RESET} {v}")
            except Exception as e:
                print(f"  {RED}melody failed: {e}{RESET}")

        if use_phonemes:
            try:
                t1 = time.time()
                sounds = phonemes.score_words(wav, sentence.split())
                ph_ms = (time.time() - t1) * 1000
                print(f"  {DIM}sound ({ph_ms:.0f}ms):{RESET}")
                print("  " + phonemes.render(sounds).replace("\n", "\n  "))
            except Exception as e:
                print(f"  {RED}phoneme scoring failed: {e}{RESET}")

        suspects = suspect_payload(sentence, alignment)
        if use_jev and suspects:
            t1 = time.time()
            try:
                verdicts = jev.judge_sentence(sentence, transcript, suspects)
                jev_ms = (time.time() - t1) * 1000
                print(f"  {DIM}jev ({jev_ms:.0f}ms, model {verdicts.get('model')}):{RESET}")
                for w in suspects:
                    i = w["index"]
                    p = verdicts["read"].get(i)
                    cat = verdicts["category"].get(i, {})
                    ptxt = f"p(read)={p:.2f}" if p is not None else "n/a"
                    ctxt = cat.get("choice") or "-"
                    conf = cat.get("confidence")
                    conf_s = f" conf={conf:.2f}" if conf is not None else ""
                    print(f"    [{w['word']:>12}] {ptxt}  {ctxt}{conf_s}")
                pron = verdicts["pronunciation"]
                score = pron.get("score")
                if score is not None:
                    print(f"    pronunciation evidence: {score:.2f}/3 "
                          f"(transcript-shaped; acoustics need the phoneme layer)")
                usage = verdicts.get("usage") or {}
                if usage:
                    print(f"    tokens: {usage}")
            except Exception as e:  # charged endpoint — never crash the loop
                print(f"  {RED}jev call failed: {e}{RESET}")
        elif use_jev:
            print(f"  {GREEN}clean sentence — no Jev call needed{RESET}")

        if use_tts:
            t2 = time.time()
            models.speak(sentence)
            tts_ms = (time.time() - t2) * 1000
            print(f"  {DIM}read-back ({tts_ms:.0f}ms incl. first-load){RESET}")
    finally:
        if wav is not None:
            wav.unlink(missing_ok=True)


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return
    settings = load(sys.argv)
    passage = passage_arg(sys.argv)
    sentences = load_passage(passage)
    use_jev = settings.jev
    use_tts = settings.tts
    use_phonemes = settings.phonemes
    use_melody = settings.melody

    print(f"{BOLD}fluency coach{RESET} — {len(sentences)} sentences "
          f"tier {settings.tier} device {settings.device} "
          f"(jev {'on' if use_jev else 'off'}, tts {'on' if use_tts else 'off'}, "
          f"phonemes {'on' if use_phonemes else 'off'}, "
          f"melody {'on' if use_melody else 'off'}, "
          f"flow {'on' if settings.flow else 'off'})")
    print("  flags: --tier  --device  --no-jev  --no-tts  --no-phonemes  "
          "--no-melody  --no-flow  --host  --port   |  ctrl-c quits\n")

    for n, sentence in enumerate(sentences, 1):
        print(f"{BOLD}— sentence {n}/{len(sentences)} —{RESET}")
        try:
            run_sentence(sentence, use_jev, use_tts, use_phonemes, use_melody)
        except KeyboardInterrupt:
            print("\nbye")
            return
        except Exception as e:
            print(f"  {YELLOW}{e}{RESET}")
        print()

    print(f"{BOLD}passage complete.{RESET}")


if __name__ == "__main__":
    main()
