"""Microphone capture: record from the default mic until trailing silence."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME = SAMPLE_RATE * FRAME_MS // 1000
SILENCE_RMS = 350.0      # int16 RMS below this = silence (tune on your room)
TRAILING_SILENCE_S = 1.2
MAX_DURATION_S = 30.0

# One lock for capture and playback so two clicks cannot open two streams.
_AUDIO = threading.Lock()


def play_wav(path) -> None:
    """Play a wav file through the default output (blocking until done)."""
    with _AUDIO:
        try:
            _play_wav(path)
        except sd.PortAudioError as e:
            raise RuntimeError(f"speaker error: {e}") from e


def play_wav_slice(path, start: float, end: float, pad: float = 0.05) -> None:
    """Play only [start, end] seconds of a wav (click-to-hear a single word)."""
    with _AUDIO:
        try:
            _play_wav_slice(path, start, end, pad)
        except sd.PortAudioError as e:
            raise RuntimeError(f"speaker error: {e}") from e


def record_utterance() -> Path:
    """Block until the user speaks and stops. Returns the wav path."""
    with _AUDIO:
        try:
            return _record_utterance()
        except sd.PortAudioError as e:
            raise RuntimeError(f"mic error: {e}") from e


def _play_wav(path) -> None:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    sd.play(data, sr)
    sd.wait()


def _play_wav_slice(path, start: float, end: float, pad: float) -> None:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    a = max(0, int((start - pad) * sr))
    b = min(len(data), int((end + pad) * sr))
    if b <= a:
        return
    sd.play(data[a:b], sr)
    sd.wait()


def _record_utterance() -> Path:
    frames: list[np.ndarray] = []
    spoke = False
    silent_frames = 0
    max_frames = int(MAX_DURATION_S * SAMPLE_RATE / FRAME)
    collected = 0

    print("  🎙  listening… (speak now)", flush=True)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=FRAME) as stream:
        while collected < max_frames:
            data, _ = stream.read(FRAME)
            mono = data[:, 0]
            frames.append(mono.copy())
            collected += 1
            rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
            if rms > SILENCE_RMS:
                spoke = True
                silent_frames = 0
            elif spoke:
                silent_frames += 1
                if silent_frames * FRAME_MS >= TRAILING_SILENCE_S * 1000:
                    break
            time.sleep(0)

    if not spoke:
        raise RuntimeError("No speech detected (check mic / SILENCE_RMS).")

    audio = np.concatenate(frames)
    fd, name = tempfile.mkstemp(suffix=".wav", prefix="reading_")
    os.close(fd)
    out = Path(name)
    try:
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
    except Exception:
        out.unlink(missing_ok=True)
        raise
    return out
