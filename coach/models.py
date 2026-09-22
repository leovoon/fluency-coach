"""Local models: Photon ASR + swappable TTS, loaded lazily.

Shipped engines are moondream/parakeet-redux (Photon) and Kokoro. Ids, device,
and voice come from coach.yaml / CLI (coach.config), not from this file.
Own-voice recordings are the reference; this module only synthesizes a model
read when no recording exists. Nothing here is the learner.
"""

from __future__ import annotations

from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
import os
import platform
import shutil
import subprocess
import tempfile
import threading

from .config import DEFAULT_ASR_ID as ASR_MODEL_ID
from .config import load

# The Photon handle is a context manager; entering it via an ExitStack kept at
# module scope keeps it alive for reuse without leaking the with-block.
# Model access is serialized: lru_cache is not thread-safe against concurrent
# first loads, and the runtime crashes under concurrent use.
_exit_stack = ExitStack()
_asr_lock = threading.RLock()
_tts_lock = threading.RLock()
_device_lock = threading.Lock()
_forced_device: str | None = None

_TTS_OFF = "TTS off in this tier — record a reference or set tts.engine"


def _settings():
    return load()


def runtime_device() -> str:
    """Configured device, or cpu if a previous load already failed over."""
    if _forced_device:
        return _forced_device
    return _settings().device


# Kokoro voices that ship with the standard 82M release, by accent/gender.
# The UI picker is built from this; anything else a learner puts in
# coach.yaml is passed through untouched.
KOKORO_VOICES = {
    "female (US)": [
        "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore",
        "af_sarah", "af_sky", "af_nova",
    ],
    "male (US)": [
        "am_adam", "am_eric", "am_michael", "am_onyx", "am_echo",
        "am_fenrir", "am_liam", "am_puck",
    ],
    "female (UK)": [
        "bf_emma", "bf_isabella", "bf_alice", "bf_lily", "bf_vale",
    ],
    "male (UK)": [
        "bm_george", "bm_daniel", "bm_lewis", "bm_fable",
    ],
}
_ALL_KOKORO = {v for vs in KOKORO_VOICES.values() for v in vs}

# Runtime voice choice. Settings is frozen; the UI picker writes here.
# Changing it does not reload the model: voice is a per-run override.
_voice_override: str | None = None


def set_voice(voice: str) -> None:
    """Pick the Kokoro voice for subsequent speak/synthesize calls."""
    global _voice_override
    voice = str(voice).strip()
    if not voice:
        _voice_override = None
        return
    if voice not in _ALL_KOKORO:
        raise RuntimeError(
            f"unknown Kokoro voice {voice!r}; pick one of {', '.join(sorted(_ALL_KOKORO))}"
        )
    _voice_override = voice


def current_voice() -> str:
    return _voice_override or _settings().tts_voice


def _remember_cpu() -> None:
    global _forced_device
    with _device_lock:
        _forced_device = "cpu"


def get_asr():
    with _asr_lock:
        return _load_asr()


@lru_cache(maxsize=1)
def _load_asr():
    settings = _settings()
    if settings.asr_engine != "photon":
        raise RuntimeError(
            f"asr.engine {settings.asr_engine!r} is not available; "
            "shipped engine is photon"
        )
    import moondream as md

    device = runtime_device()
    print(
        f"  … loading parakeet ASR on {device} (first run downloads ~178MB)",
        flush=True,
    )
    try:
        return _exit_stack.enter_context(md.photon(settings.asr_id, device=device))
    except Exception as first:
        if device == "cpu":
            raise
        print(f"  … ASR device {device} failed ({first}); retrying on cpu", flush=True)
        _remember_cpu()
        try:
            return _exit_stack.enter_context(md.photon(settings.asr_id, device="cpu"))
        except Exception as second:
            raise RuntimeError(
                f"ASR failed on {device} and on cpu: {second}"
            ) from second


def transcribe(wav_path) -> str:
    return transcribe_detailed(wav_path)["text"]


def transcribe_detailed(wav_path) -> dict:
    """Transcribe an utterance.

    Returns {"text": str, "words": list | None}. "words" holds word-level
    timestamps ([{"word", "start", "end"}, ...]) when available, for
    pause/pacing scoring later; callers that only need text use transcribe().
    """
    speech = get_asr()
    if not Path(str(wav_path)).exists():
        raise RuntimeError(f"no audio at {wav_path}")
    with _asr_lock:
        result = speech.transcribe(audio=str(wav_path), timestamps="word")
    words = None
    for seg in result.get("segments") or []:
        if seg.get("words"):
            words = (words or []) + seg["words"]
    return {"text": result["text"].strip(), "words": words}


@lru_cache(maxsize=1)
def _load_tts():
    settings = _settings()
    if settings.tts_engine == "none":
        raise RuntimeError(_TTS_OFF)
    if settings.tts_engine != "kokoro":
        raise RuntimeError(
            f"tts.engine is {settings.tts_engine}; Kokoro is not loaded"
        )
    from pykokoro.pipeline import KokoroPipeline, PipelineConfig
    print("  … loading Kokoro TTS (first run downloads ~330MB)", flush=True)
    from pykokoro import GenerationConfig
    pipe = KokoroPipeline(PipelineConfig(
        voice=settings.tts_voice,
        generation=GenerationConfig(lang=settings.tts_lang),
    ))
    pipe.warmup()
    return pipe


def get_tts():
    with _tts_lock:
        return _load_tts()


def speak(text: str) -> None:
    settings = _settings()
    if settings.tts_engine == "kokoro":
        with _tts_lock:
            pipe = get_tts()
            result = pipe.run(text, lang=settings.tts_lang, voice=current_voice())
            result.play()
        return
    from . import record
    wav = synthesize(text)
    try:
        record.play_wav(wav)
    finally:
        wav.unlink(missing_ok=True)


def speak_word(word: str) -> None:
    """Say a single word (punctuation stripped) — for click-to-hear in the UI."""
    text = word.strip(".,!?;:\"“”‘’()[]").strip() or word
    speak(text)


def synthesize(text: str) -> Path:
    """Write a wav and return its path. No playback.

    Compatible with record.play_wav. Kokoro uses soundfile; macOS system TTS
    uses `say` (afconvert if the file is not a wav). tts.engine none raises.
    """
    settings = _settings()
    engine = settings.tts_engine
    if engine == "none":
        raise RuntimeError(_TTS_OFF)
    if engine == "system":
        with _tts_lock:
            return _synthesize_system(text, settings.tts_voice)
    if engine != "kokoro":
        raise RuntimeError(
            f"unknown tts.engine {engine!r}; use kokoro, system, or none"
        )
    return _synthesize_kokoro(text, settings.tts_lang)


def _synthesize_kokoro(text: str, lang: str) -> Path:
    import soundfile as sf
    with _tts_lock:
        pipe = get_tts()
        result = pipe.run(text, lang=lang, voice=current_voice())
        out = _temp_wav()
        try:
            sf.write(out, result.audio, result.sample_rate)
        except Exception:
            out.unlink(missing_ok=True)
            raise
        return out


def _synthesize_system(text: str, voice: str) -> Path:
    if platform.system() != "Darwin":
        raise RuntimeError("system TTS failed: `say` is only available on macOS")
    if not shutil.which("say"):
        raise RuntimeError("system TTS failed: `say` not found")
    out = _temp_wav()
    cmd = ["say", "-o", str(out), "--data-format=LEI16@22050"]
    # af_heart is a Kokoro voice id, not a macOS voice. Omit -v unless set.
    if voice and voice != "af_heart":
        cmd[1:1] = ["-v", voice]
    cmd.append(text)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"system TTS failed: {e}") from e
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        out.unlink(missing_ok=True)
        raise RuntimeError(
            "system TTS failed: " + (detail or f"say exited {proc.returncode}")
        )
    try:
        _ensure_wav(out)
    except Exception:
        out.unlink(missing_ok=True)
        raise
    return out


def _ensure_wav(path: Path) -> None:
    header = path.read_bytes()[:12]
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return
    if not shutil.which("afconvert"):
        raise RuntimeError(
            "system TTS failed: say did not write a wav and afconvert is missing"
        )
    converted = path.with_suffix(".conv.wav")
    proc = subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", str(path), str(converted)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not converted.exists() or converted.stat().st_size == 0:
        converted.unlink(missing_ok=True)
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "system TTS failed: say did not write a wav"
            + (f" ({detail})" if detail else " and afconvert failed")
        )
    os.replace(converted, path)


def _temp_wav() -> Path:
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return Path(name)
