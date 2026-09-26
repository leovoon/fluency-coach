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
    # Loud reads clip near full scale, and kestrel's resampler overshoots
    # >1.0 on those peaks — sanitize every file at the ASR door.
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(wav_path), dtype="float32")
    peak = float(np.abs(data).max()) if data.size else 0.0
    if peak > 0.95:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp, data / peak * 0.95, sr, subtype="PCM_16")
            wav_path = tmp
        except Exception:
            Path(tmp).unlink(missing_ok=True)
    with _asr_lock:
        result = speech.transcribe(audio=str(wav_path), timestamps="word")
        if peak > 0.95:
            Path(str(wav_path)).unlink(missing_ok=True)
    words = None
    for seg in result.get("segments") or []:
        if seg.get("words"):
            words = (words or []) + seg["words"]
    return {"text": result["text"].strip(), "words": words}


@lru_cache(maxsize=1)
def _kokoro_pipe():
    """Kokoro pipeline — 82M, Apache, CPU via ONNX. Loads on demand even when
    the cloning engine needs a fast word reader."""
    settings = _settings()
    from pykokoro.pipeline import KokoroPipeline, PipelineConfig
    from pykokoro import GenerationConfig
    print("  … loading Kokoro TTS for word reads (first run downloads ~330MB)", flush=True)
    pipe = KokoroPipeline(PipelineConfig(
        voice=settings.tts_voice,
        generation=GenerationConfig(lang=settings.tts_lang),
    ))
    pipe.warmup()
    return pipe


@lru_cache(maxsize=1)
def _load_tts():
    """Kokoro — the default. 82M, Apache, CPU via ONNX."""
    settings = _settings()
    if settings.tts_engine == "none":
        raise RuntimeError(_TTS_OFF)
    if settings.tts_engine != "kokoro":
        raise RuntimeError(
            f"tts.engine is {settings.tts_engine}; Kokoro is not loaded"
        )
    return _kokoro_pipe()


@lru_cache(maxsize=1)
def _load_chatterbox():
    """Chatterbox Nano — 110M, paralinguistic tags, CPU-first (3x realtime on 8 cores)."""
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError as e:
        raise RuntimeError(
            "chatterbox engine needs: pip install chatterbox-tts"
        ) from e
    print("  … loading Chatterbox Nano (first run downloads ~800MB)", flush=True)
    # Nano targets CPU/on-device; mps for the LLM backbone is not proven.
    return ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)


# Live stage text for long TTS operations (model load, sentence render).
# The UI polls this while it waits, replacing the silent spinner.
TTS_LOAD = {"stage": ""}


def tts_load_stage() -> str:
    return TTS_LOAD["stage"]



def _teaching_reference() -> tuple[Path, str]:
    """Resolve the teaching voice: coach.yaml override, else the one saved
    from the UI (data_dir/references/model-voice.wav + .txt)."""
    settings = _settings()
    if settings.tts_ref_wav:
        ref_wav = Path(os.path.expanduser(settings.tts_ref_wav))
        ref_text = settings.tts_ref_text
        txt = ref_wav.with_suffix(".txt")
        if not ref_text and txt.exists():
            ref_text = txt.read_text().strip()
        return ref_wav, ref_text
    from .reference import references_dir
    ref_wav = references_dir() / "model-voice.wav"
    if not ref_wav.exists():
        raise RuntimeError(
            "no teaching voice yet — record or upload one in the UI "
            "(teaching voice card), or set tts.ref_wav / tts.ref_text in coach.yaml"
        )
    txt = ref_wav.with_suffix(".txt")
    ref_text = txt.read_text().strip() if txt.exists() else ""
    return ref_wav, ref_text


def model_voice_path() -> Path | None:
    """Where the teaching voice lives, if one exists."""
    try:
        return _teaching_reference()[0]
    except RuntimeError:
        return None


def voices_dir() -> Path:
    from .reference import references_dir
    d = references_dir() / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _active_voice_marker() -> Path:
    return voices_dir() / ".active"


def _activate_voice_file(src: Path, name: str) -> None:
    """Copy a gallery clip over the active voice file and restart workers."""
    import shutil
    from .reference import references_dir
    shutil.copyfile(src, references_dir() / "model-voice.wav")
    _active_voice_marker().write_text(name)
    _kill_fluid_worker()


def save_voice(name: str, wav_path) -> dict:
    """Convert a clip, store it in the gallery, make it the active voice."""
    import soundfile as sf
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr < 16000:
        raise RuntimeError("voice clip sample rate too low (<16kHz)")
    duration = len(data) / sr
    if duration > 30.0:
        data = data[: int(30 * sr)]
        duration = 30.0
    if duration < 2.0:
        raise RuntimeError("voice clip too short — need at least 2 seconds")
    name = ("".join(c for c in name.strip() if c not in '\\/:*?"<>|').strip()
            or "voice")
    d = voices_dir()
    dst = d / f"{name}.wav"
    fd, tmp = tempfile.mkstemp(prefix=".v-", suffix=".wav", dir=d)
    os.close(fd)
    tmp = Path(tmp)
    try:
        sf.write(str(tmp), data, sr, subtype="PCM_16")
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    _activate_voice_file(dst, name)
    return {"name": name, "duration": duration}


def activate_voice(name: str) -> dict:
    """Make a gallery voice the active teaching voice."""
    import soundfile as sf
    src = voices_dir() / f"{name}.wav"
    if not src.exists():
        raise RuntimeError(f"voice {name!r} not found")
    _activate_voice_file(src, name)
    return {"name": name, "duration": sf.info(str(src)).duration}


def delete_voice(name: str) -> None:
    (voices_dir() / f"{name}.wav").unlink(missing_ok=True)


def list_voices() -> list[dict]:
    """All gallery voices with the active marker applied."""
    import soundfile as sf
    marker = _active_voice_marker()
    active = marker.read_text().strip() if marker.exists() else ""
    out = []
    for f in sorted(voices_dir().glob("*.wav")):
        try:
            out.append({"name": f.stem, "path": str(f),
                        "duration": sf.info(str(f)).duration,
                        "active": f.stem == active})
        except Exception:
            continue
    return out

    _kill_fluid_worker()
    return {"path": dst, "trimmed": trimmed, "duration": duration,
            "transcript": bool(text.strip())}


def _resolve_tts_engine() -> str:
    return _settings().tts_engine


def get_tts():
    with _tts_lock:
        engine = _resolve_tts_engine()
        if engine == "kokoro":
            return _load_tts()
        if engine == "chatterbox":
            return _load_chatterbox()
        raise RuntimeError(_TTS_OFF)


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


def model_read_label() -> str:
    """What the UI calls a synthesized read. Own voice still wins; a cloned
    read is in the reference speaker's voice and must say so."""
    if _settings().tts_engine == "pocket":
        return "model read (cloned from your reference voice)"
    return "model read (not a native speaker)"


def speak_word(word: str) -> None:
    """Say a single word (punctuation stripped) — click-to-hear in the UI.

    Prefers PocketTTS (the same teaching voice as sentence reads, ~0.5s
    warm); falls back to kokoro (different voice, always fast) when the
    fluid worker is unavailable."""
    text = word.strip(".,!?;:\"“”‘’()[]").strip() or word
    settings = _settings()
    if settings.tts_engine == "kokoro":
        speak(text)
        return
    if settings.tts_engine == "none":
        raise RuntimeError(_TTS_OFF)
    fluid = _synthesize_fluid(text)
    if fluid is not None:
        from . import record
        try:
            record.play_wav(fluid)
        finally:
            fluid.unlink(missing_ok=True)
        return
    with _tts_lock:
        pipe = _kokoro_pipe()
        result = pipe.run(text, lang=settings.tts_lang, voice=current_voice())
        result.play()


def _tts_cache_path(engine: str, text: str) -> Path | None:
    """Stable cache path for a synthesized read; None disables caching.

    Model reads are expensive (~1x realtime per sentence) and the UI
    re-requests them on every sentence view, so cache by engine + text (+
    reference identity for cloned voices). Callers get a copy — they delete
    what we return."""
    if engine not in ("kokoro", "pocket", "chatterbox"):
        return None
    import hashlib
    parts = [engine, text]
    if engine == "pocket":
        try:
            ref, _ = _teaching_reference()
            st = ref.stat()
            parts.append(f"{ref}:{st.st_size}:{int(st.st_mtime)}")
        except Exception:
            return None  # no reference yet; nothing to key on
    h = hashlib.sha1("\x00".join(parts).encode("utf-8", "ignore")).hexdigest()
    from .reference import references_dir
    cache = references_dir().parent / "tts-cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{h}.wav"


def synthesize(text: str) -> Path:
    """Write a wav and return its path. No playback.

    Compatible with record.play_wav. Kokoro uses soundfile; macOS system TTS
    uses `say` (afconvert if the file is not a wav). tts.engine none raises.
    """
    settings = _settings()
    engine = settings.tts_engine
    if engine == "none":
        raise RuntimeError(_TTS_OFF)
    cache = _tts_cache_path(engine, text)
    if cache is not None and cache.exists():
        out = _temp_wav()
        shutil.copyfile(cache, out)
        return out
    if engine == "system":
        with _tts_lock:
            return _synthesize_system(text, settings.tts_voice)
    if engine == "kokoro":
        out = _synthesize_kokoro(text, settings.tts_lang)
    elif engine == "chatterbox":
        out = _synthesize_chatterbox(text)
    elif engine == "pocket":
        fluid = _synthesize_fluid(text)
        if fluid is None:
            raise RuntimeError(
                "PocketTTS render failed — restart the app, and check the "
                "voice card has a reference clip")
        out = fluid
    else:
        raise RuntimeError(
            f"unknown tts.engine {engine!r}; use kokoro, pocket, chatterbox, system, or none"
        )
    if cache is not None:
        try:
            shutil.copyfile(out, cache)
        except Exception:
            pass  # cache is an optimization; never block synthesis on it
    return out


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


def _synthesize_chatterbox(text: str) -> Path:
    import soundfile as sf
    import torch
    with _tts_lock:
        model = get_tts()
        out = _temp_wav()
        try:
            wav = model.generate(text)
            audio = wav.squeeze().detach().cpu().numpy()
            sf.write(out, audio, model.sr)
        except Exception:
            out.unlink(missing_ok=True)
            raise
        return out


# ── Rust render worker (adapted from shadow-companion/tts-worker-rs) ──
# Same Q8 GGUF backbone + ONNX codec as the Python path, but cold start is
# ~5s instead of ~25s and torch stays out of the render loop. Renders to
# wav files over a stdin/stdout protocol; falls back to the Python path
# whenever the binary or the worker is unavailable.
# ── PocketTTS render worker (FluidAudio, ANE) ──
# Won the blind A/B 3/3 against NeuTTS on voice similarity and renders in
# ~0.5s warm with a ~1s model load — no persistent model residency penalty.
# Sentence reads AND word clicks prefer it; NeuTTS (rust → python) stays as
# the fallback chain.
_FLUID_WORKER: dict = {"proc": None, "ref": None}


def _fluid_worker_path() -> Path | None:
    p = Path(__file__).resolve().parent.parent / "fluid-poc" / ".build" / "release" / "fluidpoc"
    return p if p.exists() else None


def _fluid_worker():
    """Persistent PocketTTS worker bound to the current teaching voice."""
    proc = _FLUID_WORKER.get("proc")
    ref, _ = _teaching_reference()
    if (proc is not None and proc.poll() is None
            and _FLUID_WORKER.get("ref") == str(ref)):
        return proc
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass
    binary = _fluid_worker_path()
    if binary is None or not ref.exists():
        return None
    import subprocess
    try:
        proc = subprocess.Popen(
            [str(binary), "--worker", str(ref)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        # CoreML may inject noise lines on stdout; wait for the real marker
        for _ in range(50):
            line = proc.stdout.readline()
            if not line:
                proc.kill()
                return None
            if "WORKER_READY" in line:
                break
        else:
            proc.kill()
            return None
    except Exception:
        return None
    _FLUID_WORKER["proc"] = proc
    _FLUID_WORKER["ref"] = str(ref)
    return proc


def _kill_fluid_worker() -> None:
    proc = _FLUID_WORKER.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    _FLUID_WORKER["proc"] = None


def _synthesize_fluid(text: str) -> Path | None:
    """PocketTTS render of the teaching voice; None on any failure."""
    proc = _fluid_worker()
    if proc is None:
        return None
    cache = _tts_cache_path("pocket", text)
    out = cache if cache is not None else _temp_wav()
    TTS_LOAD["stage"] = "rendering sentence in your voice…"
    try:
        import json as _json
        proc.stdin.write(_json.dumps({"text": text, "out": str(out)}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if line.strip() == "OK" and Path(out).exists():
            result = _temp_wav()
            shutil.copyfile(out, result)
            return result
    except Exception:
        pass
    finally:
        TTS_LOAD["stage"] = ""
    _kill_fluid_worker()  # broken/hung worker → restart on next call
    return None




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
