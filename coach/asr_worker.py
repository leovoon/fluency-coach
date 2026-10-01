"""ASR worker — the inference runtime lives here, not in the UI.

The UI/terminal process talks to this module over stdin/stdout JSON lines so
heavy runtimes never load into the long-lived server process. The worker
exits after COACH_ASR_IDLE_S (default 600s) without a request, so the stack
is resident only while practicing. The client (coach.models) respawns on
demand.

engines (asr.engine in coach.yaml):
  parakeet-cpp  whisper.cpp's parakeet-cli (ggml, Metal, q4_0). torch-free:
    ~420MB peak, ~0.15s warm compute, ~0.6s process spawn per read.
    Word timestamps come from the TDT token dump. Binary at
    ~/.fluency-coach/bin/parakeet-cli (or asr.binary), model at
    ~/.fluency-coach/models/ (or asr.model).
  photon        moondream Photon (kestrel/torch, MPS). ~650MB peak, ~1.2s
    per read; the fallback reference implementation.
  phonon        Fermion Phonon-2 (fermion-research, MLX). 164MB download,
    ~0.3s decode for a sentence read, text only: the fermion runtime does
    not expose token or word timings, so "words" is None and the UI's
    melody/flow features skip (the diff still works off the text). Model
    comes from asr.id (default FermionResearch/Phonon-2); first use
    downloads and unpacks it via the fermion cache.

protocol (one JSON object per line):
  request : {"wav": "/abs/path.wav"}
  response: {"ok": true, "text": "...", "words": [...]} or
            {"ok": false, "error": "..."}
startup : prints ASR_WORKER_READY once the engine is ready. Other stdout
          lines (download progress, CoreML noise) are ignored by the client;
          this module logs to stderr, which the parent inherits.

run: python -m coach.asr_worker   (spawned by coach.models; not for humans)
"""

from __future__ import annotations

import json
import os
import select
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

IDLE_S = float(os.environ.get("COACH_ASR_IDLE_S", "600"))
READY_MARKER = "ASR_WORKER_READY"

_exit_stack = ExitStack()


import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

IDLE_S = float(os.environ.get("COACH_ASR_IDLE_S", "600"))
READY_MARKER = "ASR_WORKER_READY"

# parakeet-cli token dump: "[ 7] id= 7877 frame= 21 ... t0= 168 t1= 168
# word_start=false ","" — t0/t1 are in 10ms units.
_TOKEN_RE = re.compile(
    r"id=\s*\d+.*?t0=\s*(\d+)\s+t1=\s*(\d+)\s+"
    r"word_start=(true|false)\s+\"(.*)\"\s*$"
)

_exit_stack = ExitStack()
_cpp: dict = {"binary": None, "model": None}
_phonon: list = []  # [loaded SpeechModel]
_pitch_on: bool = False  # tier wants melody; pitch then rides the response


def _load_phonon(settings):
    """Load the Fermion speech model named by asr.id (default Phonon-2).

    The pip package ships its own MLX runtime and model cache; a bare
    `pip install fermion-research` is enough only on paper — the Apple
    silicon engine also needs mlx/mlx-audio/mlx-lm/zstandard.
    """
    if _phonon:
        return _phonon[0]
    try:
        from fermion._catalog import entry_for, resolve_alias, SPEECH_PROFILES
        from fermion._speech import backends, fetch
    except ImportError as e:
        raise RuntimeError(
            "phonon engine needs the fermion package: "
            "pip install fermion-research mlx mlx-audio mlx-lm zstandard"
        ) from e

    def _resolve(spec):
        try:
            repo = resolve_alias(spec)
            entry = entry_for(repo)
            if entry is not None and entry.get("kind") == "speech":
                return repo, entry
        except Exception:
            pass
        return None, None

    # asr.id still holds the parakeet/photon id by default — resolve_alias
    # passes unknown ids through, so fall back to Phonon-2 rather than
    # erroring on a stale config line.
    repo, entry = _resolve(settings.asr_id)
    spec = settings.asr_id
    if entry is None:
        spec = "FermionResearch/Phonon-2"
        repo, entry = _resolve(spec)
    if entry is None:
        raise RuntimeError(
            f"asr.id {settings.asr_id!r} is not a Fermion speech model — "
            "phonon engine wants e.g. FermionResearch/Phonon-2"
        )
    key = entry["profile"]
    pin = SPEECH_PROFILES[key]
    print(
        f"  … loading Phonon ASR ({spec}) via MLX "
        "(first run downloads ~164MB)",
        file=sys.stderr, flush=True,
    )
    engine_kind = backends.resolve("coach ASR")
    model_dir = fetch.ensure(repo, key, pin)
    speech = backends.load(engine_kind, model_dir, profile=key,
                           backend=pin["backend"], quiet=True)
    _phonon.append(speech)
    # Pay the Metal shader compile on a throwaway decode so the first real
    # read doesn't stall ~20s (the fermion live path does the same).
    import numpy as np
    try:
        speech.transcribe_array_detailed(np.zeros(16000, dtype=np.float32))
    except Exception as e:
        print(f"  … phonon warmup skipped ({e})", file=sys.stderr, flush=True)
    return speech


def _load():
    """Ready the configured engine; returns a tag the caller dispatches on."""
    global _pitch_on
    from .config import load

    settings = load()
    _pitch_on = bool(settings.melody)
    if settings.asr_engine == "parakeet-cpp":
        _cpp_binary(settings)  # fail fast, before the ready marker
        _cpp_model(settings)
        return "parakeet-cpp"
    if settings.asr_engine == "phonon":
        _load_phonon(settings)  # fail fast, before the ready marker
        return "phonon"
    if settings.asr_engine != "photon":
        raise RuntimeError(
            f"asr.engine {settings.asr_engine!r} is not available; "
            "shipped engines are parakeet-cpp, phonon, and photon"
        )
    import moondream as md

    device = settings.device
    print(
        f"  … loading parakeet ASR on {device} (first run downloads ~178MB)",
        file=sys.stderr, flush=True,
    )
    try:
        return _exit_stack.enter_context(md.photon(settings.asr_id, device=device))
    except Exception as first:
        if device == "cpu":
            raise
        print(f"  … ASR device {device} failed ({first}); retrying on cpu",
              file=sys.stderr, flush=True)
        try:
            return _exit_stack.enter_context(md.photon(settings.asr_id, device="cpu"))
        except Exception as second:
            raise RuntimeError(
                f"ASR failed on {device} and on cpu: {second}"
            ) from second


def _cpp_binary(settings) -> Path:
    if _cpp["binary"]:
        return _cpp["binary"]
    from .config import default_data_dir

    candidates = []
    if settings.asr_binary:
        candidates.append(Path(os.path.expanduser(settings.asr_binary)))
    candidates.append(default_data_dir() / "bin" / "parakeet-cli")
    for c in candidates:
        if c.exists():
            _cpp["binary"] = c
            return c
    found = shutil.which("parakeet-cli")
    if not found:
        raise RuntimeError(
            "parakeet-cpp engine: parakeet-cli not found — set asr.binary in "
            "coach.yaml or install the binary at "
            f"{default_data_dir() / 'bin' / 'parakeet-cli'}"
        )
    _cpp["binary"] = Path(found)
    return _cpp["binary"]


def _cpp_model(settings) -> Path:
    if _cpp["model"]:
        return _cpp["model"]
    from .config import default_data_dir

    if settings.asr_model:
        p = Path(os.path.expanduser(settings.asr_model))
    else:
        p = default_data_dir() / "models" / "parakeet-tdt-0.6b-v3-q4_0.bin"
    if not p.exists():
        raise RuntimeError(
            f"parakeet-cpp engine: model not found at {p} — download "
            "ggml-parakeet-tdt-0.6b-v3-q4_0.bin from ggml-org/parakeet-GGUF "
            "or set asr.model in coach.yaml"
        )
    _cpp["model"] = p
    return p


def _transcribe_cpp(settings, wav_path: str) -> dict:
    """One parakeet-cli run: stdout is the transcript, stderr the token dump."""
    proc = subprocess.run(
        [str(_cpp_binary(settings)), "-m", str(_cpp_model(settings)),
         "-f", wav_path, "-ps"],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(
            "parakeet-cli failed: " + (detail[-1] if detail else f"exit {proc.returncode}")
        )
    words = []
    for line in proc.stderr.splitlines():
        m = _TOKEN_RE.search(line)
        if not m:
            continue
        t0, t1, word_start, text = int(m.group(1)), int(m.group(2)), m.group(3) == "true", m.group(4)
        tok = text.replace("\u2581", "")  # ▁ marks a word start, not a space
        if word_start or not words:
            words.append({"word": tok, "start": t0 / 100.0, "end": t1 / 100.0})
        else:
            words[-1]["word"] += tok
            words[-1]["end"] = t1 / 100.0
    return {"text": " ".join(proc.stdout.split()),
            "words": words or None}


def _transcribe_photon(speech, wav_path: str) -> dict:
    # Loud reads clip near full scale, and kestrel's resampler overshoots
    # >1.0 on those peaks — sanitize every file at the ASR door.
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32")
    peak = float(np.abs(data).max()) if data.size else 0.0
    tmp = None
    if peak > 0.95:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp, data / peak * 0.95, sr, subtype="PCM_16")
        wav_path = tmp
    try:
        result = speech.transcribe(audio=str(wav_path), timestamps="word")
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)
    words = None
    for seg in result.get("segments") or []:
        if seg.get("words"):
            words = (words or []) + seg["words"]
    return {"text": result["text"].strip(), "words": words}


def _transcribe_phonon(speech, wav_path: str) -> dict:
    """One fermion decode. Text only — no word timestamps exist on this
    path (the runtime returns text and segment windows, not token times)."""
    result = speech.transcribe_detailed(wav_path)
    text, _decode_s, _dur_s = result.triple()
    if getattr(result, "truncated", False):
        print("  … phonon: decode hit its token budget; part of the audio "
              "may be missing from the transcript", file=sys.stderr, flush=True)
    return {"text": text, "words": None}


def _transcribe(engine, wav_path: str) -> dict:
    if engine == "parakeet-cpp":
        from .config import load
        return _transcribe_cpp(load(), wav_path)
    if engine == "phonon":
        return _transcribe_phonon(_phonon[0], wav_path)
    return _transcribe_photon(engine, wav_path)


def _with_pitch(out: dict, wav_path: str) -> dict:
    """Attach word pitch to a transcription, when the tier wants melody.

    pyin pulls numba/scipy (~175MB, JIT on first call) — doing it here keeps
    that residency in this short-lived worker instead of the UI process.
    Pitch is an add-on: any failure leaves the transcription intact and the
    client falls back to scoring in its own process.
    """
    if not _pitch_on or not out.get("words"):
        return out
    try:
        from .melody import word_pitch
        out["pitch"] = word_pitch(wav_path, out["words"])
    except Exception as e:
        print(f"  … pitch scoring failed in worker: {e}", file=sys.stderr, flush=True)
    return out


def _next_request() -> str | None:
    """Next request line; None after IDLE_S idle or EOF (parent is gone)."""
    ready, _, _ = select.select([sys.stdin], [], [], IDLE_S)
    if not ready:
        return None
    line = sys.stdin.readline()
    return line or None


def main() -> int:
    speech = _load()
    print(READY_MARKER, flush=True)
    while True:
        line = _next_request()
        if line is None:
            return 0  # idle or EOF: exit; the client respawns on demand
        try:
            req = json.loads(line)
            wav = str(req["wav"])
            out = _with_pitch(_transcribe(speech, wav), wav)
            print(json.dumps({"ok": True, **out}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
