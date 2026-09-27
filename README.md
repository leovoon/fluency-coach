# fluency-coach

Read a passage aloud. See where the words, the tune, and the joins drifted.
The voice to copy is a recording you saved — yours, a teacher, a native speaker.
That file always wins.

![demo](demo.gif)

## How it works

- **ASR** — `moondream/parakeet-redux` (Photon) listens to your recording and
  writes down exactly what you said.
- **Model read** — [PocketTTS](https://github.com/kyutai-labs/pocket-tts) via
  [FluidAudio](https://github.com/orukeet/FluidAudio) renders the sentence in a
  **cloned teaching voice** on the Apple Neural Engine. The model read is
  always labeled as such — it is not a native speaker and not "the correct"
  pronunciation.
- **Word clicks** — single words are spoken in the same cloned voice, with a
  small local Kokoro model as fallback.
- **Feedback** — a word-by-word diff drives the melody, flow, and stress marks
  on screen.

If a sentence has no saved reference, the app may play a model read and labels
it that way. It never calls that file your reference.

## Quickstart

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp coach.yaml.example coach.yaml   # defaults are fine; tts.engine: pocket
.venv/bin/python -m coach.ui       # http://127.0.0.1:8080
```

### Start / stop without a terminal

If the package is installed in the venv (`.venv/bin/pip install -e .`), a
`coach` command manages the UI server as a background process:

```sh
.venv/bin/coach start --tier pro   # any coach.ui flag passes through
.venv/bin/coach status
.venv/bin/coach logs -f            # tail the server log
.venv/bin/coach stop               # clean stop; the TTS worker dies with it
```

The pid lives at `~/.fluency-coach/server.pid` and the log at
`~/.fluency-coach/server.log` (both follow `FLUENCY_DATA`). Without the
install, `python -m coach.ctl …` works the same from the project root.

macOS (Apple Silicon) is the supported platform — PocketTTS runs on the
Apple Neural Engine through CoreML. First runs download models: ~178MB for
ASR, ~766MB for the TTS English pack, once each.

### Building the voice worker

PocketTTS runs in a small Swift worker (`fluid-poc/`), pinned to a known
FluidAudio commit:

```sh
cd fluid-poc && swift build -c release
```

The UI spawns and manages this worker automatically (`fluidpoc --worker`).
No HF token is required for the TTS models.

### ASR worker

ASR runs out-of-process: `coach.models` spawns `python -m coach.asr_worker`
on first transcribe and talks JSON lines over stdin/stdout. The worker exits
by itself after 10 minutes without a request (`COACH_ASR_IDLE_S` to change),
and the client respawns on the next read.

Two engines, chosen by `asr.engine` in `coach.yaml`:

- **`parakeet-cpp`** (default) — whisper.cpp's `parakeet-cli` (ggml, Metal,
  q4_0). Torch-free: ~420MB peak, ~0.15s compute, ~0.8s per read including
  process spawn. Word timestamps come from the TDT token dump. Install:
  build whisper.cpp (`cmake -B build -DGGML_METAL=ON && cmake --build
  build --target parakeet-cli`), copy `parakeet-cli` to
  `~/.fluency-coach/bin/` and the build's `*.dylib` to `~/.fluency-coach/lib/`
  (add rpath `@executable_path/../lib` via `install_name_tool`), and download
  `ggml-parakeet-tdt-0.6b-v3-q4_0.bin` from `ggml-org/parakeet-GGUF` into
  `~/.fluency-coach/models/`.
- **`photon`** — moondream Photon (kestrel/torch, MPS), ~650MB peak, ~1.2s
  per read. The reference implementation; also the only engine that runs the
  ternary-quantized parakeet-redux weights.

### Progress history

Every scored attempt appends one row to `~/.fluency-coach/history.jsonl` —
WER, flattened beats, missed breaths, choppy links, hesitations. A small
line under the mark legends compares this week's medians to last week's.
There are no ratings and no self-judgment: only counts the scoring pass
already computed. For a trend that cannot drift with your improving
reference recordings, re-read a fixed checkpoint passage weekly.

### Teaching voices

The **teaching voice** card records or uploads a 2–30s clip, names it, and
stores it in a voices gallery (`~/.fluency-coach/references/voices/`). One
clip is active; clicking another name switches the clone instantly. Only the
first 10 seconds of a clip shape the clone — the voice encoder uses a fixed
10-second window.

### Browser extension

`coach/extension/` is an unpacked MV3 extension that grabs a passage from any
page (including per-post on X.com) and opens a practice session in the coach:

```sh
# chrome://extensions → Developer mode → Load unpacked → coach/extension/
```

The extension is thin: it only POSTs text to the local server and opens the
resulting session URL. Nothing leaves your machine.

## Tiers

One config file, `coach.yaml`, plus CLI flags. Change tier, device, model id,
or engine there. Do not edit Python to upgrade.

| Tier | Who it is for | Preset |
|---|---|---|
| `lite` | Weak CPU, first install | ASR + diff only. Melody, flow, phonemes, Jev, and TTS are off. No wav2vec2 download on first paint. No voice model: record a reference to compare. |
| `standard` | Everyday practice | Melody, flow, reference compare, model read-back. Phonemes and Jev off. |
| `pro` | A machine that can hold another model | Standard, plus phoneme scoring. Jev is on in this preset — it is a charged call. Pass `--no-jev` unless you mean to spend a request. |

```sh
python -m coach.main --tier standard --device cpu
python -m coach.ui --tier pro --device mps --no-jev
```

`--tier` and `--device` override `coach.yaml`. Also: `--host`, `--port`,
`--no-melody`, `--no-flow`, `--no-phonemes`, `--no-tts`, `--no-jev`.

### Engines

`tts.engine` in `coach.yaml`:

- **`pocket`** — PocketTTS via the FluidAudio worker. Clones the teaching
  voice; sentences and words both. The default and the only maintained path.
- **`kokoro`** — small local model, no cloning. Kept as the fallback for
  word reads and for machines without an ANE.
- **`system`**, **`none`** — say nothing, or let the OS speak.

Unimplemented ids (`chatterbox`) are accepted by config and fail with a
friendly error at use.

## Data

Everything lives in `~/.fluency-coach/` (override with `data_dir` or
`FLUENCY_DATA`):

- `references/` — your saved reference recordings and the active
  `model-voice.wav`
- `references/voices/` — the voices gallery (`.active` marks the current one)
- `tts-cache/` — rendered sentences, keyed by engine, text, and reference
  identity; deleting the folder only costs re-render time

## Project layout

```
coach/            NiceGUI app, ASR/TTS bridges, scoring (melody, flow, stress)
coach/asr_worker.py  ASR subprocess: torch-free parakeet-cpp / photon engines
coach/extension/  unpacked Chrome extension (raw JS, no build step)
fluid-poc/        Swift worker: PocketTTS cloning on the ANE
coach.yaml.example
```

## History

The original engine setup (Kokoro-only, before voice cloning) is preserved on
the `archive/kokoro-era` branch. A NeuTTS Air experiment was removed after
PocketTTS won a blind A/B (3/3) with a ~5s warm render versus ~25s cold load.
