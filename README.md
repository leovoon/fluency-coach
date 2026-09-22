# fluency-coach

Read a passage aloud. See where the words, the tune, and the joins drifted.
The voice to copy is a recording you saved — yours, a teacher, a native speaker.
That file always wins.

![demo](demo.gif)

Kokoro is a model read, not a native speaker and not "the correct" pronunciation.
If a sentence has no saved reference, the app may play a model read and labels
it that way. It never calls that file your reference.

## Tiers

One config file, `coach.yaml`, plus CLI flags. Change tier, device, model id,
or engine there. Do not edit Python to upgrade.

| Tier | Who it is for | Preset |
|---|---|---|
| `lite` | Weak CPU, first install | ASR + diff only. Melody, flow, phonemes, Jev, and TTS are off. No wav2vec2 download on first paint. No voice model: record a reference to compare. |
| `standard` | Everyday practice | Melody, flow, reference compare, Kokoro read-back. Phonemes and Jev off. |
| `pro` | A machine that can hold another model | Standard, plus phoneme scoring. Jev is on in this preset — it is a charged call. Pass `--no-jev` unless you mean to spend a request. |

```sh
python -m coach.main --tier standard --device cpu
python -m coach.ui --tier pro --device mps --no-jev
```

`--tier` and `--device` override `coach.yaml`. Also: `--host`, `--port`,
`--no-melody`, `--no-flow`, `--no-phonemes`, `--no-tts`, `--no-jev`.
A token that starts with `--` is never treated as a passage path.

`coach.yaml` (see the commented example in the repo) sets the same knobs
without editing Python:

- `tier`: `lite` | `standard` | `pro`
- `device`: `auto` | `cpu` | `mps` | `cuda`
- `asr.engine` / `asr.id` — default `photon` and `moondream/parakeet-redux`
- `tts.engine`: `kokoro` | `system` | `none` (voice and lang sit next to it)
- `data_dir`, `host`, `port`

Defaults stay moondream/parakeet-redux and Kokoro. `tts.engine: none` or a
lite tier means no voice model. The UI will not synthesize or speak. Status:
record a reference to compare.

## Own voice wins

`record reference` saves a wav for that sentence (under the data dir, keyed by
the text). After that:

- Hear reference plays that wav.
- End-of-sentence read-back plays that wav.
- Comparison arrows come from that wav and are labeled "your saved reference".

A synthesized wav is labeled "model read (not a native speaker)". Passing a
path does not make it human.

When Kokoro is the engine, a voice picker sits in the UI header (female and
male, US and UK: `am_adam`, `bm_george`, …). Changing it applies to the next
model read without reloading the model. A voice close to your own makes the
melody comparison more useful. `coach.yaml` sets the default.

Flow pauses are about place, not length. Pause marks come from your saved
reference read — wherever it actually breathed — or, with no reference, from
punctuation and syntax rules (a labeled model read never forces its own
spacing onto you). Links are consonant→vowel junctions, confirmed tight by
the reference when one exists.

Questions are not all rising. Wh-questions typically fall; yes/no questions
may rise. The phrase tune is judged on the nucleus — the last content word,
or an ALL-CAPS word when the passage marks contrastive focus — by the slope
of the word's final stretch of voicing, compared to your reference when you
have one.

Stress is taught as beats, not numbers. The reference's prominent words
(energy relative to that recording's own middle) are the expected beats;
yours are compared on the same sentence. A word you flattened gets an amber
dotted underline; a small word you pushed gets a gray dot. No absolute dB,
ms, or Hz is ever shown.

## The practice loop

One dimension per pass, in evidence order: pauses, then stress, then melody.

1. **Listen.** Fresh sentence shows no marks — hear the model or reference
   read first. Marks appear after your attempt. `show guide marks` turns the
   pre-speech scaffold back on if you want it.
2. **Read.** Record the sentence.
3. **One focus.** The app picks the weakest layer of that attempt — missed
   pauses first, then flattened/pushed beats, else melody — and paints only
   that layer in color. The legend under the passage names it
   ("this pass: pauses").
4. **Replay chunks.** Buttons under the passage split the sentence at breath
   points; each plays your slice, then the reference slice. Clicking a word
   still plays that word or its link group.
5. **`diagnose`** (off by default) reveals per-word pitch arrows and the
   pitch chart — drill-down, not the first view.

Stress-timing compares how much you alternate long and short words with the
reference. Too even means the small words (the, a, of, to) were not squeezed.

## Machine notes

Apple Silicon: `device: auto` resolves to `mps` on arm64 Macs, and to `cpu`
everywhere else. `--device mps` forces the GPU. If MPS is missing or a model
rejects it, use `--device cpu`.

Kokoro (pykokoro) runs as local ONNX, on CPU. It is the default TTS engine,
not the learner's target accent.

Phoneme scoring downloads wav2vec2 only when `phonemes` is on (the pro tier).
lite and standard stay on a letter fallback for link marks, so opening the UI
does not pull that model.

Jev is a charged hosted call, used after a recording only: it judges whether
flagged words were actually read, classifies the error, and rates
pronunciation — one call per attempt, only when the diff flags suspects.
Leave it off unless you mean to spend a request. The UI switch is labeled
"jev (charged)". It is fully optional: without configuration the app labels
the switch's outcome and carries on.

Configure it in `coach.yaml` (this file is machine-local and gitignored —
safe for a key):

```yaml
jev:
  api: https://your-endpoint.example/v1/systemone   # optional — default ships in coach/jev.py
  model: jev-latest                                 # optional, this is the default
  api_key: my-key-here                              # or env JEV_API_KEY
```

Key resolution order: `JEV_API_KEY` env → `jev.api_key` in `coach.yaml` →
macOS keychain entry (`jev_api_key`).

## Run

```sh
cd ~/personal/fluency-coach
python -m coach.main                         # built-in passage
python -m coach.main essay.txt               # your passage
python -m coach.main --tier lite --device cpu
python -m coach.ui                           # browser UI, host/port from coach.yaml
```

UI defaults for melody, flow, phonemes, Jev, read-back, and reference compare
come from `coach.yaml`, not from hardcoded switches. Previous / next / load
stay disabled while a recording is in progress.

Don't want to paste your own text? The passage editor ships with template
passages (everyday life, commuting, cooking, keeping fit, Saturday shopping) —
pick one and load it in a click. Every mark in the passage has a hover tooltip
explaining what it asks of you.

Per sentence: listen (no marks) → record → words colorize (green match, red
substituted, gray skipped) → the weakest layer paints in focus (pauses, then
stress, then melody) → chunk replay buttons → optional Jev → read-back of
your saved reference if you have one, otherwise a labeled model read if TTS
is on.

Click a word to hear it. Linked words play as one chunk: your slice, then the
reference span. With no reference and TTS on, the chunk is a model read of
the phrase. With TTS off, record a reference to hear that chunk.

Mic and speakers are the server machine's (sounddevice). One user, localhost.

## Tuning

- `coach.yaml`: tier, device, `asr.id`, `tts.engine`, host, port. Feature flags too, or the `--no-*` CLI flags.
- `coach/record.py`: `SILENCE_RMS` — raise if capture cuts you off, lower if it never stops.
- Jev suspect cap and thresholds live with the judgment layer. Do not enable it for a dry run.
