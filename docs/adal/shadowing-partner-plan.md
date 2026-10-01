# Shadowing partner redesign

Date: 2026-09-30
POC: AdaL
Status: Implemented and verified

## Verification results

- 18 automated tests passed, including NiceGUI record, text comparison, A/B playback, reset, empty passage handling, and microphone error recovery.
- Real Phonon-2 inference succeeded against an existing local WAV and returned text with no word timestamps.
- Real PocketTTS synthesis produced a 2.32-second model read; pitch extraction returned a measured contour.
- The browser play control completed real model playback, filled all eight words, and restored enabled controls.
- Paper and candlelit layouts were inspected at desktop and mobile sizes. Mobile had no horizontal overflow. Theme contrast and short-desktop spacing were corrected after inspection.
- Python compilation and git whitespace checks passed. The visual static check reported only Fraunces warnings, which are intentional because the user specified that font.
- Independent discovery/review returned no usable findings; the implementation was reviewed directly instead.
- Live learner microphone capture was not verified with a spoken user take. Recording and A/B sequencing are covered by controlled-audio UI tests.
- Karaoke timing remains approximate. Fonts load from Google Fonts, with local fallbacks.


## TL;DR

Keep the local NiceGUI app and its teaching-voice pipeline. Replace the main practice screen with the requested warm paper and candlelit designs. Use Phonon-2 for text transcription, disable arrow/flow feedback, and compare complete model/reference and learner recordings with A/B playback.

## Design

A centered column, roughly 720px wide, sits directly on #FAF7F2 without a enclosing card. Ink is #302C27, teacher ochre #A47832, and learner slate-teal #587575. Fraunces at approximately 48px carries the model sentence, scaling down on small screens.

Twelve small dots sit at the top in the design example, with the third ringed. Live progress must reflect the actual passage, rather than falsely showing sentence three or twelve sentences.

The upper row reads “MODEL READ · CLONED FROM YOUR TEACHER” when a cloned voice is actually used. Saved human references and fallback voices receive truthful labels. A soft ochre contour sits above the sentence. Spoken words turn ochre; upcoming words stay ink.

A thin waveform and one circular record/stop control separate the two readings. The learner row uses smaller slate-teal text. In the example, “We could take the quiet road home tonight.” becomes “We can take the road home.” “quiet” and “tonight” remain ghosted at their expected positions; “can” replaces “could.” These are sample differences, never fabricated live results.

Only “play”, “record”, and “again” appear in the bottom action row. Play listens to the model/reference followed by the latest take when available. Again resets the current attempt without advancing. The model and learner labels can provide individual playback without adding another toolbar. Passage and voice setup remain accessible outside the main practice area.

Candlelit uses a warm charcoal ground (#211E1A), parchment text (#E9DFCF), lighter ochre (#D2AB67), and desaturated teal (#9BB4AE). Focus indicators, record state, and error messages remain visible. Reduced-motion preferences stop idle breathing animation.

## Changes proposed

- `coach/ui.py`: replace the main composition and feedback presentation; add truthful voice labels, theme control, progress navigation, record state, and whole-sentence A/B playback. Preserve passage loading, extension entry points, saved references, and voice management.
- New presentation module/assets under `coach/`: isolate theme styles and sentence rendering from the existing large UI module. Decide the exact split after approval and a complete read of the affected callbacks.
- `coach/config.py` and `coach.yaml.example`: make Phonon-2 the intended default and turn off arrow/flow-dependent processing for this practice mode. Preserve explicit alternative-engine configuration where possible.
- `coach/asr_worker.py`: preserve and verify the existing uncommitted Phonon implementation. Change it only where compatibility or tests show a problem.
- `pyproject.toml`, `requirements.txt`, and `README.md`: reconcile installation instructions and dependency declarations with the working Phonon runtime, preserving existing edits.
- New tests: cover text-only ASR responses, skipped/substituted/inserted words, empty input, default configuration, truthful source labels, and playback sequencing/cancellation.
- Design documentation: record confirmed product constraints and final visual tokens.

## Timing and audio constraints

The existing Phonon adapter returns text with no word timestamps. It is enough for word comparison and whole-sentence A/B listening, but cannot provide precise word karaoke or word-slice playback.

For the initial implementation, propose a duration-based model playback highlight, explicitly treated as approximate. Exact karaoke would require a separate alignment stage. A real pitch contour can be extracted from the model audio independently of ASR word timing; missing pitch should produce a neutral/absent contour, not invented measurements.

The current microphone and speaker belong to the server machine. Keep that local-only behavior in this scope rather than silently migrating capture to the browser.

## Alternatives and risks

A framework rewrite would increase regression risk without helping this screen, so keep NiceGUI. A separate forced aligner could make karaoke precise, but adds model/runtime cost and is not needed for text-only ASR or whole-sentence A/B.

Playback, recording, navigation, and theme changes must not allow stale async results to overwrite the current sentence. A/B must stop or cancel safely on navigation and never overlap microphone recording.

There are existing uncommitted changes in README.md, coach/asr_worker.py, coach/main.py, pyproject.toml, and requirements.txt. Preserve them. No tracked tests were found during discovery.

## Verification

Run new automated tests and any existing checks discovered during implementation. Verify desktop and mobile layouts in both themes, keyboard focus, reduced motion, empty/error states, and state transitions. Exercise real model inference and local audio when available; report any hardware or model-download limits separately from mocked test results.

## Approval requested

Confirm the design and scope, including approximate karaoke timing and preservation of the local microphone behavior. Start with paper/candlelit visual previews, then implement the approved screen.
