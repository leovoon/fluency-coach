# Shadowing room

Date: 2026-09-30
POC: AdaL
TL;DR: A centered, unscored practice screen with distinct teacher and learner readings. The user's supplied practice-room brief defines the visual direction.

## Surface

The primary route `/` uses `coach/room.py`; rendering helpers and theme styles live in `coach/shadowing.py`. The previous diagnostic surface is retained at `/classic` as a compatibility path. The sample state at `/?preview=true` is labeled as illustrative and never counts as a learner recording.

## Color

Paper uses #FAF7F2 with #302C27 ink, #9B712D teacher ochre, and #587575 learner slate-teal. Candlelit uses #211E1A with #E9DFCF parchment, #D2AB67 ochre, and #9BB4AE slate-teal. Secondary text uses #777167 in paper and #B2A797 in candlelit. These are semantic roles, not success/failure colors.

## Type and layout

Fraunces is the reading face, 48px for the model sentence and 30px for the learner. Mobile uses 35px and 25px. Manrope carries small controls and labels. Google Fonts supplies both fonts; Georgia and sans-serif are offline fallbacks.

The practice column is at most 760px wide, with no enclosing card. Progress dots lead into the model label, a 66px contour area, the model sentence, a central microphone and fine waveform, and the quieter learner reading. Footer actions are play, record, and again. Setup stays in a collapsed section below the practice area. It contains full-width Passage, Sentence reference, and Teaching voice sections with separated headings and helper text. The passage editor has a 180px minimum height. Optional uploads are collapsed; dropdown popups inherit the active theme. Sun/moon icons switch themes, and a keyboard-icon button opens the shortcut guide. Theme colors interpolate together over 300ms through registered palette variables; the two icons crossfade in place. Word highlight colors have no transition delay. Reduced-motion mode switches the palette and icons immediately. A compact vertical rhythm keeps actions visible on short desktop screens.

## Meaning and states

Ochre fills model words during playback. Timing follows recording duration and word count; it is approximate, not forced alignment. The pitch ribbon is measured from actual model/reference audio and omitted until available. The waveform is decorative, not a live level meter.

During learner playback, upcoming recognized words use ink and progressively fill slate-teal. Substitutions and insertions follow the transcript order; punctuation does not consume timing slots. Each replay begins unfilled. Learner timing uses an energy-based estimate of the speech span, leaving surrounding silence out of the highlight schedule while preserving the audio. Words light at estimated onsets. No detected speech falls back to the file duration; this is not forced alignment. The learner row preserves skipped words as ghosts. Substituted and inserted words have dotted underlines and accessible descriptions. Before recording, the row says where the learner's voice will appear. Empty ASR output and failures are described without assigning grades.

A/B listens to the saved reference or synthesized model, followed by the learner take. Labels allow independent playback. Navigation, setup, and new recording are disabled while audio work is active; record becomes stop during capture. Again clears only the current take. Saved reference audio is never deleted by reset.

## Accessibility and motion

Controls have accessible names and keyboard focus outlines. P plays the model/learner pair; Space records or stops; left/right move between sentences outside form controls. The waveform pulses vertically only during microphone capture and stops before transcription or on failure. Reduced-motion preferences disable both waveform and microphone breathing animations. Skipped-word contrast is intentionally low to distinguish absence; keyboard focus and hover reveal it at full opacity, and the accessible label states what was not heard.

## Known constraints

Recording and speakers belong to the local server machine. Precise word timing would need a separate alignment stage. Model and voice loading can take time on first use. Fonts need a network connection on first load. No live microphone level meter or word-slice comparison is presented as available.
