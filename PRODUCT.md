# Product

<!-- impeccable:product-schema 1 -->

Date: 2026-09-30
POC: AdaL
TL;DR: A local read-aloud partner for listening to a teaching voice, recording a sentence, and hearing the two readings together.

## Platform

web

## Users

English learners practicing one sentence at a time on their own Mac.

## Product purpose

Support repeated listening and speaking without grades, scores, or judgment colors. Compare the words recognized in a learner's recording with the passage.

## Operating context

The existing NiceGUI app runs locally. Recording and playback use the server machine's microphone and speakers. Passage text can be pasted, uploaded, selected from templates, or sent by the browser extension.

## Capabilities and constraints

Phonon-2 supplies text-only ASR. It does not supply word timestamps through the current adapter. Model karaoke timing is approximate. Saved human references take precedence over synthesized speech. PocketTTS clones a selected teaching voice; labels must distinguish cloned models from human references and other synthetic voices.

Voice management, saved references, and passage import must survive the redesign. Practice feedback excludes melody arrows, flow marks, scores, and grades. A/B playback compares complete recordings.

## Brand commitments

The approved screen uses warm paper, ink, ochre for the teaching voice, muted slate-teal for the learner, Fraunces display text, and generous whitespace. A candlelit theme supports darker rooms. The footer has play, record, and again only.

## Accessibility

Keyboard-operable controls, visible focus, reduced motion, and text descriptions of skipped and substituted words. Sample results are explicitly labeled and are never shown as real learner output.
