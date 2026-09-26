"""NiceGUI web UI for the fluency coach.

Same pipeline as the terminal loop (coach/main.py), rendered in the browser:
sentence → record (server mic) → ASR → diff coloring → melody arrows →
optional Jev / phonemes → read-back.

The learner's saved reference is the voice to imitate. A TTS read is a model
read, not a native speaker, and is never played over a saved reference.

Run: python -m coach.ui   (host/port from coach.yaml / CLI, default localhost)

Note: the mic is the server machine's (sounddevice), so this is a
single-user localhost app.
"""

from __future__ import annotations

import asyncio
import inspect
import secrets
import sys
import tempfile
import time
from pathlib import Path

from nicegui import background_tasks, run, ui
from fastapi import Request

from . import diff as diffmod
from . import flow, jev, melody, models, phonemes, prominence, record, reference, rhythm


def _poll_tts_stage(status) -> None:
    """Live load/render progress while the TTS thread works — replaces the
    silent spinner with what the engine is actually doing."""
    stage = models.tts_load_stage()
    if stage:
        status.set_text(stage)
from .config import load
from .main import DEFAULT_PASSAGE, load_passage, suspect_payload

# Passages pushed by the browser extension, keyed by one-shot token.
# Bounded so a forgotten tab cannot grow this forever.
_extension_passages: dict[str, str] = {}


def _split_sentences(text: str) -> list[str]:
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


WORD_COLOR = {
    "match": "text-green-600",
    "sub": "text-red-500",
    "skip": "text-gray-400 line-through",
    "insert-context": "text-gray-400",
}
ARROW_COLOR = {"up": "text-amber-500", "down": "text-green-600",
               "flat": "text-gray-400", "none": "text-gray-300"}

NO_VOICE = "no voice model in this tier — record a reference to compare"
NEED_REF_CHUNK = "record a reference to hear this chunk"
MODEL_READ = "model read (not a native speaker)"  # engine-specific label: models.model_read_label()


def _template_passages() -> dict[str, str]:
    """Built-in practice passages: {title: text} from coach/passages/*.txt.
    First line '# Title' names the passage; the rest is the text."""
    d = Path(__file__).parent / "passages"
    out: dict[str, str] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text.startswith("# "):
            title, _, text = text.partition("\n")
            title = title[2:].strip()
        else:
            title = p.stem.replace("_", " ")
        if text.strip():
            out[title] = text.strip()
    return out


def _jev_line(verdicts: dict, suspects: list[dict]) -> str:
    """Jev verdicts as one learner-facing line: only the words that need
    work, grouped and capped. Probabilities are machine-facing — a learner
    can't act on 'p(read)=0.43'."""
    missed: list[str] = []
    unclear: list[str] = []
    for s in suspects:
        i = s["index"]
        p = verdicts["read"].get(i)
        cat = (verdicts["category"].get(i) or {}).get("choice")
        if cat in (None, "no-error"):
            continue
        if p is not None and p >= 0.5:
            continue  # flagged by the diff, judged fine by Jev
        name = s["word"]
        (missed if cat == "skipped" else unclear).append(name)
    if len(missed) > max(2, len(suspects) // 2):
        # most of the sentence didn't line up — a word-level list won't help
        return ("most words didn't line up with the passage — "
                "check your transcript and try again")
    parts = []
    if missed:
        more = "…" if len(missed) > 4 else ""
        parts.append("missed: " + ", ".join(missed[:4]) + more)
    if unclear:
        more = "…" if len(unclear) > 4 else ""
        parts.append("unclear: " + ", ".join(unclear[:4]) + more)
    return " · ".join(parts) or "every flagged word checked out fine"


def _settings():
    return load(sys.argv)


def _tts_allowed(settings, options: dict) -> bool:
    """Tier with no voice model must not synthesize or speak, even if toggled."""
    return bool(settings.tts) and bool(options.get("tts"))


def _groups_from_marks(sentence: str, marks: dict[int, str]) -> list[tuple[int, int]]:
    """Consecutive link marks as inclusive spans, so a click works pre-record."""
    words = sentence.split()
    groups: list[tuple[int, int]] = []
    start = None
    for i in range(len(words) - 1):
        if marks.get(i) == "link":
            if start is None:
                start = i
        elif start is not None:
            if i - start >= 1:
                groups.append((start, i))
            start = None
    if start is not None and (len(words) - 1) - start >= 1:
        groups.append((start, len(words) - 1))
    return groups


def colored_passage(sentence: str, alignment: diffmod.Alignment | None,
                    on_word=None, arrows: dict | None = None,
                    marks: dict | None = None, ref_arrows: dict | None = None,
                    stress: dict[int, str] | None = None,
                    focus: str | None = None) -> None:
    """Passage words, colored by diff status — the terminal's show_sentence.

    on_word(word, passage_index) fires when a word is clicked.
    The passage index is the enumerate index when alignment is missing, so
    link groups can play before the learner records.
    arrows maps passage word index -> melody direction, rendered under the word.
    marks maps boundary index -> {"kind", "ok"}: | pause, ‿ link, · hesitation.
    ok=None = guide (pre-record), True/False = scored.
    stress maps passage word index -> "flat" (beat you flattened, amber dotted
    underline) or "extra" (small word you pushed, small gray dot).
    focus names the one feedback dimension for this pass: 'flow' keeps stress
    but hides melody arrows; 'stress' hides arrows and grays the flow marks;
    'melody' grays the flow marks and hides stress. Unfocused layers stay in
    the scoring labels — only the passage paint narrows.
    """
    words = sentence.split()
    by_index = {w.index: w for w in alignment.words} if alignment else {}
    norm_index = 0
    if focus in ("flow", "stress"):
        arrows = None
        ref_arrows = None
    show_stress = None if focus == "melody" else stress
    gray_marks = focus in ("stress", "melody")
    if focus:
        legend = {"flow": "this pass: pauses", "stress": "this pass: stress",
                  "melody": "this pass: melody"}
        ui.label(legend.get(focus, "")).classes("text-xs text-amber-600")

    def mark_label(i: int) -> None:
        if not marks or i not in marks:
            return
        m = marks[i]
        if not m["kind"] and m["ok"] is not False:
            return  # unlabeled, nothing to say
        if gray_marks or m["ok"] is None:
            color = "text-gray-400"
        else:
            color = "text-green-600" if m["ok"] else "text-amber-500"
        if m["kind"] == "pause":
            icon = ui.icon("air", size="1.1rem").classes(f"{color} -mx-1 mt-1")
            icon.tooltip("breathe here, before the next word — "
                         "green: you took the pause · amber: you didn't")
        else:
            mark = ui.label(flow.mark_char(m["kind"])).classes(
                f"text-base font-bold {color} -mx-1 mt-1")
            if m["kind"] == "link":
                mark.tooltip("blend these two words into one sound — "
                             "green: connected · amber: choppy")
            elif m["ok"] is False:
                mark.tooltip("you paused here although no mark planned it — "
                             "a hesitation")

    with ui.row().classes("flex-wrap gap-x-3 gap-y-2 items-start"):
        for wi, raw in enumerate(words):
            with ui.column().classes("items-center gap-0"):
                idx = None
                color = ""
                if alignment:
                    norm = diffmod.normalize(raw)
                    while norm_index < len(words):
                        st = by_index.get(norm_index)
                        if st and st.norm == norm:
                            idx = st.index
                            color = WORD_COLOR.get(st.status, "text-amber-500")
                            break
                        norm_index += 1
                    norm_index += 1
                label = ui.label(raw).classes(f"text-xl {color}")
                if on_word:
                    # Passage index even when this word has no alignment yet.
                    click_idx = wi if idx is None else idx
                    label.classes("cursor-pointer hover:opacity-70")
                    label.on("click", lambda _, word=raw, widx=click_idx: on_word(word, widx))
                d = arrows.get(idx if idx is not None else wi) if arrows else None
                rd = (ref_arrows or {}).get(idx if idx is not None else wi)
                if d:
                    arrow = {"up": "↗", "down": "↘", "flat": "→", "none": "·"}[d]
                    ui.label(arrow).classes(f"text-sm font-bold {ARROW_COLOR[d]}")
                if rd is not None and rd != d:
                    # Reference differs from the learner here — small, dim, delta-only.
                    rarrow = {"up": "↗", "down": "↘", "flat": "→", "none": "·"}[rd]
                    ui.label(rarrow).classes("text-xs text-gray-400")
                sp = show_stress.get(idx if idx is not None else wi) if show_stress else None
                if sp == "flat":
                    label.classes(
                        "underline decoration-dotted decoration-amber-500 underline-offset-4")
                    label.tooltip("stress this")
                elif sp == "extra" and not d:
                    dot = ui.label("•").classes("text-xs text-gray-400")
                    dot.tooltip("too loud — go lighter")
            mark_label(wi)


@ui.page("/")
def index(passage: str | None = None):
    settings = _settings()
    read_label = models.model_read_label()
    # A ?passage=<token> URL comes from the browser extension. The token
    # stays in the store (bounded) so reloading the coach tab re-loads the
    # same text instead of falling back to the default passage.
    pushed = _extension_passages.get(passage) if passage else None
    initial_text = pushed or DEFAULT_PASSAGE
    sentences: list[str] = load_passage(None) if pushed is None else _split_sentences(pushed)
    current = {"i": 0}
    jump_guard = {"on": False}  # suppress on_jump while show_sentence syncs the select
    recording = {"on": False}
    # Bumped on navigation. In-flight refreshes must not paint or delete after it moves.
    gen = {"n": 0}
    options = {
        "melody": bool(settings.melody),
        "phonemes": bool(settings.phonemes),
        "jev": bool(settings.jev),
        "tts": bool(settings.tts),
        "ref": bool(settings.ref_compare),
        "flow": bool(settings.flow),
        "guide": False,     # pre-speech rule guide — opt-in scaffold
        "diagnose": False,  # pitch chart + per-word arrows — drill-down
    }
    result_state: dict = {}  # last scoring render (alignment/arrows/marks/ref)
    # Measured labels per (sentence, has-human-ref). Without this, every
    # sentence navigation repainted marks twice: rule guide instantly, measured
    # labels seconds later after synth+ASR — marks appeared and disappeared.
    label_cache: dict = {}

    ui.page_title("fluency coach")
    ui.query(".nicegui-content").classes("max-w-4xl mx-auto p-4")

    def stale(token: int) -> bool:
        return token != gen["n"]

    async def on_ref_toggle(e) -> None:
        options["ref"] = e.value
        if e.value:
            if result_state.get("alignment") is not None:
                status.set_text("comparing with reference…")
                await refresh_ref_arrows()
        else:
            result_state.pop("ref_arrows", None)
            if result_state.get("alignment") is not None:
                _paint_result(sentences[current["i"]])
                status.set_text("")

    def on_jump(e) -> None:
        if recording["on"] or jump_guard["on"] or e.value is None:
            return
        current["i"] = e.value
        show_sentence()

    # Primary loop first: controls, then the sentence card, then the actions
    # that operate on it. Configuration lives at the bottom of the page.
    with ui.column().classes('w-full gap-2'):
        with ui.row().classes('w-full items-baseline justify-between'):
            ui.label('fluency coach').classes('text-2xl font-bold')
            ui.label('read · compare · repeat, one sentence at a time').classes(
                'text-sm text-gray-400')
        with ui.row().classes('gap-2 items-center'):
            record_btn = ui.button("record", icon="mic").props(
                'unelevated color=primary size=lg')
            stop_btn = ui.button("stop", icon="stop").props(
                'flat color=negative size=lg').classes('hidden')
            jump_select = ui.select(
                {i: f"{i + 1}. {s[:40]}" for i, s in enumerate(sentences)},
                value=0, label="sentence", on_change=on_jump,
            ).classes("w-64")
            prev_btn = ui.button("previous", icon="skip_previous").props('flat')
            next_btn = ui.button("next sentence", icon="skip_next").props('flat')

        progress = ui.label("")
        status = ui.label("").classes("text-gray-500")

        with ui.card().classes('w-full'):
            passage_view = ui.column().classes('w-full gap-2')
            transcript_label = ui.label("").classes('text-gray-600')
            tune_label = ui.label("").classes('text-sm text-gray-500')
            rhythm_label = ui.label("").classes('text-sm text-gray-500')
            flow_label = ui.label("").classes('text-sm text-gray-500')
            extra_label = ui.label("").classes('text-sm text-amber-600')
            jev_label = ui.label("").classes('text-sm')
            phon_label = ui.label("").classes('text-sm')

        with ui.row().classes("gap-1 items-center"):
            ref_rec_btn = ui.button("record reference", icon="record_voice_over").props("flat dense")
            hear_ref_btn = ui.button("hear reference", icon="hearing").props("flat dense")
            unlink_ref_btn = ui.button("unlink reference", icon="link_off").props("flat dense")
            hear_user_btn = ui.button("hear your take", icon="graphic_eq").props("flat dense")
            hear_user_btn.disable()
            unlink_ref_btn.disable()

        with ui.row().classes("gap-x-4 flex-wrap items-center text-xs text-gray-400"):
            melody_legend = ui.label("↗ bold = your melody · ↗ small gray = reference differs")
            melody_legend.tooltip("the arrow shows where your pitch moved on that word; "
                                  "a small gray arrow appears only where the reference went "
                                  "a different way")
            air_legend = ui.icon("air", size="0.9rem")
            air_legend.tooltip("a chunk boundary: let the phrase end, take a little "
                               "breath, start the next phrase fresh")
            ui.label("= breathe here (green took it / amber missed)")
            link_legend = ui.label("‿ = link the words (green connected / amber choppy)")
            link_legend.tooltip("a consonant meeting a vowel: say the two words as one — "
                                "'an apple' becomes 'anapple'")
            stress_legend = ui.label(".... = beat you flattened · • = small word you pushed")
            stress_legend.tooltip("the reference gives some words more energy than others; "
                                  "dotted underline: push that word more · dot: keep it light")

    def _stress_marks(sentence: str) -> dict[int, str] | None:
        """Prominence marks, you vs reference. None until both sides exist."""
        if not options["melody"]:
            return None
        ref_e = result_state.get("ref_energy")
        user_e = result_state.get("user_energy")
        if not ref_e or not user_e:
            return None
        pm = prominence.prominence(ref_e, user_e, sentence.split())
        result_state["stress"] = pm
        return pm or None

    def _show_rhythm(sentence: str) -> None:
        """Stress-timing + nucleus tone. Safe before a reference exists."""
        user_ts = result_state.get("user_ts") or {}
        user_dir = result_state.get("user_dir") or {}
        ref_ts = result_state.get("ref_ts") or None
        ref_dir = result_state.get("ref_dir") or None
        user_slopes = result_state.get("user_slopes") or None
        ref_slopes = result_state.get("ref_slopes") or None
        parts = []
        try:
            accepts_slopes = ("user_slopes"
                              in inspect.signature(rhythm.phrase_final).parameters)
        except (TypeError, ValueError):
            accepts_slopes = False
        if accepts_slopes:
            ending = rhythm.phrase_final(sentence, user_dir, ref_dir,
                                         user_slopes=user_slopes,
                                         ref_slopes=ref_slopes)
        else:
            # rhythm without slope params yet — slopes land on the next merge.
            ending = rhythm.phrase_final(sentence, user_dir, ref_dir)
        beats = rhythm.rhythm_line(sentence, user_ts, ref_ts)
        if ending:
            parts.append(ending)
        if beats:
            parts.append(beats)
        pm = _stress_marks(sentence)
        if pm:
            line = prominence.summarise(pm, sentence.split())
            if line:
                parts.append(line)
        rhythm_label.set_text(" · ".join(parts))

    @ui.refreshable
    def pitch_chart() -> None:
        if not options["melody"]:
            return
        data = result_state.get("chart")
        if not data:
            ui.label("record to see your pitch against the reference").classes(
                "text-sm text-gray-400")
            return
        ui.echart({
            "grid": {"left": 40, "right": 16, "top": 28, "bottom": 56},
            "xAxis": {"type": "category", "data": data["words"],
                      "axisLabel": {"rotate": 30, "fontSize": 10}},
            "yAxis": {"type": "value", "name": "st",
                      "axisLabel": {"fontSize": 10}},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["you", "reference"], "top": 0},
            "series": [
                {"name": "you", "type": "line", "data": data["you"],
                 "connectNulls": True, "symbolSize": 6,
                 "lineStyle": {"width": 2.5}, "color": "#f59e0b"},
                {"name": "reference", "type": "line", "data": data["ref"],
                 "connectNulls": True, "symbolSize": 4,
                 "lineStyle": {"width": 1.5, "type": "dashed"}, "color": "#10b981"},
            ],
        }).classes("w-full h-44")

    chart_card = ui.card().classes("w-full")
    with chart_card:
        ui.label("melody — where each word landed, vs that recording's own middle. Compare the shape, not the numbers. Click a linked word to hear the group.").classes(
            "text-xs text-gray-500")
        chart_view = ui.column().classes("w-full")
        with chart_view:
            pitch_chart()
    if not options["melody"] or not options["diagnose"]:
        chart_card.set_visibility(False)

    def _on_melody(value: bool) -> None:
        options["melody"] = value
        chart_card.set_visibility(bool(value) and options["diagnose"])
        if value:
            pitch_chart.refresh()

    def _on_guide(value: bool) -> None:
        """Guide marks are an opt-in pre-speech scaffold, off by default."""
        options["guide"] = value
        if result_state.get("alignment") is not None:
            return  # a recorded result stays — only the fresh paint changes
        sentence = sentences[current["i"]]
        labels = result_state.get("labels")
        if labels is None:
            labels = flow.boundaries(sentence)
        guide = ({n: {"kind": k, "ok": None} for n, k in labels.items()
                  if k != "none"} if options["flow"] else None)
        passage_view.clear()
        with passage_view:
            colored_passage(sentence, None, on_word=on_word_click,
                            marks=guide if value else None)

    def _on_diagnose(value: bool) -> None:
        """Pitch chart + per-word arrows are drill-down diagnostics, off by
        default. Scoring is untouched — display only."""
        options["diagnose"] = value
        chart_card.set_visibility(value and options["melody"])
        if result_state.get("alignment") is not None:
            _paint_result(sentences[current["i"]])

    with ui.expansion('passage text', icon='edit').classes('w-full'):
        ui.label('paste your passage, then load it — the coach splits it into sentences').classes(
            'text-xs text-gray-400')
        _templates = _template_passages()
        if _templates:
            tpl_select = ui.select(_templates, label='or start from a template passage',
                                   value=next(iter(_templates)), with_input=False,
                                   ).props('dense outlined').classes('w-full')

            def _use_template() -> None:
                if recording["on"]:
                    return
                text = _templates.get(tpl_select.value)
                if text:
                    passage_input.value = text
                    load_passage_text()

            with ui.row().classes('gap-2 items-center'):
                ui.button('use this template', icon='auto_stories').props(
                    'flat dense').on_click(_use_template)
        passage_input = ui.textarea("passage", value=initial_text
                                    ).classes("w-full")
        with ui.row().classes('gap-2'):
            load_btn = ui.button("load passage").props('flat dense')
            read_all_btn = ui.button("read whole passage", icon="campaign").props('flat dense')

            async def _on_passage_file(e) -> None:
                if recording["on"]:
                    return
                passage_input.value = (await e.file.read()).decode("utf-8", "ignore")
                load_passage_text()

            ui.upload(label="or upload a .txt passage", multiple=False,
                      on_upload=_on_passage_file
                      ).props("accept=.txt,text/plain,dont-verify") \
             .classes("max-w-xs").props('dense')

    with ui.card().classes('w-full bg-gray-50'):
        ui.label('LAYERS & VOICE').classes('text-xs text-gray-400 tracking-wide')
        with ui.row().classes('gap-x-5 gap-y-1 flex-wrap items-center'):
            ui.switch("melody", value=options["melody"],
                      on_change=lambda e: _on_melody(e.value)).props('dense')
            ui.switch("flow", value=options["flow"],
                      on_change=lambda e: options.update(flow=e.value)).props('dense')
            ui.switch("show guide marks", value=False,
                      on_change=lambda e: _on_guide(e.value)).props('dense')
            ui.switch("diagnose", value=False,
                      on_change=lambda e: _on_diagnose(e.value)).props('dense')
            ui.switch("phonemes", value=options["phonemes"],
                      on_change=lambda e: options.update(phonemes=e.value)).props('dense')
            ui.switch("jev (charged)", value=options["jev"],
                      on_change=lambda e: options.update(jev=e.value)).props('dense')
            tts_switch = ui.switch(
                "read-back", value=options["tts"],
                on_change=lambda e: options.update(tts=e.value)).props('dense')
            if not settings.tts:
                tts_switch.disable()
            ui.switch("compare reference", value=options["ref"],
                      on_change=on_ref_toggle).props('dense')
        if settings.tts_engine == "kokoro" and settings.tts:
            def _on_voice(e) -> None:
                try:
                    models.set_voice(e.value)
                except Exception as ex:
                    ui.notify(str(ex), type="negative")
                    return
                # Model-read reference data was made with the old voice — drop it.
                old = result_state.get("ref_wav")
                if old and result_state.get("ref_wav_is_tmp"):
                    Path(old).unlink(missing_ok=True)
                for k in ("ref_wav", "ref_ts", "ref_st", "ref_arrows",
                          "ref_gaps", "ref_dir", "ref_slopes", "ref_energy", "groups"):
                    result_state.pop(k, None)
                if not reference.has(sentences[current["i"]]):
                    status.set_text(f"voice: {e.value} — model read will use it from now on")
                    if result_state.get("alignment") is not None:
                        background_tasks.create(refresh_ref_arrows())

            if settings.tts_engine == "kokoro":
                # Chatterbox/PocketTTS have no preset voice menu — the model
                # model, Air speaks in the reference clip's voice.
                _voice_opts = {
                    v: v
                    for group, vs in models.KOKORO_VOICES.items()
                    for v in vs
                }
                with ui.row().classes("gap-1 items-center"):
                    ui.label("voice — the model read and read-back you'll hear").classes(
                        "text-xs text-gray-500")
                    ui.select(
                        _voice_opts, value=models.current_voice(),
                        on_change=_on_voice,
                    ).props("dense outlined").classes("w-44")


        if settings.tts_engine == "pocket":
            cur_voice = models.model_voice_path()
            ui.label("teaching voice — what the model read sounds like").classes(
                "text-xs text-gray-400 mt-2")
            voice_status = ui.label(
                f"current clip: {cur_voice.name}" if cur_voice else
                "none yet — record or upload a 3–30s clip of the voice to imitate"
            ).classes("text-xs text-gray-500")
            pending_voice: dict = {}

            async def _save_voice() -> None:
                if not pending_voice.get("wav"):
                    voice_status.set_text("record or upload a clip first")
                    return
                try:
                    res = models.save_voice(voice_name.value or "voice",
                                            pending_voice["wav"])
                    Path(pending_voice["wav"]).unlink(missing_ok=True)
                    pending_voice.clear()
                    voices_gallery.refresh()
                    note = ", trimmed" if res["trimmed"] else ""
                    voice_status.set_text(
                        f"saved + active: {res['name']} ({res['duration']:.1f}s{note})")
                except Exception as e:
                    voice_status.set_text(f"failed: {str(e)[:140]}")

            def _activate_voice(n: str) -> None:
                try:
                    res = models.activate_voice(n)
                    voices_gallery.refresh()
                    voice_status.set_text(
                        f"switched to {res['name']} ({res['duration']:.1f}s) — "
                        "next model read uses it")
                except Exception as e:
                    voice_status.set_text(f"failed: {str(e)[:140]}")

            async def _delete_voice(n: str) -> None:
                models.delete_voice(n)
                voices_gallery.refresh()
                voice_status.set_text(f"deleted {n}")

            async def _play_voice(p: str) -> None:
                from . import record
                await run.io_bound(record.play_wav, p)

            async def _record_voice() -> None:
                voice_status.set_text("recording — the voice speaking naturally…")
                try:
                    pending_voice["wav"] = await run.io_bound(record.record_utterance)
                except Exception as e:
                    voice_status.set_text(f"record failed: {str(e)[:120]}")
                    return
                import soundfile as sf
                dur = sf.info(str(pending_voice["wav"])).duration
                voice_status.set_text(
                    f"clip captured ({dur:.1f}s) — save it as the teaching voice")

            async def _on_voice_file(e) -> None:
                fd, tmp = tempfile.mkstemp(suffix=Path(e.file.name).suffix or ".wav")
                import os as _os
                _os.close(fd)
                tmp = Path(tmp)
                try:
                    tmp.write_bytes(await e.file.read())
                    import soundfile as sf
                    dur = sf.info(str(tmp)).duration
                except Exception as ex:
                    tmp.unlink(missing_ok=True)
                    voice_status.set_text(
                        f"could not read {e.file.name}: {str(ex)[:100]} — "
                        "use wav, flac, ogg or aiff")
                    return
                pending_voice["wav"] = tmp
                voice_status.set_text(
                    f"uploaded {e.file.name} ({dur:.1f}s) — save it as the teaching voice")

            voice_name = ui.input("voice name", value="my voice").classes("w-40")
            with ui.row().classes("gap-2 items-center w-full"):
                ui.button("record clip", icon="mic").props(
                    "flat dense").on_click(_record_voice)
                ui.upload(auto_upload=True, on_upload=_on_voice_file
                          ).props("accept=.wav,.flac,.ogg,.aiff,.aif,.mp3,dont-verify") \
                 .classes("max-w-xs").props("dense")
                ui.button("save", icon="record_voice_over").props(
                    "flat dense").on_click(_save_voice)

            @ui.refreshable
            def voices_gallery() -> None:
                voices = models.list_voices()
                if not voices:
                    ui.label("no saved voices yet").classes("text-xs text-gray-400")
                    return
                with ui.column().classes("gap-1 w-full"):
                    ui.label("voices gallery — click a name to switch").classes(
                        "text-xs text-gray-400")
                    for v in voices:
                        with ui.row().classes("gap-1 items-center"):
                            label = f"{v['name']} ({v['duration']:.0f}s)" + (
                                " — active" if v["active"] else "")
                            ui.button(label, icon="record_voice_over").props(
                                "flat dense").classes("text-xs").on_click(
                                    lambda _, n=v["name"]: _activate_voice(n))
                            ui.button("play", icon="play_arrow").props(
                                "flat dense").classes("text-xs").on_click(
                                    lambda _, p=v["path"]: _play_voice(p))
                            ui.button("delete", icon="delete").props(
                                "flat dense").classes("text-xs").on_click(
                                    lambda _, n=v["name"]: _delete_voice(n))

            voices_gallery()

    def set_nav(enabled: bool) -> None:
        for el in (prev_btn, next_btn, load_btn, jump_select):
            el.enable() if enabled else el.disable()

    def load_passage_text() -> None:
        if recording["on"]:
            return
        import re
        sentences[:] = [s.strip() for s in
                        re.split(r"(?<=[.!?])\s+", passage_input.value) if s.strip()]
        current["i"] = 0
        show_sentence()

    def on_word_click(word: str, idx: int | None = None) -> None:
        """Hear your slice, then the reference chunk. Own voice wins over TTS."""
        async def say():
            token = gen["n"]
            sentence = sentences[current["i"]]
            human = reference.has(sentence)
            group = None
            if idx is not None:
                for a, b in result_state.get("groups") or []:
                    if a <= idx <= b:
                        group = (a, b)
                        break
            user_wav = result_state.get("user_wav")
            user_ts = result_state.get("user_ts") or {}
            ref_wav = reference.path_for(sentence) if human else result_state.get("ref_wav")
            ref_ts = result_state.get("ref_ts") or {}
            tts_ok = _tts_allowed(settings, options)
            played = False

            def alive() -> bool:
                return not stale(token)

            if group and idx is not None:
                a, b = group
                phrase = " ".join(sentence.split()[a:b + 1])
                if not alive():
                    return
                status.set_text(f"group “{phrase}” — your voice, then reference…")
                if user_wav and a in user_ts and b in user_ts:
                    await run.io_bound(record.play_wav_slice, user_wav,
                                       user_ts[a]["start"], user_ts[b]["end"], pad=0.03)
                    played = True
                    await asyncio.sleep(0.15)
                if not alive():
                    return
                if ref_wav and a in ref_ts and b in ref_ts:
                    await run.io_bound(record.play_wav_slice, ref_wav,
                                       ref_ts[a]["start"], ref_ts[b]["end"], pad=0.03)
                    played = True
                elif not human and tts_ok:
                    await run.io_bound(models.speak, phrase)
                    played = True
                elif not ref_wav and not tts_ok:
                    ui.notify(NEED_REF_CHUNK)
                    if alive():
                        status.set_text(NEED_REF_CHUNK)
                    return
            else:
                if not alive():
                    return
                status.set_text(f"“{word}” — your voice, then reference…")
                if user_wav and idx in user_ts:
                    t = user_ts[idx]
                    await run.io_bound(record.play_wav_slice, user_wav, t["start"], t["end"])
                    played = True
                    await asyncio.sleep(0.15)
                if not alive():
                    return
                if ref_wav and idx in ref_ts:
                    t = ref_ts[idx]
                    await run.io_bound(record.play_wav_slice, ref_wav, t["start"], t["end"])
                    played = True
                elif not played and not human and tts_ok:
                    await run.io_bound(models.speak_word, word)
                    played = True
                elif not played and not ref_wav and not tts_ok:
                    ui.notify(NEED_REF_CHUNK)
                    if alive():
                        status.set_text(NEED_REF_CHUNK)
                    return
            if alive():
                status.set_text("")
        background_tasks.create(say())

    def _play_chunk(a: int, b: int) -> None:
        """Play the learner's slice, then the reference slice for words a..b.
        A side with missing timestamps is skipped silently."""
        sentence = sentences[current["i"]]
        token = gen["n"]

        async def play() -> None:
            phrase = " ".join(sentence.split()[a:b + 1])
            if not stale(token):
                status.set_text(f"chunk “{phrase}” — you, then the reference…")
            user_wav = result_state.get("user_wav")
            user_ts = result_state.get("user_ts") or {}
            if user_wav and a in user_ts and b in user_ts:
                await run.io_bound(record.play_wav_slice, user_wav,
                                   user_ts[a]["start"], user_ts[b]["end"], pad=0.03)
                await asyncio.sleep(0.15)
            if stale(token):
                return
            human = reference.has(sentence)
            ref_wav = (reference.path_for(sentence) if human
                       else result_state.get("ref_wav"))
            ref_ts = result_state.get("ref_ts") or {}
            if ref_wav and a in ref_ts and b in ref_ts:
                await run.io_bound(record.play_wav_slice, ref_wav,
                                   ref_ts[a]["start"], ref_ts[b]["end"], pad=0.03)
            if not stale(token):
                status.set_text("")

        background_tasks.create(play())

    def _chunk_buttons(sentence: str) -> None:
        """One small button per pause-delimited chunk (flow.pause_groups),
        under the passage. Whole sentence as one chunk when no pause marks.
        Missing pause_groups degrades to that same whole-sentence fallback."""
        words = sentence.split()
        marks = result_state.get("marks")
        if not options["flow"] or not marks or len(words) < 2:
            return
        groups = None
        fn = getattr(flow, "pause_groups", None)
        if fn is not None:
            try:
                groups = fn(sentence, marks)
            except Exception:
                groups = None
        if not groups:
            groups = [(0, len(words) - 1)]
        with ui.row().classes("gap-1 flex-wrap items-center mt-1"):
            ui.label("replay a chunk — you, then the reference").classes(
                "text-xs text-gray-400")
            for n, span in enumerate(groups[:8]):
                a, b = span
                ui.button(f"chunk {n + 1}", on_click=lambda _, a=a, b=b: _play_chunk(a, b)
                          ).props("flat dense").classes("text-xs")
            if len(groups) > 8:
                ui.label(f"+{len(groups) - 8} more").classes("text-xs text-gray-400")

    def _paint_result(sentence: str) -> None:
        """Repaint a recorded attempt: diff + the pass's one focus dimension
        + chunk replay buttons. Diagnose gates the per-word arrows."""
        stress = result_state.get("stress")
        focus = prominence.choose_focus(
            result_state.get("marks") if options["flow"] else None,
            stress if options["melody"] else None)
        result_state["focus"] = focus
        if not options["melody"]:
            focus = None  # melody lens is meaningless with the layer switched off
        show_arrows = (result_state.get("arrows")
                       if options["melody"] and options["diagnose"] else None)
        show_ref = (result_state.get("ref_arrows")
                    if options["melody"] and options["ref"] and options["diagnose"]
                    else None)
        passage_view.clear()
        with passage_view:
            colored_passage(
                sentence, result_state.get("alignment"), on_word=on_word_click,
                arrows=show_arrows,
                marks=result_state.get("marks") if options["flow"] else None,
                ref_arrows=show_ref,
                stress=stress if options["melody"] else None,
                focus=focus)
            _chunk_buttons(sentence)

    async def on_record_reference() -> None:
        if recording["on"] or not sentences:
            return
        sentence = sentences[current["i"]]
        token = gen["n"]
        recording["on"] = True
        set_nav(False)
        ref_rec_btn.disable()
        ref_rec_btn.props('loading')
        record_btn.disable()
        try:
            status.set_text("🎙 record the reference — read it the way you want to sound")
            wav = await run.io_bound(record.record_utterance)
            if stale(token):
                Path(wav).unlink(missing_ok=True)
                return
            await run.io_bound(reference.save, sentence, wav)
            wav.unlink(missing_ok=True)
            if stale(token):
                return
            ref_rec_btn.text = "re-record reference"
            unlink_ref_btn.enable()
            ui.notify("reference saved — your voice now drives the comparison",
                      type="positive")
            status.set_text("")
            await refresh_ref_arrows()
        except Exception as e:
            ui.notify(f"reference recording failed: {e}", type="negative")
            if not stale(token):
                status.set_text("reference recording failed")
        finally:
            recording["on"] = False
            ref_rec_btn.enable()
            ref_rec_btn.props(remove='loading')
            record_btn.enable()
            set_nav(True)

    async def on_unlink_reference() -> None:
        """Drop the saved reference for this sentence; comparison falls back
        to the model read / Jev labels until a new reference is recorded."""
        if recording["on"] or not sentences:
            return
        sentence = sentences[current["i"]]
        token = gen["n"]
        unlink_ref_btn.disable()
        unlink_ref_btn.props('loading')
        try:
            removed = await run.io_bound(reference.remove, sentence)
            if stale(token):
                return
            if removed:
                # Saved-reference wavs are never temp files: nothing to unlink
                # on disk beyond what reference.remove deleted. Drop the
                # reference-derived state so nothing stale survives.
                for k in ("ref_ts", "ref_st", "ref_arrows", "ref_gaps",
                          "ref_slopes", "ref_energy", "ref_dir", "groups",
                          "labels", "stress", "ref_wav"):
                    result_state.pop(k, None)
                ref_rec_btn.text = "record reference"
                ui.notify("reference unlinked — comparison falls back to the "
                          "model read / Jev labels", type="positive")
                _paint_result(sentence)
                background_tasks.create(refresh_ref_arrows())
            else:
                ui.notify("no saved reference for this sentence", type="warning")
        except Exception as e:
            ui.notify(f"unlink failed: {e}", type="negative")
        finally:
            unlink_ref_btn.props(remove='loading')
            if not stale(token) and reference.has(sentences[current["i"]]):
                unlink_ref_btn.enable()

    async def on_hear_take() -> None:
        """Play the learner's own last recording back, no analysis attached."""
        wav = result_state.get("user_wav")
        if not wav:
            return
        hear_user_btn.disable()
        hear_user_btn.props('loading')
        try:
            await run.io_bound(record.play_wav, wav)
        except Exception as e:
            ui.notify(f"playback failed: {e}", type="negative")
        finally:
            hear_user_btn.props(remove='loading')
            if result_state.get("user_wav"):
                hear_user_btn.enable()

    async def on_hear_reference() -> None:
        if not sentences:
            return
        sentence = sentences[current["i"]]
        token = gen["n"]
        hear_ref_btn.disable()
        hear_ref_btn.props('loading')
        try:
            if reference.has(sentence):
                status.set_text("playing your saved reference…")
                await run.io_bound(record.play_wav, reference.path_for(sentence))
            elif _tts_allowed(settings, options):
                status.set_text(f"no saved reference — playing {read_label}…")
                await run.io_bound(models.speak, sentence)
            elif not settings.tts:
                status.set_text(NO_VOICE)
                ui.notify(NO_VOICE, type="warning")
                return
            else:
                status.set_text("read-back is off — record a reference to hear this sentence")
                return
            if not stale(token):
                status.set_text("")
        except Exception as e:
            ui.notify(f"playback failed: {e}", type="negative")
        finally:
            hear_ref_btn.props(remove='loading')
            hear_ref_btn.enable()

    async def refresh_ref_arrows() -> None:
        """Reference melody / flow from a human wav, or a model read if TTS is on.

        Captures the navigation token and returns before any UI write or unlink
        once the sentence has changed. Never deletes another sentence's wav.
        A deleted client (page closed mid-task) dies quietly.
        """
        try:
            await _refresh_ref_arrows_inner()
        except RuntimeError as e:
            if "has been deleted" in str(e):
                return
            raise

    async def _refresh_ref_arrows_inner() -> None:
        token = gen["n"]
        if not sentences:
            return
        sentence = sentences[current["i"]]
        human = reference.has(sentence)
        if stale(token):
            return

        ref_wav = None
        is_tmp = False

        # Labels come from a measured reference read when one exists (own voice
        # wins); otherwise the flow layer runs rule-only — punctuation and
        # syntax decide pause candidates, no charged calls per sentence view.
        need_audio = human or options["ref"] or options["melody"]
        if human:
            ref_wav = reference.path_for(sentence)
        elif need_audio and _tts_allowed(settings, options):
            prev_status = status.text
            done = {"now": False}

            async def _poll_stage() -> None:
                while not done["now"]:
                    stage = models.tts_load_stage()
                    if stage:
                        status.set_text(stage)
                    await asyncio.sleep(0.4)

            # NOTE: no ui.timer here — background tasks may not create UI
            # elements; a plain task that only updates the existing label is
            # the legal shape.
            poll_task = asyncio.create_task(_poll_stage())
            try:
                ref_wav = await run.io_bound(models.synthesize, sentence)
            finally:
                done["now"] = True
                await asyncio.gather(poll_task, return_exceptions=True)
                status.set_text(prev_status)
            is_tmp = True
            if stale(token):
                Path(ref_wav).unlink(missing_ok=True)
                return
        elif need_audio:
            if not settings.tts and not stale(token):
                status.set_text(NO_VOICE)
            return

        if ref_wav is None:
            # Flow-only, no reference audio: rule-based groups, and the
            # post-record score falls back to its rule path.
            result_state.pop("labels", None)
            result_state["groups"] = _groups_from_marks(
                sentence, flow.boundaries(sentence))
            words_ts = result_state.get("words_ts")
            spoken_j = result_state.get("spoken_j")
            alignment = result_state.get("alignment")
            if options["flow"] and alignment is not None and words_ts is not None \
                    and spoken_j is not None:
                fr = await run.io_bound(
                    flow.score, words_ts, sentence, spoken_j, None, False, None)
                if stale(token):
                    return
                result_state["marks"] = fr["marks"]
                flow_label.set_text(fr["summary"])
            if old_tmp := result_state.get("ref_wav"):
                if result_state.get("ref_wav_is_tmp"):
                    Path(old_tmp).unlink(missing_ok=True)
            status.set_text("")
            return

        origin = "human" if human else "engine"
        r = await run.io_bound(melody.reference_arrows, sentence, ref_wav, origin)
        if stale(token):
            if is_tmp:
                Path(ref_wav).unlink(missing_ok=True)
            return

        ref_gaps = flow.ref_gaps_from_ts(r["ts"], len(sentence.split()))
        groups = flow.link_groups(sentence, ref_gaps)
        result_state["groups"] = groups
        # One labeling pass, cached: the pre-speech guide and the post-record
        # score must agree. Re-deriving labels per pass let a threshold drift
        # flip a pause (|) into a rule-based link (‿) after speech.
        measured = flow.marks_from_ref(sentence, ref_gaps, human)
        result_state["labels"] = {i: measured.get(i, "none")
                                  for i in range(len(sentence.split()) - 1)}
        label_cache[(sentence, human)] = result_state["labels"]
        words_ts = result_state.get("words_ts")
        spoken_j = result_state.get("spoken_j")
        alignment = result_state.get("alignment")
        fr = None
        if (options["flow"] and alignment is not None and words_ts is not None
                and spoken_j is not None):
            fr = await run.io_bound(
                flow.score, words_ts, sentence, spoken_j, ref_gaps, human,
                result_state.get("labels"))
            if stale(token):
                if is_tmp:
                    Path(ref_wav).unlink(missing_ok=True)
                return

        if stale(token):
            if is_tmp:
                Path(ref_wav).unlink(missing_ok=True)
            return

        old_tmp = result_state.get("ref_wav") if result_state.get("ref_wav_is_tmp") else None
        result_state["ref_wav"] = str(ref_wav)
        result_state["ref_wav_is_tmp"] = is_tmp
        result_state["ref_arrows"] = r["arrows"]
        result_state["ref_ts"] = r["ts"]
        result_state["ref_st"] = r["st"]
        result_state["ref_slopes"] = r.get("slopes")
        result_state["ref_energy"] = r.get("energy")
        result_state["ref_gaps"] = ref_gaps
        result_state["groups"] = groups
        # Label from the saved reference, never from "a path was passed".
        src = "your saved reference" if human else read_label

        if alignment is None:
            # Repaint the guide only when a human reference was measured — its
            # breaths improve on the rule scaffold. A model read's measured
            # labels must NOT repaint the guide (kokoro doesn't breathe; the
            # rule marks would vanish). They still feed post-record scoring.
            if options["flow"] and human:
                guide = {i: {"kind": k, "ok": None}
                         for i, k in (result_state.get("labels") or {}).items()
                         if k != "none"}
                if stale(token):
                    return
                passage_view.clear()
                with passage_view:
                    colored_passage(sentence, None, on_word=on_word_click,
                                    marks=guide if options["guide"] else None)
            if old_tmp and old_tmp != str(ref_wav):
                Path(old_tmp).unlink(missing_ok=True)
            return

        marks = result_state.get("marks")
        if fr is not None:
            result_state["marks"] = fr["marks"]
            marks = fr["marks"]
            flow_label.set_text(fr["summary"])
        if stale(token):
            return
        _stress_marks(sentence)
        _paint_result(sentence)
        if options["melody"] and options["ref"]:
            status.set_text(
                f"small gray arrows = where {src} differs — click a word to hear both")
        ch = result_state.get("chart")
        if options["melody"] and ch:
            ch["ref"] = [r["st"].get(i) for i in range(len(ch["words"]))]
            pitch_chart.refresh()
        if options["melody"]:
            result_state["ref_dir"] = r.get("arrows") or {}
            _show_rhythm(sentence)
        if old_tmp and old_tmp != str(ref_wav):
            Path(old_tmp).unlink(missing_ok=True)

    async def on_read_all() -> None:
        import re
        text = passage_input.value.strip()
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not parts:
            ui.notify("nothing to read", type="warning")
            return
        read_all_btn.disable()
        read_all_btn.props('loading')
        try:
            for sentence in parts:
                if reference.has(sentence):
                    status.set_text("playing your saved reference…")
                    await run.io_bound(record.play_wav, reference.path_for(sentence))
                elif _tts_allowed(settings, options):
                    status.set_text(f"reading {read_label}…")
                    await run.io_bound(models.speak, sentence)
                elif not settings.tts:
                    status.set_text(NO_VOICE)
                    ui.notify(NO_VOICE, type="warning")
                    return
                else:
                    status.set_text("read-back is off")
                    return
            status.set_text("")
        except Exception as e:
            ui.notify(f"read-back failed: {e}", type="negative")
        finally:
            read_all_btn.props(remove='loading')
            read_all_btn.enable()

    def show_sentence() -> None:
        if not sentences:
            return
        gen["n"] += 1
        i = current["i"]
        # Drop only this view's temp wavs. A human reference is never unlinked.
        # In-flight work for the previous token must not unlink after this.
        doomed = []
        for k in ("user_wav", "ref_wav"):
            p = result_state.get(k)
            if p and result_state.get(k + "_is_tmp", k == "user_wav"):
                doomed.append(p)
        result_state.clear()
        for p in doomed:
            Path(p).unlink(missing_ok=True)

        jump_guard["on"] = True
        jump_select.options = {n: f"{n + 1}. {s[:40]}" for n, s in enumerate(sentences)}
        jump_select.value = i
        jump_guard["on"] = False
        if options["melody"]:
            pitch_chart.refresh()
        progress.set_text(f"sentence {i + 1}/{len(sentences)}")
        sentence = sentences[i]
        ref_rec_btn.text = ("re-record reference" if reference.has(sentence)
                            else "record reference")
        if reference.has(sentence):
            unlink_ref_btn.enable()
        else:
            unlink_ref_btn.disable()
        transcript_label.set_text("")
        tune_label.set_text("")
        rhythm_label.set_text("")
        hear_user_btn.disable()
        flow_label.set_text("")
        extra_label.set_text("")
        jev_label.set_text("")
        phon_label.set_text("")
        rule = flow.boundaries(sentence)
        # The pre-speech scaffold is human breaths when we have them, otherwise
        # punctuation/syntax rules — never kokoro's word spacing, which is why
        # the breath icons used to vanish once the model read was measured.
        human_labels = (label_cache.get((sentence, True))
                        if reference.has(sentence) else None)
        if human_labels is not None:
            result_state["labels"] = human_labels
        result_state["groups"] = _groups_from_marks(sentence, human_labels or rule)
        mark_source = human_labels or rule
        guide = ({n: {"kind": k, "ok": None} for n, k in mark_source.items()
                  if k != "none"}
                 if options["flow"] and options["guide"] else None)
        passage_view.clear()
        with passage_view:
            colored_passage(sentence, None, on_word=on_word_click, marks=guide)
        status.set_text("listen to the model, then read — marks appear after your attempt")
        background_tasks.create(refresh_ref_arrows())

    async def on_record() -> None:
        if recording["on"] or not sentences:
            if not sentences:
                ui.notify("load a passage first", type="warning")
            return
        sentence = sentences[current["i"]]
        token = gen["n"]
        human = reference.has(sentence)
        recording["on"] = True
        set_nav(False)
        record_btn.disable()
        record_btn.props('loading')
        stop_btn.classes(remove='hidden')
        stop_btn.enable()
        ref_rec_btn.disable()
        hear_user_btn.disable()
        wav = None
        try:
            status.set_text("🎙 listening… read the sentence aloud")
            wav = await run.io_bound(record.record_utterance)
            if stale(token):
                Path(wav).unlink(missing_ok=True)
                return

            status.set_text("transcribing…")
            t0 = time.time()
            asr = await run.io_bound(models.transcribe_detailed, wav)
            if stale(token):
                Path(wav).unlink(missing_ok=True)
                return
            transcript = asr["text"]
            asr_ms = (time.time() - t0) * 1000
            transcript_label.set_text(f"you said ({asr_ms:.0f}ms ASR): {transcript}")

            alignment = await run.io_bound(diffmod.align, sentence, transcript)
            if stale(token):
                Path(wav).unlink(missing_ok=True)
                return

            arrows = None
            marks = None
            user_ts, user_st = {}, {}
            user_energy, user_slopes = {}, {}
            if asr["words"] and options["melody"]:
                status.set_text("scoring melody…")
                wp = await run.io_bound(melody.word_pitch, wav, asr["words"])
                if stale(token):
                    Path(wav).unlink(missing_ok=True)
                    return
                arrows = {}
                for w in alignment.words:
                    if w.spoken_j is not None and w.spoken_j < len(wp):
                        e = wp[w.spoken_j]
                        arrows[w.index] = e["direction"]
                        user_ts[w.index] = {"start": e["start"], "end": e["end"]}
                        user_st[w.index] = e["st"]
                        user_energy[w.index] = e.get("energy_db_rel")
                        user_slopes[w.index] = e.get("slope")
                tune = melody.verdict(wp, sentence)
                tune_label.set_text(tune or "")
                result_state["user_dir"] = arrows
                result_state["user_ts"] = user_ts
                result_state["user_energy"] = user_energy
                result_state["user_slopes"] = user_slopes
                _show_rhythm(sentence)
                result_state["chart"] = {
                    "words": sentence.split(),
                    "you": [user_st.get(n) for n in range(len(sentence.split()))],
                    "ref": [(result_state.get("ref_st") or {}).get(n)
                            for n in range(len(sentence.split()))],
                }
                pitch_chart.refresh()
            elif asr["words"]:
                tune_label.set_text("")

            if options["flow"] and asr["words"]:
                spoken_j = {w.index: w.spoken_j for w in alignment.words
                            if w.spoken_j is not None}
                result_state["words_ts"] = asr["words"]
                result_state["spoken_j"] = spoken_j
                fr = await run.io_bound(
                    flow.score, asr["words"], sentence, spoken_j,
                    result_state.get("ref_gaps"), human,
                    result_state.get("labels"))
                if stale(token):
                    Path(wav).unlink(missing_ok=True)
                    return
                marks = fr["marks"]
                flow_label.set_text(fr["summary"])
            else:
                flow_label.set_text("")

            if stale(token):
                Path(wav).unlink(missing_ok=True)
                return

            if user_ts:
                result_state["user_ts"] = user_ts
                result_state["user_st"] = user_st
            result_state.update(alignment=alignment, arrows=arrows, marks=marks)
            _paint_result(sentence)

            if alignment.spoken_words:
                extra_label.set_text(
                    "extra words: " + " ".join(alignment.spoken_words))
            else:
                extra_label.set_text("")

            if options["phonemes"]:
                status.set_text("scoring phonemes…")
                sounds = await run.io_bound(phonemes.score_words, wav, sentence.split())
                if stale(token):
                    Path(wav).unlink(missing_ok=True)
                    return
                ns = phonemes.notes(sounds)
                if ns:
                    phon_label.set_text("sound notes: " + "; ".join(ns))
                else:
                    phon_label.set_text("sound: every word close to the model's phones")
            else:
                phon_label.set_text("")

            if options["jev"]:
                suspects = suspect_payload(sentence, alignment)
                if not jev.configured():
                    jev_label.set_text(
                        "Jev on but not configured — set jev.api_key in "
                        "coach.yaml or the JEV_API_KEY env (see README)")
                elif suspects:
                    # 3-5s provider round trip — run it behind the read-back
                    # playback instead of blocking on it. Verdict paints when
                    # it lands; stale() guards the label write.
                    async def ask_jev() -> None:
                        try:
                            verdicts = await run.io_bound(
                                jev.judge_sentence, sentence, transcript, suspects)
                            if stale(token):
                                return
                            jev_label.set_text(_jev_line(verdicts, suspects))
                        except Exception as e:
                            if not stale(token):
                                ui.notify(f"Jev failed: {e}", type="negative")
                    background_tasks.create(ask_jev())
                else:
                    jev_label.set_text("clean sentence — no Jev call needed")
            else:
                jev_label.set_text("")

            if human:
                status.set_text("playing your saved reference…")
                await run.io_bound(record.play_wav, reference.path_for(sentence))
            elif _tts_allowed(settings, options):
                status.set_text(f"read-back — {read_label}…")
                await run.io_bound(models.speak, sentence)
            elif not settings.tts:
                status.set_text(NO_VOICE)

            if stale(token):
                Path(wav).unlink(missing_ok=True)
                return
            status.set_text("done — record again or move to the next sentence")
            result_state["user_wav"] = str(wav)
            result_state["user_wav_is_tmp"] = True
            wav = None  # ownership moved to result_state
            hear_user_btn.enable()
            background_tasks.create(refresh_ref_arrows())
        except RuntimeError as e:
            ui.notify(str(e), type="warning")
            if not stale(token):
                status.set_text("recording failed")
        except Exception as e:
            ui.notify(f"error: {e}", type="negative")
            if not stale(token):
                status.set_text("failed")
        finally:
            if wav is not None and stale(token):
                Path(wav).unlink(missing_ok=True)
            recording["on"] = False
            record_btn.props(remove='loading')
            record_btn.enable()
            stop_btn.classes(add='hidden')
            stop_btn.disable()
            ref_rec_btn.enable()
            set_nav(True)

    def on_prev() -> None:
        if recording["on"] or not sentences:
            return
        current["i"] = (current["i"] - 1) % len(sentences)
        show_sentence()

    def on_next() -> None:
        if recording["on"] or not sentences:
            return
        current["i"] = (current["i"] + 1) % len(sentences)
        show_sentence()

    record_btn.on_click(on_record)
    stop_btn.on_click(lambda: record.stop_recording())
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    read_all_btn.on_click(on_read_all)
    ref_rec_btn.on_click(on_record_reference)
    hear_ref_btn.on_click(on_hear_reference)
    unlink_ref_btn.on_click(on_unlink_reference)
    hear_user_btn.on_click(on_hear_take)
    load_btn.on_click(load_passage_text)
    show_sentence()


def main() -> None:
    settings = _settings()

    from nicegui import app

    # app is the FastAPI application in NiceGUI 3.x
    async def _extension_passage(request: Request) -> dict:
        """Accept page text from the fluency coach browser extension."""
        try:
            body = await request.json()
        except Exception:
            return {"error": "json body required"}
        text = str(body.get("text") or "").strip()
        if not text:
            return {"error": "empty text"}
        text = text[:20000]
        token = secrets.token_hex(8)
        _extension_passages[token] = text
        while len(_extension_passages) > 50:
            _extension_passages.pop(next(iter(_extension_passages)))
        host = "127.0.0.1" if settings.host in ("0.0.0.0", "::", "") else settings.host
        url = f"http://{host}:{int(settings.port)}/?passage={token}"
        return {"token": token, "url": url}

    app.add_api_route("/extension/passage", _extension_passage, methods=["POST"])

    ui.run(title="fluency coach", host=settings.host, port=int(settings.port),
           reload=False)


if __name__ == "__main__":
    main()
