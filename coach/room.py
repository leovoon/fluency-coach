"""The shadowing practice surface; audio stays on the local server."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import re
import tempfile
import time

from nicegui import run, ui

from . import models, record, reference, shadowing as view

log = logging.getLogger(__name__)


def split_sentences(text: str | None) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def build(settings, text: str | None, templates: dict[str, str],
          preview: bool = False, theme: str = "paper") -> None:
    sentences = split_sentences(text if text is not None else view.PASSAGE)
    state = {"index": 2 if preview and text is None else 0, "busy": False,
             "recording": False, "closed": False, "take": None, "preview": preview,
             "transcript": None, "dark": theme == "candlelit"}
    client = ui.context.client
    ui.page_title("shadowing partner")
    ui.add_css(view.STYLE)
    ui.colors(primary="#587575")
    ui.add_css('.room .q-btn.text-primary { color:var(--ink)!important; } '
               '.room .q-btn.read-label, .room-head .q-btn.text-primary { color:var(--muted)!important; } '
               '.room .q-btn.mic.text-primary { color:var(--slate)!important; }')

    def sentence():
        return sentences[state["index"]] if sentences else ""

    def label():
        voice = models.model_voice_path()
        return view.source_label(reference.has(sentence()), settings.tts_engine,
                                 bool(voice and voice.exists()))

    def discard_take():
        if state["take"]:
            Path(state["take"]).unlink(missing_ok=True)
        state["take"] = None
        state["transcript"] = None

    def disconnected():
        state["closed"] = True
        if state["recording"]:
            record.stop_recording()
        if not state["busy"]:
            discard_take()

    client.on_disconnect(disconnected)
    client.on_connect(lambda: state.update(closed=False))

    def lock(busy):
        state["busy"] = busy
        for control in (play_btn, again_btn, model_button, take_button, setup,
                        record_text, mic):
            control.set_enabled(not busy)
        if busy and state["recording"]:
            mic.enable()
            record_text.enable()
        dots.refresh()
        if not busy:
            take_button.set_enabled(bool(state["take"]))

    def reset():
        if state["busy"]:
            return
        state["preview"] = False
        discard_take()
        paint()
        status.set_text("A fresh breath. Read whenever you're ready.")

    def go(index):
        if state["busy"] or not sentences:
            return
        state["index"] = index
        reset()

    def toggle_theme():
        state["dark"] = not state["dark"]
        root.classes(toggle="candlelit")
        theme_button.props(f'aria-pressed={str(state["dark"]).lower()}')
        ui.run_javascript(
            f"localStorage.setItem('shadowing-theme', '{'candlelit' if state['dark'] else 'paper'}')")

    async def restore_theme():
        if theme == "candlelit":
            return
        saved = await ui.run_javascript("localStorage.getItem('shadowing-theme')")
        if saved == "candlelit" and not state["dark"]:
            toggle_theme()

    @ui.refreshable
    def dots():
        # A moving window keeps long imported passages usable.
        start = min(max(0, state["index"] - 5), max(0, len(sentences) - 12))
        with ui.row().classes("dots").props('role=navigation aria-label="Sentences"'):
            for i in range(start, min(start + 12, len(sentences))):
                button = ui.button("", on_click=lambda _, i=i: go(i)).props(
                    f'flat round aria-label="Sentence {i + 1}"').classes(
                    "dot " + ("current" if i == state["index"] else
                              "past" if i < state["index"] else ""))
                if i == state["index"]:
                    button.props('aria-current=step')
                button.set_enabled(not state["busy"])
        ui.label(f"{state['index'] + 1:02d} / {len(sentences):02d}" if sentences
                 else "NO PASSAGE").classes("position")

    def paint():
        model_line.set_content(view.model_html(sentence(), .4 if state["preview"] else 0))
        model_button.set_text(
            "MODEL READ · CLONED FROM YOUR TEACHER" if state["preview"] else label())
        contour.set_content(view.SAMPLE_CONTOUR if state["preview"] else "")
        if state["preview"]:
            learner.set_content(view.learner_html(sentence(), view.SAMPLE_READ))
            learner.classes(remove="learner-empty")
        elif state["transcript"] is not None:
            learner.set_content(view.learner_html(sentence(), state["transcript"]))
            learner.classes(remove="learner-empty")
        else:
            learner.set_content("Your voice will find its place here.")
            learner.classes(add="learner-empty")
        sample_note.set_visibility(state["preview"])
        take_button.set_enabled(bool(state["take"]) and not state["busy"])
        dots.refresh()

    async def get_model():
        if reference.has(sentence()):
            return reference.path_for(sentence()), False
        if not settings.tts or settings.tts_engine == "none":
            raise RuntimeError("Save a reference recording in practice setup to hear this sentence.")
        status.set_text("Preparing the teaching voice…")
        path = await run.io_bound(models.synthesize, sentence())
        return path, True

    async def play_audio(path, is_model):
        import soundfile as sf
        duration = await run.io_bound(lambda: sf.info(str(path)).duration)
        span = None
        if not is_model:
            span = await run.io_bound(view.speech_span, path)
        span = span or (0, duration)

        def highlight(fraction):
            if is_model:
                model_line.set_content(view.model_html(sentence(), fraction))
            else:
                learner.set_content(view.learner_html(
                    sentence(), state["transcript"] or "", fraction))

        started = time.monotonic()
        highlight(0 if is_model else -1)
        task = asyncio.create_task(run.io_bound(record.play_wav, path))
        try:
            while not task.done():
                if not state["closed"]:
                    elapsed = time.monotonic() - started
                    fraction = (min(.99, elapsed / max(.1, duration)) if is_model
                                else view.speech_progress(elapsed, span))
                    highlight(fraction)
                await asyncio.sleep(.06)
            await task
            if not state["closed"]:
                highlight(1)
        except Exception:
            if not state["closed"]:
                if is_model:
                    highlight(0)
                else:
                    learner.set_content(view.learner_html(sentence(), state["transcript"] or ""))
            raise
        finally:
            # Never unlink a temporary recording while an output stream is reading it.
            if not task.done():
                await task

    async def listen(which="both"):
        if state["busy"] or not sentences:
            return
        state["preview"] = False
        paint()
        lock(True)
        model_path, temporary = None, False
        try:
            paths = []
            if which != "take":
                model_path, temporary = await get_model()
                if state["closed"]:
                    return
                model_button.set_text(label())
                try:
                    contour.set_content(await run.io_bound(view.contour_svg, model_path))
                except Exception:
                    log.debug("Pitch contour unavailable", exc_info=True)
                    contour.set_content("")
                paths.append(model_path)
            if which != "model" and state["take"]:
                paths.append(state["take"])

            async def before(_i, path):
                status.set_text("Listening to the model / reference · word timing is approximate"
                                if path == model_path else
                                "Listening to your read · word timing is approximate")
            await view.play_sequence(
                paths, lambda path: play_audio(path, path == model_path),
                alive=lambda: not state["closed"], before=before)
            if not state["closed"]:
                status.set_text("Model, then you. Listen for what feels different."
                                if len(paths) == 2 else "Leave a little space, then try it in your voice.")
        except Exception as exc:
            if not state["closed"]:
                status.set_text(f"Couldn't play this read. {exc}")
        finally:
            if temporary and model_path:
                Path(model_path).unlink(missing_ok=True)
            if state["closed"]:
                discard_take()
            else:
                lock(False)

    async def capture(kind="take"):
        if state["recording"]:
            record.stop_recording()
            status.set_text("Finishing the recording…")
            mic.disable()
            record_text.disable()
            return
        if state["busy"] or not sentences:
            return
        state["preview"] = False
        paint()
        state["recording"] = True
        lock(True)
        mic.props('icon=stop aria-label="Stop recording"')
        mic.classes(add="listening")
        wave.classes(add="listening")
        record_text.set_text("stop")
        status.set_text("Listening on this Mac. Read naturally; a pause will finish your take.")
        wav = None
        try:
            wav = await run.io_bound(record.record_utterance)
            state["recording"] = False
            mic.disable()
            record_text.disable()
            mic.classes(remove="listening")
            wave.classes(remove="listening")
            if state["closed"]:
                return
            if kind == "reference":
                await run.io_bound(reference.save, sentence(), wav)
                model_button.set_text(label())
                status.set_text("Reference saved. It will play before your take.")
            elif kind == "voice":
                result = await run.io_bound(models.save_voice, voice_name.value or "teacher", wav)
                gallery.refresh()
                model_button.set_text(label())
                status.set_text(f"Teaching voice saved: {result['name']}.")
            else:
                status.set_text("Listening back to your words…")
                result = await run.io_bound(models.transcribe_detailed, wav)
                if state["closed"]:
                    return
                discard_take()
                state["take"], wav = wav, None
                state["transcript"] = result["text"].strip()
                paint()
                status.set_text("Your read is ready. Play to hear the model, then you."
                                if state["transcript"] else
                                "No words were recognized. You can listen to the take or record again.")
        except Exception as exc:
            if not state["closed"]:
                status.set_text(f"Couldn't finish this recording. {exc}")
        finally:
            if wav:
                Path(wav).unlink(missing_ok=True)
            state["recording"] = False
            if state["closed"]:
                discard_take()
            else:
                mic.props('icon=mic_none aria-label="Record your read"')
                mic.classes(remove="listening")
                wave.classes(remove="listening")
                record_text.set_text("record")
                lock(False)

    def load_text(value):
        if state["busy"]:
            return
        new = split_sentences(value)
        if not new:
            status.set_text("Add a sentence before loading a passage.")
            return
        sentences[:] = new
        state["index"] = 0
        reset()
        status.set_text("Your passage is ready. Listen, then make it your own.")

    async def upload_passage(event):
        if state["busy"]:
            return
        text = (await event.file.read()).decode("utf-8", "replace")
        passage_input.set_value(text)
        load_text(text)

    async def upload_voice(event):
        if state["busy"]:
            return
        lock(True)
        fd, name = tempfile.mkstemp(suffix=Path(event.file.name).suffix or ".wav")
        os.close(fd)
        path = Path(name)
        try:
            path.write_bytes(await event.file.read())
            result = await run.io_bound(models.save_voice, voice_name.value or "teacher", path)
            gallery.refresh()
            model_button.set_text(label())
            status.set_text(f"Teaching voice saved: {result['name']}.")
        except Exception as exc:
            status.set_text(f"Couldn't use this voice clip. {exc}")
        finally:
            path.unlink(missing_ok=True)
            lock(False)

    async def voice_action(action, name):
        if state["busy"]:
            return
        lock(True)
        try:
            if action == "activate":
                await run.io_bound(models.activate_voice, name)
            elif action == "delete":
                await run.io_bound(models.delete_voice, name)
            else:
                await run.io_bound(record.play_wav, name)
            gallery.refresh()
            model_button.set_text(label())
        except Exception as exc:
            status.set_text(f"Couldn't {action} this voice. {exc}")
        finally:
            lock(False)

    @ui.refreshable
    def gallery():
        voices = models.list_voices()
        if not voices:
            ui.label("No teaching voices saved yet.")
        for voice in voices:
            with ui.row().classes("items-center gap-2"):
                ui.button(voice["name"] + (" · active" if voice["active"] else ""),
                          on_click=lambda _, n=voice["name"]: voice_action("activate", n)).props("flat")
                ui.button("listen", on_click=lambda _, p=voice["path"]: voice_action("play", p)).props("flat")
                ui.button("delete", on_click=lambda _, n=voice["name"]: voice_action("delete", n)).props("flat")

    def remove_reference():
        if state["busy"]:
            return
        reference.remove(sentence())
        model_button.set_text(label())
        contour.set_content("")
        status.set_text("Reference unlinked. Your teaching voice will be used for model reads.")

    with ui.element("main").classes("room" + (" candlelit" if state["dark"] else "")) as root:
        ui.html("<!-- THESIS: A room for listening, without scores. OWN-WORLD: paper, ochre, slate, Fraunces. "
                "STORY: listen, record, compare. FIRST VIEWPORT: progress, two readings, centered microphone. "
                "FORM: user-pinned practice room. FINISH: unreviewed and undocumented is unfinished; "
                "this build ends with the finish review, the verdict, and DESIGN.md -->", sanitize=False)
        with ui.element("header").classes("room-head"):
            ui.label("shadowing partner").classes("room-name")
            with ui.button(on_click=toggle_theme).props(
                    f'flat round aria-label="Toggle candlelit theme" aria-pressed={str(state["dark"]).lower()}').classes("theme-switch").mark("theme-switch") as theme_button:
                ui.icon("light_mode").classes("theme-icon theme-sun").props('aria-hidden=true')
                ui.icon("dark_mode").classes("theme-icon theme-moon").props('aria-hidden=true')
                ui.tooltip("Switch paper / candlelit")
        with ui.column().classes("practice"):
            dots()
            with ui.element("section").classes("model-section").props('aria-label="Model reading"'):
                model_button = ui.button(label(), on_click=lambda: listen("model")).props(
                    'flat aria-label="Play model or saved reference"').classes("read-label")
                contour = ui.html("", sanitize=False).classes("contour")
                model_line = ui.html("", sanitize=False).classes("model-line")
            with ui.element("div").classes("breath-space"):
                wave = ui.html(view.WAVE, sanitize=False).classes("wave")
                mic = ui.button(icon="mic_none", on_click=capture).props(
                    'flat round aria-label="Record your read"').classes("mic")
            with ui.element("section").classes("learner-section").props('aria-label="Your reading"'):
                take_button = ui.button("YOUR READ", on_click=lambda: listen("take")).props(
                    'flat aria-label="Play your read"').classes("read-label")
                learner = ui.html("", sanitize=False).classes("learner-line")
            sample_note = ui.label("Design preview · sample reading, not a recorded attempt").classes("sample-note")
            status = ui.label("Listen first. There is no need to hurry.").classes("room-status").props(
                'role=status aria-live=polite')
            with ui.row().classes("actions"):
                play_btn = ui.button("play", on_click=listen).props('flat aria-keyshortcuts="P"')
                record_text = ui.button("record", on_click=capture).props("flat")
                again_btn = ui.button("again", on_click=reset).props("flat")
            with ui.element("div").classes("room-note"):
                with ui.button(icon="keyboard").props(
                        'flat round aria-label="Keyboard shortcuts" aria-haspopup=menu'):
                    with ui.menu().classes("room-popup shortcut-menu"):
                        ui.label("Keyboard shortcuts").classes("shortcut-title")
                        for key, action in (("P", "Play model, then you"),
                                            ("Space", "Record / stop"),
                                            ("← / →", "Previous / next sentence")):
                            with ui.row().classes("shortcut-row"):
                                ui.label(key).classes("shortcut-key")
                                ui.label(action)
        with ui.expansion("practice setup", icon="tune").classes("setup") as setup:
            with ui.column().classes("setup-body"):
                with ui.element("section").classes("setup-section").props('aria-label="Passage setup"'):
                    ui.label("Passage").classes("setup-title")
                    ui.label("Choose something you want to say. Practice it one sentence at a time.").classes("setup-help")
                    passage_input = ui.textarea(value=text if text is not None else view.PASSAGE,
                                                label="Passage text").props("outlined autogrow").classes("passage-editor")
                    with ui.row().classes("setup-actions"):
                        ui.button("load passage", on_click=lambda: load_text(passage_input.value)).props("unelevated").classes("setup-primary")
                    if templates:
                        with ui.row().classes("template-picker"):
                            template_select = ui.select(list(templates), label="Practice passage").props(
                                'outlined popup-content-class="room-popup"').classes("template-select")
                            def use_template():
                                if template_select.value and not state["busy"]:
                                    passage_input.set_value(templates[template_select.value])
                                    load_text(passage_input.value)
                            ui.button("use passage", on_click=use_template).props("flat")
                    with ui.expansion("Or upload a text file", icon="upload_file").classes("setup-secondary"):
                        ui.upload(label="Upload passage", auto_upload=True, on_upload=upload_passage).props("accept=.txt")
                    with ui.row().classes("setup-actions"):
                        ui.button("previous sentence", on_click=lambda: go((state["index"] - 1) % len(sentences))).props("flat")
                        ui.button("next sentence", on_click=lambda: go((state["index"] + 1) % len(sentences))).props("flat")
                with ui.element("section").classes("setup-section").props('aria-label="Sentence reference setup"'):
                    ui.label("Sentence reference").classes("setup-title")
                    ui.label("Save a human reading of this sentence. It will play instead of the model.").classes("setup-help")
                    with ui.row().classes("setup-actions"):
                        ui.button("record reference", on_click=lambda: capture("reference")).props("flat")
                        ui.button("unlink reference", on_click=remove_reference).props("flat")
                with ui.element("section").classes("setup-section").props('aria-label="Teaching voice setup"'):
                    ui.label("Teaching voice").classes("setup-title")
                    ui.label("A 2–30 second clip for the model voice. Use a voice you have permission to clone.").classes("setup-help")
                    with ui.row().classes("voice-controls"):
                        voice_name = ui.input("Voice name", value="teacher").props("outlined")
                        ui.button("record voice clip", on_click=lambda: capture("voice")).props("flat")
                    with ui.expansion("Or upload a voice clip", icon="upload_file").classes("setup-secondary"):
                        ui.upload(label="Upload voice clip", auto_upload=True, on_upload=upload_voice).props(
                            "accept=.wav,.flac,.ogg,.aiff,.mp3")
                    gallery()
                ui.label(f"Audio stays on this Mac · ASR: {settings.asr_engine} · text comparison only").classes("setup-footnote")
    async def on_key(event):
        if (not event.action.keydown or event.modifiers.ctrl or
                event.modifiers.meta or event.modifiers.alt):
            return
        if event.key.name.lower() == "p":
            await listen()
        elif event.key.space:
            await capture()
        elif event.key.arrow_left and sentences:
            go((state["index"] - 1) % len(sentences))
        elif event.key.arrow_right and sentences:
            go((state["index"] + 1) % len(sentences))
    ui.keyboard(on_key=on_key, repeating=False, ignore=["input", "textarea", "select", "button"])
    ui.run_javascript("""
        window.addEventListener('keydown', e => {
            const t = e.target;
            if (t.closest('input, textarea, select, button, [contenteditable=true]')) return;
            if (!e.ctrlKey && !e.metaKey && !e.altKey &&
                ['Space', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
        });
    """)
    # Quasar's generated text-primary rule would override the two theme palettes.
    for element in list(client.elements.values()):
        if isinstance(element, ui.button):
            element.props(remove="color")
    ui.timer(.1, restore_theme, once=True)
    paint()
