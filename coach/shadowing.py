"""Presentation and audio helpers for the quiet shadowing room."""
from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
import re

SAMPLE = "We could take the quiet road home tonight."
SAMPLE_READ = "We can take the road home."
PASSAGE = (
    "There is no need to hurry. "
    "Let the afternoon unfold a little more slowly. "
    + SAMPLE + " "
    "The light is still warm on the windows. "
    "Someone has left the garden gate open. "
    "I can hear the wind moving through the trees. "
    "Shall we stop for a cup of tea? "
    "There is a small place just around the corner. "
    "You were telling me about your journey. "
    "I would like to hear the rest of the story. "
    "Take your time with the words. "
    "We have nowhere else to be."
)
TOKEN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def source_label(human: bool, engine: str, has_voice: bool) -> str:
    if human:
        return "SAVED REFERENCE · HUMAN READ"
    if engine == "pocket" and has_voice:
        return "MODEL READ · CLONED FROM YOUR TEACHER"
    return "MODEL READ · SYNTHETIC VOICE"


def model_html(sentence: str, fraction: float = 0) -> str:
    words = sentence.split()
    count = int(max(0, min(1, fraction)) * len(words))
    return " ".join(
        f'<span class="{"spoken" if i < count else "upcoming"}">{escape(w)}</span>'
        for i, w in enumerate(words)
    )


def learner_html(sentence: str, transcript: str, fraction: float | None = None) -> str:
    """Keep changes accessible; optional playback progress counts heard words only."""
    target, spoken = TOKEN.findall(sentence), TOKEN.findall(transcript)
    word_indices = [i for i, token in enumerate(spoken) if re.match(r"\w", token)]
    # Negative progress means playback has not reached the detected speech yet.
    count = (min(len(word_indices), int(min(1, fraction) * len(word_indices)) + 1)
             if fraction is not None and fraction >= 0 else 0)
    heard = set(word_indices[:count])

    def spoken_token(index: int) -> str:
        text = escape(spoken[index])
        if fraction is None or not re.match(r"\w", spoken[index]):
            return text
        style = "learner-spoken" if index in heard else "learner-upcoming"
        return f'<span class="{style}">{text}</span>'

    pieces = []
    matcher = SequenceMatcher(a=[w.casefold() for w in target],
                              b=[w.casefold() for w in spoken], autojunk=False)
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag == "equal":
            pieces.extend(spoken_token(i) for i in range(c, d))
        elif tag == "delete":
            pieces.extend(f'<span class="skipped" tabindex="0" title="Not heard: {escape(w, quote=True)}" '
                          f'aria-label="Not heard: {escape(w, quote=True)}">{escape(w)}</span>'
                          for w in target[a:b])
        else:
            label = ("Heard instead of " + " ".join(target[a:b])) if tag == "replace" else "Additional words"
            pieces.append(f'<span class="substituted" tabindex="0" title="{escape(label, quote=True)}" '
                          f'aria-label="{escape(label, quote=True)}: {escape(" ".join(spoken[c:d]), quote=True)}">'
                          f'{" ".join(spoken_token(i) for i in range(c, d))}</span>')
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(pieces))


def speech_span(path: str | Path) -> tuple[float, float] | None:
    """Estimate the audible span, not word alignment; leave the WAV unchanged."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if not len(data):
        return None
    # Channel energy avoids cancelling opposite-phase stereo recordings.
    energy = np.mean(np.square(data.astype(float)), axis=1)
    size = max(1, round(rate * .02))
    starts = np.arange(0, len(energy), size)
    lengths = np.minimum(size, len(energy) - starts)
    rms = np.sqrt(np.add.reduceat(energy, starts) / lengths)
    peak = float(rms.max())
    if not np.isfinite(peak) or peak < .001:
        return None
    # Relative threshold keeps quiet takes usable; the floor rejects near-silence.
    active = np.flatnonzero(rms >= max(.001, peak * .08))
    if not len(active):
        return None
    start = max(0, (int(starts[active[0]]) / rate) - .02)
    end = min(len(data) / rate, (int(starts[active[-1]]) + size) / rate + .02)
    return start, end


def speech_progress(elapsed: float, span: tuple[float, float]) -> float:
    start, end = span
    if elapsed < start:
        return -1
    return min(1, max(0, (elapsed - start) / max(.02, end - start)))


def contour_svg(path: str | Path) -> str:
    """A measured, unscored pitch trace, independent of ASR word timings."""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if not data.size:
        return ""
    # Autocorrelation on short windows avoids loading an alignment model.
    stride = max(1, sr // 8000)
    data, rate = data[::stride], sr / stride
    size = int(rate * .04)
    hop = max(size, len(data) // 90)
    samples = []
    for start in range(0, len(data) - size, hop):
        frame = data[start:start + size].astype(float)
        frame -= frame.mean()
        if np.sqrt(np.mean(frame ** 2)) < .008:
            samples.append(None)
            continue
        corr = np.correlate(frame * np.hanning(size), frame * np.hanning(size), "full")[size - 1:]
        lo, hi = int(rate / 450), min(len(corr), int(rate / 65))
        lag = lo + int(np.argmax(corr[lo:hi]))
        samples.append(float(np.log2(rate / lag)) if corr[lag] > .3 * corr[0] else None)
    voiced = [v for v in samples if v is not None]
    if len(voiced) < 3:
        return ""
    low, high = np.percentile(voiced, [5, 95])
    spread = max(float(high - low), .25)
    parts, pen = [], False
    for i, value in enumerate(samples):
        if value is None:
            pen = False
            continue
        x = 4 + i / max(1, len(samples) - 1) * 632
        y = 49 - min(1, max(0, (value - low) / spread)) * 34
        parts.append(f'{"L" if pen else "M"} {x:.1f},{y:.1f}')
        pen = True
    trace = " ".join(parts)
    return (f'<svg viewBox="0 0 640 64" role="img" aria-label="Pitch contour of the model or reference recording">'
            f'<path d="{trace}" fill="none" stroke="currentColor" stroke-width="9" opacity=".12" stroke-linecap="round"/>'
            f'<path d="{trace}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>')


async def play_sequence(paths, play, alive=lambda: True, before=None):
    """One recording at a time; check cancellation between A and B."""
    for i, path in enumerate(paths):
        if not alive():
            return
        if before:
            await before(i, path)
        await play(path)
        if i + 1 < len(paths):
            await asyncio.sleep(.35)


STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,450&family=Manrope:wght@400;500;600&display=swap');
@property --paper { syntax:'<color>'; inherits:true; initial-value:#FAF7F2; }
@property --ink { syntax:'<color>'; inherits:true; initial-value:#302C27; }
@property --muted { syntax:'<color>'; inherits:true; initial-value:#777167; }
@property --ochre { syntax:'<color>'; inherits:true; initial-value:#9B712D; }
@property --slate { syntax:'<color>'; inherits:true; initial-value:#587575; }
@property --line { syntax:'<color>'; inherits:true; initial-value:#E4DED4; }
@property --button { syntax:'<color>'; inherits:true; initial-value:#FFFCF7; }
:root { --paper:#FAF7F2; --ink:#302C27; --muted:#777167; --ochre:#9B712D;
 --slate:#587575; --line:#E4DED4; --button:#FFFCF7; }
body:has(.candlelit) { --paper:#211E1A; --ink:#E9DFCF; --muted:#B2A797;
 --ochre:#D2AB67; --slate:#9BB4AE; --line:#413B32; --button:#302B24; color-scheme:dark; }
body { background:var(--paper); color:var(--ink); font-family:Manrope,sans-serif;
 transition:--paper 300ms ease-out, --ink 300ms ease-out, --muted 300ms ease-out,
 --ochre 300ms ease-out, --slate 300ms ease-out, --line 300ms ease-out, --button 300ms ease-out; }
.theme-switch { width:44px; height:44px; }
.theme-switch .theme-icon { position:absolute; font-size:20px; transition:opacity 300ms ease-out; }
.theme-switch .theme-sun { opacity:0; }
.theme-switch .theme-moon { opacity:1; }
.candlelit .theme-switch .theme-sun { opacity:1; }
.candlelit .theme-switch .theme-moon { opacity:0; }
.nicegui-content { padding:0!important; gap:0!important; }
.room { width:100%; min-height:100vh; background:var(--paper); color:var(--ink); }
.room ::selection { background:var(--ochre); color:var(--paper); }
.room :focus-visible { outline:2px solid var(--ochre); outline-offset:5px; }
.room .q-btn { color:inherit!important; text-transform:none; font-weight:400; letter-spacing:0; }
.room .q-btn:before { box-shadow:none; }
.room .q-focus-helper { opacity:0!important; }
.room .q-btn:hover { color:var(--ochre)!important; }
.room .q-btn:disabled { opacity:.38!important; }
.room-head { width:100%; max-width:1320px; margin:auto; padding:30px 48px; display:flex; justify-content:space-between; align-items:center; }
.room-name { font-family:Fraunces,Georgia,serif; font-size:21px; }
.room-head .q-btn { font-size:12px; color:var(--muted)!important; }
.practice { max-width:760px; width:calc(100% - 48px); margin:14px auto 0; text-align:center; align-items:center; gap:0; }
.dots { display:flex; justify-content:center; flex-wrap:wrap; gap:0; min-height:36px; }
.dot.q-btn { min-width:25px!important; min-height:32px!important; padding:0!important; }
.dot .q-btn__content::after { content:''; width:5px; height:5px; background:var(--line); border-radius:50%; }
.dot.past .q-btn__content::after { background:var(--ochre); opacity:.5; }
.dot.current .q-btn__content::after { background:var(--ochre); outline:1px solid var(--ochre); outline-offset:5px; }
.position { font-size:10px; letter-spacing:.12em; color:var(--muted); margin-top:7px; }
.model-section { width:100%; margin-top:67px; }
.read-label { font-size:10px!important; letter-spacing:.16em!important; line-height:1.8; color:var(--muted)!important; }
.contour { width:90%; height:66px; margin:12px auto 4px; color:var(--ochre); }
.contour svg { width:100%; height:100%; }
.model-line { font-family:Fraunces,Georgia,serif; font-size:48px; font-weight:400; line-height:1.3; letter-spacing:-.025em; text-wrap:balance; min-height:125px; }
.spoken { color:var(--ochre); }
.breath-space { height:147px; display:flex; align-items:center; justify-content:center; position:relative; width:100%; }
.wave { width:300px; height:38px; color:var(--slate); opacity:.45; }
.wave svg { width:100%; height:100%; }
.wave.listening { opacity:.8; }
.wave.listening svg { transform-origin:center; animation:wave-breathe 1.8s ease-in-out infinite; }
@keyframes wave-breathe { 0%,100% { transform:scaleY(.55); } 50% { transform:scaleY(1); } }
.mic.q-btn { position:absolute; left:50%; transform:translateX(-50%); width:64px; height:64px; border-radius:50%; background:var(--button)!important; color:var(--slate)!important; box-shadow:0 6px 24px #30271912; }
.mic.listening { outline:1px solid var(--slate); outline-offset:6px; animation:breathe 2.5s ease-in-out infinite; }
@keyframes breathe { 50% { box-shadow:0 8px 32px #58757535; } }
.learner-section { width:100%; min-height:155px; }
.learner-line { font-family:Fraunces,Georgia,serif; color:var(--slate); font-size:30px; line-height:1.55; text-wrap:balance; margin:19px auto 0; max-width:640px; }
.learner-empty { font-size:23px; font-style:italic; opacity:.8; }
.learner-upcoming { color:var(--ink); }
.learner-spoken { color:var(--slate); }
.skipped { opacity:.32; }
.skipped:focus,.skipped:hover { opacity:1; }
.substituted { border-bottom:1px dotted currentColor; padding-bottom:2px; }
.room-status { min-height:36px; max-width:600px; font-size:12px; line-height:1.6; color:var(--muted); margin:9px auto; }
.actions { display:flex; gap:32px; justify-content:center; padding:20px 0 25px; width:100%; border-top:1px solid var(--line); }
.actions .q-btn { min-width:60px; font-size:14px; }
.room-note { font-size:10px; color:var(--muted); letter-spacing:.03em; padding-bottom:32px; }
.setup { max-width:760px; width:calc(100% - 48px); margin:0 auto 40px; border-top:1px solid var(--line); font-size:13px; }
.setup > .q-expansion-item__container > .q-expansion-item__content { padding:24px 0; }
.setup-body { width:100%; align-items:stretch; gap:0; }
.setup-section { width:100%; min-width:0; padding:0 0 28px; }
.setup-section + .setup-section { border-top:1px solid var(--line); padding-top:28px; }
.setup-title { font-family:Fraunces,Georgia,serif; font-size:25px; margin-bottom:8px; }
.setup-help { color:var(--muted); line-height:1.7; margin-bottom:20px; max-width:65ch; }
.setup .passage-editor { width:100%; }
.setup .passage-editor textarea { min-height:180px!important; line-height:1.8; font-size:15px; }
.setup-actions { width:100%; align-items:center; gap:12px; margin:12px 0; }
.room .setup-primary.q-btn { background:var(--slate)!important; color:var(--paper)!important; padding:8px 20px; }
.template-picker,.voice-controls { width:100%; flex-wrap:wrap; align-items:center; gap:12px; margin:20px 0; }
.setup .template-select { flex:1 1 260px; min-width:0; }
.setup .voice-controls .q-field { flex:1 1 220px; min-width:0; }
.setup-secondary { margin-top:12px; }
.setup-footnote { color:var(--muted); font-size:11px; line-height:1.8; }
.room-popup { background:var(--button)!important; color:var(--ink)!important; border:1px solid var(--line); }
.room-popup .q-item { color:var(--ink)!important; }
.room-popup .q-item--active { color:var(--ochre)!important; background:var(--paper); }
.room-popup .q-item:hover,.room-popup .q-item:focus { background:var(--paper); }
.shortcut-menu { min-width:260px; padding:20px; }
.shortcut-title { font-size:14px; margin-bottom:16px; }
.shortcut-row { align-items:center; justify-content:space-between; gap:20px; margin:12px 0; font-size:12px; }
.shortcut-key { border:1px solid var(--line); border-radius:4px; padding:3px 8px; }
.room-note .q-btn { color:var(--muted)!important; }
.setup .q-field,.setup .q-field__label,.setup .q-field__native,.setup .q-field__input { color:var(--ink)!important; }
.setup .q-field__control:before { border-color:var(--line)!important; }
.setup .q-field__append,.setup .q-field__prepend { color:var(--muted)!important; }
.setup .q-uploader { background:var(--button); color:var(--ink); box-shadow:none; max-width:100%; }
.setup .q-uploader__header { background:var(--slate); }
.setup .q-item { color:var(--muted); }
.setup .q-field { width:100%; }
.sample-note { color:var(--ochre); font-size:11px; margin:10px 0; }
@media(max-width:600px) {
 .room-head { padding:20px 20px; } .room-name { font-size:18px; }
 .practice { margin-top:8px; width:calc(100% - 40px); }
 .model-section { margin-top:38px; } .model-line { font-size:35px; min-height:135px; }
 .read-label { font-size:8px!important; letter-spacing:.12em!important; }
 .contour { height:50px; } .breath-space { height:125px; }
 .learner-line { font-size:25px; } .learner-section { min-height:155px; }
 .actions { gap:22px; } .room-note { font-size:9px; }
}
@media(min-width:601px) and (max-height:800px) {
 .room-head { padding-top:20px; padding-bottom:15px; }
 .practice { margin-top:0; } .model-section { margin-top:24px; }
 .breath-space { height:112px; } .learner-section { min-height:119px; }
 .room-status { margin:0 auto; min-height:29px; }
 .actions { padding:10px 0; } .room-note { padding-bottom:18px; }
}
@media(prefers-reduced-motion:reduce) {
 body, .room *, .room-popup, .room-popup * { animation:none!important; transition:none!important; }
}
"""
WAVE = ('<svg viewBox="0 0 300 38" aria-hidden="true"><path d="M0 19 H30 '
        'Q34 19 37 16 Q40 12 43 19 Q46 28 49 19 Q52 6 55 19 Q58 34 61 19 '
        'Q64 2 67 19 Q70 35 73 19 Q76 7 79 19 Q82 28 85 19 Q88 14 91 19 H209 '
        'Q212 14 215 19 Q218 28 221 19 Q224 7 227 19 Q230 34 233 19 '
        'Q236 2 239 19 Q242 32 245 19 Q248 9 251 19 Q254 26 257 19 '
        'Q260 15 263 19 H300" fill="none" stroke="currentColor" stroke-width="1"/></svg>')
SAMPLE_CONTOUR = ('<svg viewBox="0 0 640 64" aria-label="Illustrative pitch contour" role="img">'
                  '<path d="M10 43 C65 43 65 16 125 22 S195 52 250 28 S320 10 375 26 '
                  'S440 12 483 27 S560 55 630 44" fill="none" stroke="currentColor" '
                  'stroke-width="9" opacity=".18" stroke-linecap="round"/>'
                  '<path d="M10 43 C65 43 65 16 125 22 S195 52 250 28 S320 10 375 26 '
                  'S440 12 483 27 S560 55 630 44" fill="none" stroke="currentColor" '
                  'stroke-width="2" stroke-linecap="round"/></svg>')
