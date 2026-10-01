import asyncio
import threading
import time
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from nicegui import ui

from coach import config, room, shadowing


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / 'take.wav'
    sf.write(path, np.zeros(1600), 16000)
    return path


def mount(templates=None):
    with patch('coach.config._read_config', return_value={}):
        settings = replace(config.load([]), tts_engine='pocket', tts=True)

    @ui.page('/')
    def page():
        room.build(settings, shadowing.SAMPLE, templates or {}, theme='candlelit')


@pytest.mark.asyncio
async def test_record_compare_and_reset(user, audio):
    played = []
    with (patch('coach.room.models.model_voice_path', return_value=None),
          patch('coach.room.models.list_voices', return_value=[]),
          patch('coach.room.reference.has', return_value=True),
          patch('coach.room.reference.path_for', return_value='reference.wav'),
          patch('coach.room.record.record_utterance', return_value=audio),
          patch('coach.room.models.transcribe_detailed',
                return_value={'text': shadowing.SAMPLE_READ, 'words': None}),
          patch('coach.room.view.contour_svg', return_value=''),
          patch('soundfile.info') as info,
          patch('coach.room.record.play_wav', side_effect=lambda p: played.append(str(p)))):
        info.return_value.duration = .1
        mount()
        await user.open('/')
        user.find(kind=ui.button, content='record').click()
        await user.should_see('Your read is ready.')
        await user.should_see('Not heard: quiet')
        user.find(kind=ui.button, content='play').click()
        await asyncio.sleep(.8)
        assert played == ['reference.wav', str(audio)]
        user.find(kind=ui.button, content='again').click()
        await user.should_see('Your voice will find its place here.')
        assert not audio.exists()


@pytest.mark.asyncio
async def test_record_error_recovers(user):
    with (patch('coach.room.models.model_voice_path', return_value=None),
          patch('coach.room.models.list_voices', return_value=[]),
          patch('coach.room.reference.has', return_value=False),
          patch('coach.room.record.record_utterance', side_effect=RuntimeError('No speech detected'))):
        mount()
        await user.open('/')
        user.find(kind=ui.button, content='record').click()
        await user.should_see('No speech detected')
        assert user.find(kind=ui.button, content='record').elements.pop().enabled
        wave = next(e for e in user.client.elements.values() if 'wave' in e.classes)
        assert 'listening' not in wave.classes


@pytest.mark.asyncio
async def test_empty_passage_does_not_replace_current_sentence(user):
    with (patch('coach.room.models.model_voice_path', return_value=None),
          patch('coach.room.models.list_voices', return_value=[]),
          patch('coach.room.reference.has', return_value=False)):
        mount()
        await user.open('/')
        user.find('practice setup').click()
        user.find(ui.textarea).clear()
        user.find(kind=ui.button, content='load passage').click()
        await user.should_see('Add a sentence before loading a passage.')
        await user.should_see('We')


def press_play(user, **overrides):
    args = dict(action='keydown', key='p', code='KeyP', location=0,
                repeat=False, altKey=False, ctrlKey=False, metaKey=False, shiftKey=False)
    args.update(overrides)
    user.find(ui.keyboard).trigger('key', args)


@pytest.mark.asyncio
async def test_play_shortcut_and_modifier_guard(user):
    with (patch('coach.room.models.model_voice_path', return_value=None),
          patch('coach.room.models.list_voices', return_value=[]),
          patch('coach.room.reference.has', return_value=True),
          patch('coach.room.reference.path_for', return_value='reference.wav'),
          patch('coach.room.view.contour_svg', return_value=''),
          patch('soundfile.info') as info,
          patch('coach.room.record.play_wav') as play):
        info.return_value.duration = .1
        mount()
        await user.open('/')
        press_play(user, ctrlKey=True)
        press_play(user, metaKey=True)
        press_play(user, altKey=True)
        press_play(user, action='keyup')
        await asyncio.sleep(.1)
        play.assert_not_called()
        press_play(user, key='P', shiftKey=True)
        await user.should_see('Leave a little space')
        play.assert_called_once_with('reference.wav')
        keyboard = next(iter(user.find(ui.keyboard).elements))
        assert {'input', 'textarea', 'select'} <= set(keyboard._props['ignore'])


@pytest.mark.asyncio
async def test_wave_stops_before_transcription_and_blocks_play(user, audio):
    captured, transcribed = threading.Event(), threading.Event()

    def capture():
        assert captured.wait(3)
        return audio

    def transcribe(_):
        assert transcribed.wait(3)
        return {'text': shadowing.SAMPLE_READ, 'words': None}

    with (patch('coach.room.models.model_voice_path', return_value=None),
          patch('coach.room.models.list_voices', return_value=[]),
          patch('coach.room.reference.has', return_value=False),
          patch('coach.room.record.record_utterance', side_effect=capture),
          patch('coach.room.models.transcribe_detailed', side_effect=transcribe),
          patch('coach.room.models.synthesize') as synth):
        mount()
        await user.open('/')
        wave = next(e for e in user.client.elements.values() if 'wave' in e.classes)
        try:
            user.find(kind=ui.button, content='record').click()
            await user.should_see('Listening on this Mac')
            assert 'listening' in wave.classes
            press_play(user)
            await asyncio.sleep(.05)
            synth.assert_not_called()
            captured.set()
            await user.should_see('Listening back to your words')
            assert 'listening' not in wave.classes
            transcribed.set()
            await user.should_see('Your read is ready')
        finally:
            captured.set()
            transcribed.set()


@pytest.mark.asyncio
async def test_learner_playback_progress_and_replay(user, audio):
    with (patch('coach.room.models.model_voice_path', return_value=None),
          patch('coach.room.models.list_voices', return_value=[]),
          patch('coach.room.reference.has', return_value=False),
          patch('coach.room.record.record_utterance', return_value=audio),
          patch('coach.room.models.transcribe_detailed',
                return_value={'text': shadowing.SAMPLE_READ, 'words': None}),
          patch('soundfile.info') as info,
          patch('coach.room.record.play_wav', side_effect=lambda _: time.sleep(.6))):
        info.return_value.duration = .6
        mount()
        await user.open('/')
        user.find(kind=ui.button, content='record').click()
        await user.should_see('Your read is ready')
        learner = next(e for e in user.client.elements.values() if 'learner-line' in e.classes)
        for _ in range(2):
            user.find(kind=ui.button, content='YOUR READ').click()
            await asyncio.sleep(.28)
            assert 0 < learner.content.count('class="learner-spoken"') < 6
            assert 'learner-upcoming' in learner.content
            assert learner.content.count('class="skipped"') == 2
            await asyncio.sleep(.45)
            assert learner.content.count('class="learner-spoken"') == 6
            assert 'learner-upcoming' not in learner.content
        with patch('coach.room.record.play_wav', side_effect=RuntimeError('speaker unavailable')):
            user.find(kind=ui.button, content='YOUR READ').click()
            await user.should_see('speaker unavailable')
            assert 'learner-upcoming' not in learner.content
            assert 'learner-spoken' not in learner.content


@pytest.mark.asyncio
async def test_setup_controls_and_icon_theme(user):
    with (patch('coach.room.models.model_voice_path', return_value=None),
          patch('coach.room.models.list_voices', return_value=[]),
          patch('coach.room.reference.has', return_value=False)):
        mount({'Quiet afternoon': 'A quiet afternoon.'})
        await user.open('/')
        buttons = [e for e in user.client.elements.values() if isinstance(e, ui.button)]
        theme = next(e for e in buttons if e.props.get('aria-label') == 'Toggle candlelit theme')
        assert theme.props['aria-pressed'] == 'true'
        assert not theme.text
        assert 'theme-switch' in theme.classes
        user.find(marker='theme-switch').click()
        assert theme.props['aria-pressed'] == 'false'
        root = next(e for e in user.client.elements.values() if 'room' in e.classes)
        assert 'candlelit' not in root.classes
        user.find(marker='theme-switch').click()
        assert theme.props['aria-pressed'] == 'true'
        assert 'candlelit' in root.classes
        assert not any(isinstance(e, ui.link) and e.text == 'See the sample design'
                       for e in user.client.elements.values())
        user.find('practice setup').click()
        await user.should_see('Sentence reference')
        select = next(iter(user.find(ui.select).elements))
        assert select.props['popup-content-class'] == 'room-popup'
        assert 'passage-editor' in next(iter(user.find(ui.textarea).elements)).classes
        shortcut = next(e for e in buttons if e.props.get('aria-label') == 'Keyboard shortcuts')
        assert shortcut.props['icon'] == 'keyboard'

