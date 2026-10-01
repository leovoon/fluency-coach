from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from coach import config, shadowing
from coach.asr_worker import _transcribe_phonon, _with_pitch


class PresentationTests(unittest.TestCase):
    def test_sample_has_two_skips_and_one_substitution(self):
        html = shadowing.learner_html(shadowing.SAMPLE, shadowing.SAMPLE_READ)
        self.assertEqual(html.count('class="skipped"'), 2)
        self.assertEqual(html.count('class="substituted"'), 1)
        self.assertIn('>can</span>', html)
        self.assertIn('Not heard: quiet', html)
        self.assertIn('Not heard: tonight', html)

    def test_learner_playback_highlights_only_heard_words(self):
        start = shadowing.learner_html(shadowing.SAMPLE, shadowing.SAMPLE_READ, 0)
        middle = shadowing.learner_html(shadowing.SAMPLE, shadowing.SAMPLE_READ, .5)
        end = shadowing.learner_html(shadowing.SAMPLE, shadowing.SAMPLE_READ, 1)
        self.assertEqual(start.count('class="learner-spoken"'), 1)
        self.assertEqual(start.count('class="learner-upcoming"'), 5)
        self.assertEqual(middle.count('class="learner-spoken"'), 4)
        self.assertEqual(end.count('class="learner-spoken"'), 6)
        self.assertEqual(end.count('class="skipped"'), 2)
        self.assertIn('class="learner-spoken">can</span>', middle)
        self.assertNotIn('class="learner-spoken">quiet', end)
        self.assertNotIn('class="learner-spoken">.</span>', end)

    def test_playback_insertions_empty_and_clamping(self):
        html = shadowing.learner_html('I like tea.', 'I I really like tea.', 2)
        self.assertEqual(html.count('class="learner-spoken"'), 5)
        self.assertIn('Additional words', html)
        self.assertNotIn('learner-spoken', shadowing.learner_html('hello', '', 1))
        self.assertNotIn('learner-spoken', shadowing.learner_html('hello', 'hello', -1))

    def test_repeated_insertions_survive(self):
        html = shadowing.learner_html('I like tea.', 'I I like tea.')
        self.assertIn('Additional words', html)
        self.assertEqual(html.count('I'), 3)  # visible twice plus the accessible insertion label

    def test_html_is_escaped(self):
        self.assertNotIn('<script>', shadowing.model_html('<script>alert(1)</script>'))
        self.assertNotIn('<img', shadowing.learner_html('hello', '<img src=x onerror=alert(1)>'))

    def test_empty_transcript_is_all_ghosts(self):
        html = shadowing.learner_html('Read slowly', '')
        self.assertEqual(html.count('class="skipped"'), 2)

    def test_model_progress_is_clamped(self):
        self.assertEqual(shadowing.model_html('one two', -1).count('class="spoken"'), 0)
        self.assertEqual(shadowing.model_html('one two', 2).count('class="spoken"'), 2)

    def test_source_labels_are_truthful(self):
        self.assertIn('HUMAN', shadowing.source_label(True, 'pocket', True))
        self.assertIn('CLONED', shadowing.source_label(False, 'pocket', True))
        self.assertNotIn('CLONED', shadowing.source_label(False, 'pocket', False))
        self.assertNotIn('CLONED', shadowing.source_label(False, 'kokoro', True))

    def test_contour_silence_and_real_tone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            sf.write(path, np.zeros(16000), 16000)
            self.assertEqual(shadowing.contour_svg(path), '')
            wave = .2 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)
            sf.write(path, wave, 16000)
            self.assertIn('<svg', shadowing.contour_svg(path))


class SpeechTimingTests(unittest.TestCase):
    def test_silence_padding_is_excluded_without_changing_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'padded.wav'
            tone = .1 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)
            audio = np.concatenate([np.zeros(8000), tone, np.zeros(19200)])
            sf.write(path, audio, 16000)
            original = path.read_bytes()
            start, end = shadowing.speech_span(path)
            self.assertAlmostEqual(start, .5, delta=.04)
            self.assertAlmostEqual(end, 1.5, delta=.04)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(shadowing.speech_progress(.2, (start, end)), -1)
            self.assertEqual(shadowing.speech_progress(1.6, (start, end)), 1)
            self.assertAlmostEqual(shadowing.speech_progress(1, (start, end)), .5, delta=.04)

    def test_silent_empty_and_short_stereo_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            for audio in (np.zeros(0), np.zeros(1600), np.full(1600, .0001)):
                sf.write(path, audio, 16000)
                self.assertIsNone(shadowing.speech_span(path))
            tone = .01 * np.sin(2 * np.pi * 180 * np.arange(160) / 16000)
            sf.write(path, np.column_stack([tone, -tone]), 16000)
            self.assertEqual(shadowing.speech_span(path), (0, .01))

    def test_words_start_at_estimated_onsets(self):
        for elapsed, count in ((.2, 0), (.5, 1), (.76, 2), (1.49, 4), (2.6, 4)):
            fraction = shadowing.speech_progress(elapsed, (.5, 1.5))
            html = shadowing.learner_html('one two three four', 'one two three four', fraction)
            self.assertEqual(html.count('class="learner-spoken"'), count)


class ConfigTests(unittest.TestCase):
    @patch('coach.config._read_config', return_value={})
    def test_default_phonon_and_no_arrow_flow(self, _read):
        settings = config.load([])
        self.assertEqual(settings.asr_engine, 'phonon')
        self.assertEqual(settings.asr_id, 'FermionResearch/Phonon-2')
        self.assertFalse(settings.melody)
        self.assertFalse(settings.flow)

    @patch('coach.config._read_config', return_value={'asr': {'engine': 'photon'}})
    def test_alternative_engine_preserved(self, _read):
        self.assertEqual(config.load([]).asr_id, 'moondream/parakeet-redux')

    @patch('coach.config._read_config', return_value={'asr': {'engine': 'photon', 'id': 'custom'}})
    def test_explicit_model_preserved(self, _read):
        self.assertEqual(config.load([]).asr_id, 'custom')


class PhononTests(unittest.TestCase):
    def test_text_only_does_not_request_pitch(self):
        class Result:
            truncated = False
            def triple(self):
                return 'Hello there.', .1, 1
        class Speech:
            def transcribe_detailed(self, path):
                self.path = path
                return Result()
        speech = Speech()
        result = _transcribe_phonon(speech, 'read.wav')
        self.assertEqual(result, {'text': 'Hello there.', 'words': None})
        self.assertEqual(speech.path, 'read.wav')
        with patch('coach.asr_worker._pitch_on', True):
            self.assertIs(_with_pitch(result, 'missing.wav'), result)


class PlaybackTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_then_take(self):
        played = []
        async def play(path):
            played.append(path)
        await shadowing.play_sequence(['model.wav', 'take.wav'], play)
        self.assertEqual(played, ['model.wav', 'take.wav'])

    async def test_cancel_after_model(self):
        played = []
        async def play(path):
            played.append(path)
        await shadowing.play_sequence(['model', 'take'], play, alive=lambda: not played)
        self.assertEqual(played, ['model'])

    async def test_failure_does_not_play_second(self):
        played = []
        async def play(path):
            played.append(path)
            raise RuntimeError('speaker unavailable')
        with self.assertRaises(RuntimeError):
            await shadowing.play_sequence(['model', 'take'], play)
        self.assertEqual(played, ['model'])

    async def test_empty_sequence(self):
        async def play(_):
            self.fail('Nothing should play')
        await shadowing.play_sequence([], play)


if __name__ == '__main__':
    unittest.main()
