import re
import sys
from functools import partial

import pyaudio
import rich
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.asr import RecognitionCallback, RecognitionResult
from dashscope.audio.tts import ResultCallback, SpeechSynthesisResult
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text

from llm import LLMCallback
from tts import TTS, TTSLocal


class TerminalASRCallback(RecognitionCallback):
    def __init__(self, logger=None):
        self.sentence = None
        self.done = False
        self.logger = logger if logger is not None else Printer(rich.Console())

    def on_open(self) -> None:
        self.done = False
        self.sentence = None

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        text = sentence['text']
        self.logger(text)
        if result.is_sentence_end(sentence):
            self.done = True
            self.sentence = text
            self.logger(text, add_history=True)


class TerminalCallback(LLMCallback):
    def __init__(self, tts: TTS | TTSLocal = None, logger=None):
        super().__init__()
        self.head_idx = 0
        self.read_idx = 0
        self.text = ''
        self.logger = logger if logger is not None else Printer(rich.Console())
        self.tts = tts if tts is not None else None

    def on_complete(self):
        self.logger(self.text, add_history=True)
        if self.tts is not None:
            read_text = self.text[self.read_idx:].strip()
            if read_text == '':
                pass
            else:
                self.tts.say(read_text)
        self.head_idx = 0
        self.read_idx = 0

    def on_event(self, msg):
        text = msg.content
        if '`' in text or 'get' in text:
            if 'get_' in text:
                text = re.sub(r'`?get_(news|weather)?\(?.*?(\)|$)`?', '', text).strip()
                self.text = text
            else:
                return
        else:
            self.text = text
        self.logger(text)
        read_text = text[self.read_idx:]
        last_punctuation_idx = max(map(read_text.rfind, ('. ', '。', '!', '！', '?', '？', ';', '；', ':', '：')))
        if last_punctuation_idx != -1:
            read_text = read_text[:last_punctuation_idx + 1].strip()

            if len(read_text) <= 2:
                pass
            else:
                if self.tts is not None:
                    self.tts.say(read_text)
                self.read_idx += last_punctuation_idx + 1

        self.head_idx = len(text)


class TerminalTTSCallback(ResultCallback):
    def __init__(self, audio_stream):
        self.audio_stream = audio_stream

    def on_error(self, response: SpeechSynthesisResponse):
        print('Speech synthesizer failed, response is %s' % (str(response)))

    def on_event(self, result: SpeechSynthesisResult):
        if result.get_audio_frame() is not None:
            self.audio_stream.write(result.get_audio_frame())


class Printer:
    def __init__(self, console):
        self.console = console
        self.history = []

    def _print(self, role, text):
        text = Text.from_markup(f'[yellow]{text}', overflow='fold')
        panel = Panel(text, title=f'[b]{role}[/b]', title_align='right' if role == 'User' else 'left')
        columns = Columns((panel,))
        self.console.print(columns, justify='right' if role == 'User' else 'left', width=80)

    def print_all(self):
        self.console.clear()
        for role, text in self.history:
            self._print(role, text)

    def live_print(self, text, role, add_history=False):
        self.print_all()
        self._print(role, text)
        if add_history:
            self.history.append((role, text))

    def get_live_print(self, role):
        return partial(self.live_print, role=role)
