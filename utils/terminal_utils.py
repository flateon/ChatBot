import random
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
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


class StreamWithVolumeDisplay:
    def __init__(self, audio_stream, volume_displayer):
        self.audio_stream = audio_stream
        self.volume_display = volume_displayer

    def read(self, num_frames, exception_on_overflow=False):
        audio = self.audio_stream.read(num_frames, exception_on_overflow)

        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768
        volume = np.log10(np.mean(audio_np ** 2) + 1e-10) * 20 + 100
        self.volume_display(volume)

        return audio

    def write(self, audio):
        self.audio_stream.write(audio)

    def close(self):
        self.audio_stream.close()


class HelloByeHandler:
    def __init__(self, tts, printer, audio_output, audio_path):
        self.tts = tts
        self.printer = printer
        self.audio_output = audio_output
        self.audio_path = Path(audio_path)
        self.hello_text = ['我在，有什么我可以帮助你的吗？', '你好，我是小圆，有什么可以帮到你吗？']
        self.bye_text = [
            '很高兴能与您聊天，如果您有任何其他问题，随时欢迎回来。再见了！',
            '感谢您的使用，希望我们的对话对您有所帮助。祝您有美好的一天，再见！',
            '祝您一切顺利，期待下次与您再次交谈。再见！'
        ]
        self.hello_audio = []
        self.bye_audio = []
        self.load_audio()

    def load_audio(self):
        if len(list(self.audio_path.glob('hello_*.pcm'))) < len(self.hello_text):
            for i, start in enumerate(self.hello_text):
                result = self.tts.say(start)
                with open(self.audio_path / f'hello_{i}.pcm', 'wb') as f:
                    f.write(result.get_audio_data())
                self.hello_audio.append(result.get_audio_data())
        else:
            for i in range(len(self.hello_text)):
                with open(self.audio_path / f'hello_{i}.pcm', 'rb') as f:
                    self.hello_audio.append(f.read())

        if len(list(self.audio_path.glob('bye_*.pcm'))) < len(self.bye_text):
            for i, end in enumerate(self.bye_text):
                result = self.tts.say(end)
                with open(self.audio_path / f'bye_{i}.pcm', 'wb') as f:
                    f.write(result.get_audio_data())
                self.bye_audio.append(result.get_audio_data())
        else:
            for i in range(len(self.bye_text)):
                with open(self.audio_path / f'bye_{i}.pcm', 'rb') as f:
                    self.bye_audio.append(f.read())

    def hello(self):
        idx = random.choice(range(len(self.hello_text)))
        self.printer.get_live_print('Bot')(self.hello_text[idx], add_history=True)

        data = self.hello_audio[idx]
        self.audio_output.write(data)

    def bye(self):
        idx = random.choice(range(len(self.bye_text)))
        self.printer.get_live_print('Bot')(self.bye_text[idx], add_history=True)
        self.audio_output.write(self.bye_audio[idx])
