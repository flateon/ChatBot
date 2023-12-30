import re

import pyaudio
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.asr import RecognitionResult, RecognitionCallback
from dashscope.audio.tts import ResultCallback, SpeechSynthesisResult

from asr import ASR
from llm import LLMPlugin, LLMCallback
from tts import TTS, TTSLocal


class TerminalASRCallback(RecognitionCallback):
    def __init__(self):
        self.return_sentence = None
        self.done = None
        self.mic = None
        self.stream = None

    def on_open(self) -> None:
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=16000,
                                    input=True)

    def on_close(self) -> None:
        self.stream.stop_stream()
        self.stream.close()
        self.mic.terminate()

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        text = sentence['text']
        print(text)
        if result.is_sentence_end(sentence):
            self.done = True
            self.return_sentence = sentence['text']


class TerminalCallback(LLMCallback):
    def __init__(self, tts: TTS | TTSLocal = None):
        super().__init__()
        self.head_idx = 0
        self.read_idx = 0
        self.text = ''
        self.tts = tts if tts is not None else None

    def on_open(self):
        print('\nassistant: ', end='')

    def on_complete(self):
        print('')
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
        self.text = text
        if '`' in text or 'get' in text:
            if 'get_' in text:
                text = re.sub(r'`?get_(news|weather)?\(?.*?(\)|$)`?', '', text).strip()
                self.text = text
            else:
                # wait func call finish
                return
        print(text[self.head_idx:len(text)], end='', flush=True)

        if self.tts is not None:
            read_text = text[self.read_idx:]
            last_punctuation_idx = max(map(read_text.rfind, ('. ', '。', '!', '！', '?', '？', ';', '；', ':', '：')))
            if last_punctuation_idx != -1:
                read_text = read_text[:last_punctuation_idx + 1].strip()
                if len(read_text) <= 2:
                    pass
                else:
                    self.tts.say(read_text)
                    self.read_idx += last_punctuation_idx + 1

        self.head_idx = len(text)


class TerminalTTSCallback(ResultCallback):
    def __init__(self, sample_rate=24000):
        self._player = None
        self._stream = None
        self.sample_rate = sample_rate

    def on_open(self):
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True)

    def on_error(self, response: SpeechSynthesisResponse):
        print('Speech synthesizer failed, response is %s' % (str(response)))

    def on_close(self):
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, result: SpeechSynthesisResult):
        if result.get_audio_frame() is not None:
            self._stream.write(result.get_audio_frame())

        if result.get_timestamp() is not None:
            print('timestamp result:', str(result.get_timestamp()))


def test():
    test_case = ['今天南京的天气怎么样', '今天北京的天气怎么样', '昨天的天气怎么样', '今天有什么新闻',
                 '今天有什么AI新闻']
    callback = TerminalCallback()
    llm = LLMPlugin(callback)
    for t in test_case:
        print(f'\nuser: {t}')
        reply = llm.generate(t)


if __name__ == '__main__':
    asr = ASR(TerminalASRCallback())
    tts = TTS(TerminalTTSCallback(sample_rate=48000), sample_rate=48000)
    llm = LLMPlugin(TerminalCallback(tts))

    while True:
        user_msg = asr.start()
        # user_msg = input('user: ')
        mgs = llm.generate(user_msg).content
