import base64
import re

import dashscope
import sys

import numpy as np
import pyaudio
import requests
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts import ResultCallback, SpeechSynthesizer, SpeechSynthesisResult
from keys import dashscope_keys

dashscope.api_key = dashscope_keys


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


class TTS:
    def __init__(self, callback: ResultCallback = None, model={'zh': 'sambert-zhiyuan-v1', 'en': 'sambert-cindy-v1'},
                 sample_rate=24000):
        self.callback = callback if callback is not None else ResultCallback()
        self.model = model
        self.sample_rate = sample_rate

    def say(self, text: str):
        # 如果中文字符所占比例大于0.2
        if len(''.join(re.findall(r'[\u4e00-\u9fff]+', text))) / len(text) > 0.2:
            model = self.model['zh']
        else:
            model = self.model['en']

        return SpeechSynthesizer.call(model=model, text=text, sample_rate=self.sample_rate, format='wav',
                                      callback=self.callback)


class TTSLocal:
    def __init__(self, callback: ResultCallback = None, server='127.0.0.1', port=8888, sample_rate=None):
        self.callback = callback if callback is not None else ResultCallback()
        self.url = f'http://{server}:{port}/paddlespeech/tts/streaming'
        self.sr_url = f'http://{server}:{port}/paddlespeech/tts/streaming/samplerate'

        if sample_rate is None:
            self.sample_rate = requests.get(self.sr_url).json()["sample_rate"]
        else:
            self.sample_rate = sample_rate
        self.callback.sample_rate = self.sample_rate

    def say(self, text: str):
        all_bytes = b''
        html = requests.post(url=self.url, json={'text': text, 'spk_id': 0}, stream=True)
        self.callback.on_open()

        for chunk in html.iter_content(chunk_size=None):
            audio = base64.b64decode(chunk)
            all_bytes += audio
            self.callback.on_event(SpeechSynthesisResult(audio, None, None, None, None))

        html.close()

        self.callback.on_close()

        return SpeechSynthesisResult(all_bytes, None, None, None, None)


if __name__ == '__main__':
    text = """我在获取今天的科技新闻，稍等一下。"""

    callback = TerminalTTSCallback()
    tts = TTS(callback=callback, sample_rate=24000)
    result = tts.say(text)

    if result.get_audio_data() is not None:
        with open('output_48k.wav', 'wb') as f:
            f.write(result.get_audio_data())
