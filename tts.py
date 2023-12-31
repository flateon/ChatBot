import base64
import re

import dashscope
import requests
from dashscope.audio.tts import ResultCallback, SpeechSynthesizer, SpeechSynthesisResult

from keys import dashscope_keys

dashscope.api_key = dashscope_keys


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

        return SpeechSynthesizer.call(model=model, text=text, sample_rate=self.sample_rate, format='pcm',
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
