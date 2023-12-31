import time

import dashscope
from dashscope.audio.asr import (Recognition, RecognitionCallback)

from keys import dashscope_keys

dashscope.api_key = dashscope_keys


class ASR(Recognition):
    def __init__(self, callback: RecognitionCallback, audio_input, model: str = 'paraformer-realtime-v1',
                 format: str = 'pcm', sample_rate: int = 16000, disfluency_removal=True, **kwargs) -> None:
        super().__init__(model, callback, format, sample_rate, disfluency_removal=disfluency_removal, **kwargs)
        self.callback = callback
        self.audio_input = audio_input

    def get_sentence(self):
        if self.callback.done:
            return self.callback.sentence
        else:
            return None

    def is_done(self):
        return self.callback.done

    def start(self, phrase_id: str = None, idle_timeout: int = 30, **kwargs):
        super().start(phrase_id, **kwargs)
        start_time = time.time()
        while not self.is_done() and time.time() - start_time < idle_timeout:
            audio = self.audio_input.read(1600, exception_on_overflow=False)
            self.send_audio_frame(audio)
        self.stop()
        return self.get_sentence()
