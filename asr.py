import dashscope
from dashscope.audio.asr import (Recognition, RecognitionCallback)

from keys import dashscope_keys

dashscope.api_key = dashscope_keys


class ASR(Recognition):
    def __init__(self, callback: RecognitionCallback, model: str = 'paraformer-realtime-v1', format: str = 'pcm',
                 sample_rate: int = 16000, disfluency_removal=True, **kwargs) -> None:
        super().__init__(model, callback, format, sample_rate, disfluency_removal=disfluency_removal, **kwargs)
        self.callback = callback

    def get_sentence(self):
        if self.callback.done:
            return self.callback.sentence

    def is_done(self):
        return self.callback.done
