import dashscope
from dashscope.audio.asr import (Recognition, RecognitionCallback)

from keys import dashscope_keys

dashscope.api_key = dashscope_keys


class ASR:
    def __init__(self, callback: RecognitionCallback, disfluency_removal=True) -> None:
        self.callback = callback
        self.recognizer = Recognition(model='paraformer-realtime-v1', format='pcm', sample_rate=16000,
                                      callback=callback,
                                      disfluency_removal_enabled=disfluency_removal)

    def start(self):
        self.callback.done = False

        self.recognizer.start()

        while not self.callback.done:
            data = self.callback.stream.read(3200, exception_on_overflow=False)
            self.recognizer.send_audio_frame(data)

        self.recognizer.stop()

        return self.callback.return_sentence
