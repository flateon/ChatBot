import pyaudio
import dashscope
from dashscope.audio.asr import (Recognition, RecognitionCallback,
                                 RecognitionResult)
from keys import dashscope_keys

dashscope.api_key = dashscope_keys


class TerminalASRCallback(RecognitionCallback):
    def on_open(self) -> None:
        global mic
        global stream
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=16000,
                          input=True)

    def on_close(self) -> None:
        global mic
        global stream
        stream.stop_stream()
        stream.close()
        mic.terminate()

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        text = sentence['text']
        print(text)
        if result.is_sentence_end(sentence):
            global done
            global return_sentence
            done = True
            return_sentence = sentence['text']


class ASR:
    def __init__(self, callback: RecognitionCallback, disfluency_removal=True) -> None:
        self.callback = callback
        self.recognizer = Recognition(model='paraformer-realtime-v1', format='pcm', sample_rate=16000,
                                      callback=callback,
                                      disfluency_removal_enabled=disfluency_removal)

    def start(self):
        global done
        global return_sentence
        done = False

        self.recognizer.start()

        while not done:
            data = stream.read(3200, exception_on_overflow=False)
            self.recognizer.send_audio_frame(data)

        self.recognizer.stop()

        return return_sentence


if __name__ == '__main__':
    asr = ASR(TerminalASRCallback())
    print(f'#### {asr.start()}')
