import io
import os

import numpy as np
import pyaudio
import scipy as sc
import torch
import torchaudio
from dtaidistance.dtw_ndim import distance_fast

from kws_model import Wav2Vec2Finetuner


class KeywordSpotter:
    def __init__(self, templates_folder='./Records/positive/'):
        # bundle = torchaudio.pipelines.WAV2VEC2_BASE
        # self.acoustic_model = bundle.get_model()
        # self.sr = bundle.sample_rate
        self.acoustic_model = Wav2Vec2Finetuner(114514, 219)
        self.acoustic_model.load_state_dict(
            torch.load('./Checkpoints/Finetune_test/epoch=5-step=60000.pt'))
        self.sr = 16000
        self.buffer = []
        self.window_size = 20
        self.overlap_size = 10
        self.overlap_count = 0
        self.threshold = 200

        self.templates_folder = templates_folder
        # templates_paths = ['template1.wav', 'template2.wav', 'template3.wav']
        templates_paths = []
        for f in os.listdir(templates_folder):
            if f.startswith('template') and f.endswith('wav'):
                templates_paths.append(f)
        self.templates = []
        for t_path in templates_paths:
            path = os.path.join(templates_folder, t_path)
            t_input, _ = torchaudio.load(path)
            with torch.inference_mode():
                t_output, _ = self.acoustic_model(t_input)
            self.templates.append(t_output.squeeze(0).numpy().astype(np.float64))

    def check(self, chunk):
        '''Collect chunks and CHECK if the keyword appears'''
        ndata = np.frombuffer(chunk, dtype=np.int16)
        if len(self.buffer) == self.window_size:
            self.buffer.pop(0)
            self.overlap_count = (self.overlap_count + 1) % self.overlap_size
        self.buffer.append(ndata)
        # print(time.time())
        if len(self.buffer) == self.window_size and self.overlap_count == 0:
            window = self.chunks2tensor()
            # time1 = time.time()
            with torch.inference_mode():
                emission, _ = self.acoustic_model(window)
            # time2 = time.time()
            score = self.match(emission.squeeze(0))
            if score < self.threshold:
                # print(True)
                self.buffer = []
                self.overlap_count = 0
                return True
            # time3 = time.time()
            # print(f'InferTime: {time2-time1}')
            # print(f'MatchTime: {time3-time2}')
        return False

    def test(self):
        examples = ['positive', 'negative', 'noise']
        for eg in examples:
            wave, _ = torchaudio.load(os.path.join(self.templates_folder, f'{eg}.wav'))
            with torch.inference_mode():
                emission, _ = self.acoustic_model(wave)
            # time2 = time.time()
            score = self.match(emission.squeeze(0))
            print(f"{eg} test: {score < self.threshold}")

    def single_test(self, path):
        wave, _ = torchaudio.load(path)
        with torch.inference_mode():
            emission, _ = self.acoustic_model(wave)
        # time2 = time.time()
        score = self.match(emission.squeeze(0))
        print(f"single test: {score < self.threshold}")

    def match(self, search):
        losses = []
        for template in self.templates:
            # losses.append(dtw.accelerated_dtw(template, search, self.dist)[0])
            losses.append(distance_fast(template, search.numpy().astype(np.float64)))
        loss = np.array(losses).mean()
        print(loss, losses)
        return loss

    def chunks2tensor(self):
        nchunk = np.concatenate(self.buffer)
        bytes_io = io.BytesIO()
        sc.io.wavfile.write(bytes_io, 16000, nchunk)
        tchunk, _ = torchaudio.load(bytes_io)
        return tchunk

    def detect(self, stream):
        while True:
            pcm = stream.read(1600)
            if self.check(pcm):
                break


if __name__ == '__main__':
    KWSer = KeywordSpotter('./Records/dingzhen/')

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        frames_per_buffer=1600,
        input=True,
        # stream_callback=callback
    )

    while stream:
        data = stream.read(1600)
        res = KWSer.check(data)
        print(f"Check Result: {res}")

    # data = stream.read(3200)
    # for _ in range(1000):
    #     res = KWSer.check(data)
