import io
import os

import numpy as np
import pyaudio
import scipy as sc
import torch
import torchaudio
from dtaidistance.dtw_ndim import distance_fast


class KeywordSpotter():
    def __init__(self):
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.acoustic_model = bundle.get_model()
        self.sr = bundle.sample_rate
        self.buffer = []
        self.window_size = 10
        self.overlap_size = 10
        self.overlap_count = 0
        self.threshold = 800

        def manhattan_distance(x, y):
            return np.linalg.norm(x - y, ord=1)

        def euclidean_distance(x, y):
            return np.linalg.norm(x - y, ord=2)

        def cosine_distance(x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        self.dist = euclidean_distance

        templates_folder = './positive/'
        templates_paths = ['template1.wav', 'template2.wav', 'template3.wav']
        self.templates = []
        for t_path in templates_paths:
            path = os.path.join(templates_folder, t_path)
            t_input, _ = torchaudio.load(path)
            with torch.inference_mode():
                t_output, _ = self.acoustic_model(t_input)
            self.templates.append(t_output.squeeze(0).numpy().astype(np.float64))
        # self.acoustic_model = torch.jit.trace(self.acoustic_model, (t_input,))

    def check(self, chunk):
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
                return True
            # time3 = time.time()
            # print(f'InferTime: {time2-time1}')
            # print(f'MatchTime: {time3-time2}')
        return False

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


if __name__ == '__main__':
    KWSer = KeywordSpotter()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        frames_per_buffer=3200,
        input=True,
        # stream_callback=callback
    )

    while stream:
        data = stream.read(3200)
        res = KWSer.check(data)
        print(f"Check Result: {res}")

    # data = stream.read(3200)
    # for _ in range(1000):
    #     res = KWSer.check(data)
