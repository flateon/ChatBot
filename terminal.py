import numpy as np
from rich.progress import Progress, BarColumn, TextColumn

from asr import ASR
from llm import LLMPlugin
from tts import TTS
from utils.terminal_utils import TerminalASRCallback, TerminalCallback, TerminalTTSCallback, Printer

if __name__ == '__main__':
    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn()) as progress:
        vol = progress.add_task('[green]Volume', total=100)
        console = progress.console
        printer = Printer(console)
        console.clear()

        asr = ASR(TerminalASRCallback(logger=printer.get_live_print('User')))
        tts = TTS(TerminalTTSCallback(sample_rate=24000), sample_rate=24000,
                  model={'zh': 'sambert-zhiwei-v1', 'en': 'sambert-cindy-v1'})
        llm = LLMPlugin(TerminalCallback(tts, logger=printer.get_live_print('Bot')))

        while True:
            asr.start()

            while not asr.is_done():
                audio = asr.callback.read()

                audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768
                for a in audio_np.reshape(10, -1):
                    volume = np.log10(np.mean(a ** 2) + 1e-10) * 20
                    progress.update(vol, completed=volume + 100, visible=True)

                asr.send_audio_frame(audio)
            progress.update(vol, visible=False)

            asr.stop()
            user_msg = asr.get_sentence()
            # user_msg = input('user: ')
            mgs = llm.generate(user_msg).content
