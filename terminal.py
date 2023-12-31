import numpy as np
import pyaudio
from rich.progress import Progress, BarColumn, TextColumn

from asr import ASR
from llm import LLMPlugin
from tts import TTS
from utils.terminal_utils import TerminalASRCallback, TerminalCallback, TerminalTTSCallback, Printer, \
    StreamWithVolumeDisplay

if __name__ == '__main__':
    audio_player = pyaudio.PyAudio()
    audio_input = audio_player.open(16000, 1, pyaudio.paInt16, input=True, frames_per_buffer=1600)
    audio_output = audio_player.open(24000, 1, pyaudio.paInt16, output=True)

    try:
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn()) as progress:
            vol = progress.add_task('[green]Volume', total=100)
            console = progress.console
            printer = Printer(console)
            console.clear()
            audio_input_vol = StreamWithVolumeDisplay(audio_input,
                                                      lambda v: progress.update(vol, visible=True, completed=v))

            asr = ASR(TerminalASRCallback(logger=printer.get_live_print('User')), audio_input=audio_input_vol)
            tts = TTS(TerminalTTSCallback(audio_output), sample_rate=24000,
                      model={'zh': 'sambert-zhiwei-v1', 'en': 'sambert-cindy-v1'})
            llm = LLMPlugin(TerminalCallback(tts, logger=printer.get_live_print('Bot')))

            while True:
                asr.start()
                progress.update(vol, visible=False)

                user_msg = asr.get_sentence()
                # user_msg = input('user: ')
                mgs = llm.generate(user_msg).content
    except Exception as e:
        print(e)

    finally:
        audio_input.close()
        audio_output.close()
        audio_player.terminate()
