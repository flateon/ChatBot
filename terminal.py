import pyaudio
from rich.progress import Progress, BarColumn, TextColumn

import KWS
from asr import ASR
from draw import Draw
from llm import LLMPlugin
from tts import TTS
from utils.terminal_utils import TerminalASRCallback, TerminalCallback, TerminalTTSCallback, Printer, \
    StreamWithVolumeDisplay, HelloByeHandler

if __name__ == '__main__':
    audio_player = pyaudio.PyAudio()
    audio_input = audio_player.open(16000, 1, pyaudio.paInt16, input=True, frames_per_buffer=1600)
    audio_output = audio_player.open(24000, 1, pyaudio.paInt16, output=True)

    try:
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn()) as progress:
            console = progress.console
            printer = Printer(console)
            console.clear()

            draw = Draw(console, './icon_gif/')
            draw.draw()

            vol = progress.add_task('[green]音量', total=100)
            audio_input_vol = StreamWithVolumeDisplay(audio_input,
                                                      lambda v: progress.update(vol, visible=True, completed=v))

            kws = KWS.KeywordSpotter('./Records/xiaoyuan')
            asr = ASR(TerminalASRCallback(logger=printer.get_live_print('User')), audio_input=audio_input_vol)
            tts = TTS(TerminalTTSCallback(audio_output), sample_rate=24000,
                      model={'zh': 'sambert-zhiwei-v1', 'en': 'sambert-cindy-v1'})
            start_end_handler = HelloByeHandler(tts, printer, audio_output, './Records/hellobye')

            while True:
                kws.detect(audio_input_vol)

                start_end_handler.hello()

                llm = LLMPlugin(TerminalCallback(tts, logger=printer.get_live_print('Bot')))
                while True:
                    user_msg = asr.start(idle_timeout=15)
                    if user_msg is None:
                        # asr idle timeout
                        start_end_handler.bye()
                        draw.draw()
                        break

                    progress.update(vol, visible=False)
                    # user_msg = input('user: ')
                    mgs = llm.generate(user_msg).content
    except Exception as e:
        print(e)

    finally:
        audio_input.close()
        audio_output.close()
        audio_player.terminate()
