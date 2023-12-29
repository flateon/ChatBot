from asr import ASR, TerminalASRCallback
from llm import LLMPlugin, TerminalCallback
from tts import TTS, TTSLocal, TerminalTTSCallback

if __name__ == '__main__':
    asr = ASR(TerminalASRCallback())
    tts = TTS(TerminalTTSCallback())
    llm = LLMPlugin(TerminalCallback(tts))

    while True:
        user_msg = asr.start()
        # user_msg = input('user: ')
        mgs = llm.generate(user_msg).content
