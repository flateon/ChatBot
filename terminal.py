from asr import ASR, TerminalASRCallback
from llm import LLMPlugin, TerminalCallback
from tts import TTS, TTSLocal, TerminalTTSCallback

if __name__ == '__main__':
    asr = ASR(TerminalASRCallback())
    tts = TTS(TerminalTTSCallback())
    llm = LLMPlugin(TerminalCallback(tts))

    # mgs = llm.generate('今天天气怎么样？').content
    while True:
        user_msg = asr.start()
        mgs = llm.generate(user_msg).content
