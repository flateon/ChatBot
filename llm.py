import random
import re
from datetime import date

import dashscope
from dashscope.api_entities.dashscope_response import Message, Role
from dashscope.audio.tts import ResultCallback

from keys import dashscope_keys
from plugins import get_weather, get_news
from prompts import SYSTEM_PROMPT, TOOL_PROMPT
from tts import TerminalTTSCallback, TTS

dashscope.api_key = dashscope_keys


class LLMCallback:
    def __init__(self):
        pass

    def on_open(self):
        pass

    def on_event(self, result: Message):
        pass

    def on_complete(self):
        pass

    def on_error(self, response: Message):
        pass

    def on_close(self):
        pass


class LLM:
    def __init__(self, callback: LLMCallback = None, model='qwen-max', max_token=200, enable_search=False, stream=True,
                 **kwargs):
        self.callback = callback if callback is not None else LLMCallback()
        self.date = date.today().strftime('%Y-%m-%d')
        self.seed = random.randint(0, 2 ** 16)
        self.messages = [Message(role=Role.SYSTEM, content=SYSTEM_PROMPT.format(self.date))]
        self.model = model
        self.max_token = max_token
        self.enable_search = enable_search
        self.stream = stream
        self.kwargs = kwargs

    def generate(self, messages):
        self.messages.append(Message(role=Role.USER, content=messages))

        response = dashscope.Generation.call(
            model=self.model,
            messages=self.messages,
            stream=self.stream,
            seed=self.seed,
            result_format='message',
            enable_search=self.enable_search,
            max_tokens=self.max_token,
            **self.kwargs
        )
        self.callback.on_open()

        if self.stream is True:
            resp = None
            for resp in response:
                if resp.status_code != 200:
                    raise Exception(f'Error code {resp.code}: {resp.message}')
                self.callback.on_event(Message.from_generation_response(resp))
            response = resp
            self.callback.on_complete()
        else:
            if response.status_code != 200:
                raise Exception(f'Error code {response.code}: {response.message}')

        reply_message = Message.from_generation_response(response)
        self.messages.append(reply_message)

        return reply_message


class LLMPlugin(LLM):
    def __init__(self, callback: LLMCallback = None, model='qwen-max', max_token=200, enable_search=False, stream=True,
                 **kwargs):
        super().__init__(callback=callback, model=model, max_token=max_token, enable_search=enable_search,
                         stream=stream, **kwargs)

    def generate(self, messages):
        reply = super().generate(messages)

        if 'get_weather' in reply.content.lower():
            tool_func = get_weather
            args = (re.search(r'get_weather\((.*?)\)', reply.content).group(1),)

        elif 'get_news' in reply.content.lower():
            tool_func = get_news
            args = (re.search(r'get_news\((.*?)\)', reply.content).group(1),)
        else:
            # no tool used
            return reply

        try:
            api_response = tool_func(*args)
        except Exception as e:
            api_response = f'调用错误：{e}'

        user_request = self.messages[-2].content
        reply = super().generate(TOOL_PROMPT.format(api_response, user_request))

        return reply


class TerminalCallback(LLMCallback):
    def __init__(self, tts: TTS = None):
        super().__init__()
        self.head_idx = 0
        self.read = []
        self.tts = tts if tts is not None else None

    def on_open(self):
        print('\nassistant: ', end='')

    def on_complete(self):
        print('')
        self.head_idx = 0
        self.read = []

    def on_event(self, msg):
        text = msg.content
        if '`' in text or 'get' in text:
            if 'get_' in text:
                text = re.sub(r'`?get_(news|weather)?\(?.*?(\)|$)`?', '', text).strip()
            else:
                return
        print(text[self.head_idx:len(text)], end='', flush=True)

        if self.tts is not None:
            sentences = re.split(r"([.。!！?？;；:：\n])", text)
            # discard last sentence
            for sentence in [t + p for t, p in zip(sentences[::2], sentences[1::2])]:
                if sentence.strip() in '.。!！?？;；:：\n' or sentence in self.read:
                    pass
                else:
                    self.tts.say(sentence)
                    self.read.append(sentence)

        self.head_idx = len(text)


def demo():
    callback = TerminalCallback()
    llm = LLMPlugin(callback)
    for i in range(10):
        reply = llm.generate(input('user: '), )


def test():
    test_case = ['今天南京的天气怎么样', '今天北京的天气怎么样', '昨天的天气怎么样', '今天有什么新闻',
                 '今天有什么AI新闻']
    callback = TerminalCallback()
    llm = LLMPlugin(callback)
    for t in test_case:
        print(f'\nuser: {t}')
        reply = llm.generate(t)


if __name__ == '__main__':
    # demo()
    test()
