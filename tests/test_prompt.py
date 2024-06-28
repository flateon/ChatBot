import unittest

from llm import LLMPlugin, LLMCallback


class TestCallback(LLMCallback):
    text = ''

    def on_event(self, msg):
        self.text = msg.content

    def on_complete(self):
        print(self.text)


class TestChinesePrompt(unittest.TestCase):

    def test_weather(self):
        test_case = [
            ['今天北京的天气怎么样'],
            ['今天的天气怎么样'],
            ['昨天的天气怎么样'],
            ['南京，不对，北京今天天气怎么样？'],
        ]
        for case in test_case:
            for text in case:
                LLMPlugin(TestCallback()).generate(text)
        self.assertEqual(input('Is the Prompt OK?[y/n]'), 'y')

    def test_news(self):
        test_case = [
            ['今天有什么新闻'],
            ['告诉我一些科技新闻'],
            ['昨天有什么新闻'],
            ['如何获取今天的新闻'],
            ['南京有什么新闻'],
        ]
        for case in test_case:
            for text in case:
                LLMPlugin(TestCallback()).generate(text)
        self.assertEqual(input('Is the Prompt OK?[y/n]'), 'y')

    def test_python(self):
        test_case = [
            ['根号10是多少'],
            ['86寸电视有长多少米'],
            ['我身高一米八，体重七十公斤，我的BMI是多少']
        ]
        for case in test_case:
            for text in case:
                LLMPlugin(TestCallback()).generate(text)
        self.assertEqual(input('Is the Prompt OK?[y/n]'), 'y')


class TestEnglishPrompt(unittest.TestCase):

    def test_weather(self):
        test_case = [
            ['How is the weather in Beijing today?'],
            ['How is the weather today?'],
            ['How is the weather in yesterday?'],
        ]
        for case in test_case:
            for text in case:
                LLMPlugin(TestCallback()).generate(text)
        self.assertEqual(input('Is the Prompt OK?[y/n]'), 'y')

    def test_news(self):
        test_case = [
            ["What's the news today"],
            ['Tell me some technology news'],
            ["What's the news yesterday"],
            ["How to get today's news"]
        ]
        for case in test_case:
            for text in case:
                LLMPlugin(TestCallback()).generate(text)
        self.assertEqual(input('Is the Prompt OK?[y/n]'), 'y')