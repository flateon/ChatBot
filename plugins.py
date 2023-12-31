import random

import requests

from keys import weather_keys, news_keys, top_news_keys


def get_weather(city: str = '南京'):
    forcast_url = f"https://api.seniverse.com/v3/weather/daily.json?key={weather_keys}&location={city}&language=zh-Hans&unit=c&start=0&days=3"
    now_url = f"https://api.seniverse.com/v3/weather/now.json?key={weather_keys}&location={city}&language=zh-Hans&unit=c&start=0&days=3"
    response = requests.get(forcast_url)
    data = response.json()['results'][0]

    for i in range(3):
        for k in list(data['daily'][i].keys()):
            if k in ['code_day', 'code_night', 'wind_direction', 'wind_direction_degree', 'wind_scale', 'wind_speed',
                     'rainfall', 'precip', 'humidity']:
                del data['daily'][i][k]

    response = requests.get(now_url)
    data.update(response.json()['results'][0])

    data['city'] = data['location']['name']
    data['today'] = data['daily'][0]['date']
    data.pop('location')
    data.pop('last_update')
    data['now'].pop('code')
    return str(data).replace('0.00', '0').replace("'", '')


def get_news(channel: str = '头条'):
    channel2id = {
        '头条':          0,
        'top':           0,
        'headline':      0,
        '国内':          7,
        'domestic':      7,
        '国际':          8,
        'world':         8,
        'international': 8,
        '娱乐':          10,
        'entertainment': 10,
        '体育':          12,
        'sports':        12,
        '科技':          13,
        'technology':    13,
        '军事':          27,
        'military':      27,
        'it':            22,
        'ai':            29,
        '财经':          32,
        'finance':       32,
    }

    if channel.lower() in channel2id:
        news_id = channel2id[channel.lower()]
    else:
        return 'ValueError: 新闻频道不存在'

    news = []
    if news_id == 0:
        top_url = f"https://v2.alapi.cn/api/new/toutiao?start=1&token={top_news_keys}&num=20"
        response = requests.get(top_url)
        data = response.json()
        if data['code'] != 200:
            news_id = 7
            # return data['msg']
            # return 'NetworkError: 请求失败'
        else:
            data = data['data']
            random.shuffle(data)
            # prefer news with digest
            data = sorted(data, key=lambda x: x['digest'] == '')
            for n in data[:3]:
                news.append({'title': n['title'], 'digest': n['digest']})

    if news_id >= 7:
        channel_url = f"https://apis.tianapi.com/allnews/index?key={news_keys}&num=10&col={news_id}"
        response = requests.get(channel_url)
        data = response.json()
        if data['code'] != 200:
            return 'NetworkError: 请求失败'

        data = data['result']['newslist']
        random.shuffle(data)
        # prefer news with digest
        data = sorted(data, key=lambda x: x['description'] == '')
        for n in data[:3]:
            news.append({'title': n['title'], 'digest': n['description']})

    return str(news).replace("'", '')


if __name__ == '__main__':
    # start = time.time()
    # print(get_weather('北京'))
    print(get_news('top'))
    # print(time.time() - start)
    # print(get_news('AI'))
