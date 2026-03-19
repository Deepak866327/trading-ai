from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key="e8979489d7424e80a60f9a5863eb7ab6")

def get_news(stock):
    articles = newsapi.get_everything(q=stock, language='en', page_size=10)
    
    news_list = []
    for article in articles['articles']:
        news_list.append(article['title'])
    
    return news_list