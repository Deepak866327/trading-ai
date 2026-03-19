from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(news_list):
    if not news_list:
        return 0

    scores = []
    for news in news_list:
        score = analyzer.polarity_scores(news)['compound']
        scores.append(score)

    return sum(scores)/len(scores)