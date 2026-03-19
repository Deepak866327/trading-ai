from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

def get_sentiment(news_list):
    scores = []

    for news in news_list:
        result = sentiment_model(news[:512])[0]
        
        if result['label'] == 'POSITIVE':
            scores.append(1)
        else:
            scores.append(-1)

    return sum(scores)/len(scores) if scores else 0