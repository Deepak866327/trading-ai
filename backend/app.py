from fastapi import FastAPI
import joblib
import yfinance as yf
import pandas as pd
import ta
from news import get_news
from sentiment import get_sentiment

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

def get_features(stock):
    df = yf.download(stock, period="6mo")

    close = df['Close'].squeeze()

    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['ema'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()

    df.dropna(inplace=True)

    return df[['rsi','ema']].iloc[-1:]

@app.get("/")
def home():
    return {"message": "AI Trading API Running 🚀"}

@app.get("/predict")
def predict(stock: str):
    try:
        X = get_features(stock)

        # NEWS PART
        news = get_news(stock)
        sentiment_score = get_sentiment(news)

        pred = model.predict(X)[0]

        return {
            "stock": stock,
            "signal": "BUY 📈" if pred==1 else "SELL 📉",
            "sentiment": sentiment_score
        }

    except Exception as e:
        return {"error": str(e)}