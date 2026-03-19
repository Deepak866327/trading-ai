from fastapi import FastAPI
import joblib
import yfinance as yf
import pandas as pd
import ta
from news import get_news
from sentiment import get_sentiment

app = FastAPI()

model = joblib.load("model.pkl")

# ------------------ SYMBOL ------------------
def normalize_symbol(user_input):
    stock = user_input.upper().strip()
    if stock.endswith(".NS") or stock.endswith(".BO"):
        return stock
    return stock + ".NS"

# ------------------ FEATURES ------------------
def get_features(stock):
    stock = normalize_symbol(stock)

    df = yf.download(stock, period="1y", interval="1d")

    if df.empty:
        raise Exception("Invalid stock")

    close = df['Close'].squeeze()

    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['ema'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df['ma50'] = close.rolling(50).mean()
    df['ma200'] = close.rolling(200).mean()

    df = df[['rsi','ema','ma50','ma200']]
    df.dropna(inplace=True)

    return df.tail(1)

# ------------------ EXPLANATION ------------------
def generate_explanation(rsi, sentiment, prediction):
    if prediction == 1:
        if rsi > 50 and sentiment > 0:
            return "Bullish momentum with positive sentiment."
        elif rsi > 50:
            return "Uptrend based on technical indicators."
        else:
            return "Model predicts upside despite weak signals."
    else:
        if rsi < 50 and sentiment < 0:
            return "Bearish trend with negative sentiment."
        elif rsi < 50:
            return "Downtrend based on indicators."
        else:
            return "Model predicts downside risk."

# ------------------ MAIN API ------------------
@app.get("/predict")
def predict(stock: str):
    try:
        X = get_features(stock)

        news = get_news(stock)
        sentiment = float(get_sentiment(news))

        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][pred])

        rsi = float(X['rsi'].values[0])

        explanation = generate_explanation(rsi, sentiment, pred)

        return {
            "stock": normalize_symbol(stock),
            "signal": "BUY 📈" if pred==1 else "SELL 📉",
            "confidence": round(prob*100,2),
            "sentiment": round(sentiment,2),
            "explanation": explanation
        }

    except Exception as e:
        return {"error": str(e)}

# ------------------ SECTOR DATA ------------------
SECTORS = {
    "BANKING": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS"],
    "IT": ["TCS.NS","INFY.NS","WIPRO.NS"],
    "AUTO": ["TATAMOTORS.NS","MARUTI.NS"],
    "FMCG": ["ITC.NS","HINDUNILVR.NS"],
    "METAL": ["TATASTEEL.NS","JSWSTEEL.NS"]
}

# ------------------ SECTOR ANALYSIS ------------------
@app.get("/sector-analysis")
def sector_analysis():
    result = {}

    for sector, stocks in SECTORS.items():
        signals = []

        for stock in stocks:
            try:
                X = get_features(stock)
                pred = int(model.predict(X)[0])
                signals.append(pred)
            except:
                continue

        if signals:
            score = sum(signals)/len(signals)

            trend = "Bullish 📈" if score>0.6 else "Bearish 📉" if score<0.4 else "Neutral ⚖️"

            result[sector] = {
                "score": round(score,2),
                "trend": trend
            }

    return result

# ------------------ TOP STOCKS ------------------
@app.get("/top-stocks")
def top_stocks():
    stocks = [
        "RELIANCE.NS","TCS.NS","INFY.NS",
        "HDFCBANK.NS","SBIN.NS","ITC.NS","TATASTEEL.NS"
    ]

    results = []

    for stock in stocks:
        try:
            X = get_features(stock)
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][pred])

            if pred == 1:
                results.append({
                    "stock": stock,
                    "confidence": round(prob*100,2)
                })

        except:
            continue

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    return results[:5]