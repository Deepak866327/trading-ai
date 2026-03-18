import yfinance as yf
import pandas as pd
import ta
import xgboost as xgb
import joblib

df = yf.download("RELIANCE.NS", start="2018-01-01")

# FIX
close = df['Close'].squeeze()

df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
df['ema'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()

df['target'] = (close.shift(-1) > close).astype(int)

df.dropna(inplace=True)

X = df[['rsi','ema']]
y = df['target']

model = xgb.XGBClassifier()
model.fit(X, y)

joblib.dump(model, "backend/model.pkl")

print("Model trained successfully ✅")