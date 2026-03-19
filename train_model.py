import yfinance as yf
import pandas as pd
import ta
import xgboost as xgb
import joblib

stock = "RELIANCE.NS"

df = yf.download(stock, period="1y", interval="1d")

close = df['Close'].squeeze()

# TECHNICAL
df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
df['ema'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
df['ma50'] = close.rolling(50).mean()
df['ma200'] = close.rolling(200).mean()

# TARGET
df['future_return'] = df['Close'].shift(-1) - df['Close']
df['target'] = df['future_return'].apply(lambda x: 1 if x > 0 else 0)

df.dropna(inplace=True)

# ---------------- FUNDAMENTALS ----------------
ticker = yf.Ticker(stock)
info = ticker.info

pe = info.get("trailingPE", 0)
roe = info.get("returnOnEquity", 0)
debt = info.get("debtToEquity", 0)
profit_margin = info.get("profitMargins", 0)

# Add constant columns (same for all rows)
df["pe"] = pe
df["roe"] = roe
df["debt"] = debt
df["profit_margin"] = profit_margin

# FEATURES
X = df[['rsi','ema','ma50','ma200','pe','roe','debt','profit_margin']]
y = df['target']

# MODEL
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    eval_metric='logloss'
)

model.fit(X, y)

joblib.dump(model, "backend/model.pkl")

print("Hybrid model trained successfully 🚀")