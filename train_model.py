import yfinance as yf
import pandas as pd
import ta
import xgboost as xgb
import joblib

# ------------------ STEP 1: DOWNLOAD DATA ------------------
stock = "RELIANCE.NS"   # training base stock (later model generalize karega)

df = yf.download(stock, period="1y", interval="1d")

# ------------------ STEP 2: FEATURES ------------------
close = df['Close'].squeeze()

df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
df['ema'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
df['ma50'] = close.rolling(50).mean()
df['ma200'] = close.rolling(200).mean()

# ------------------ STEP 3: TARGET (LABEL) ------------------
# Agar next day price bada → BUY (1), warna SELL (0)

df['future_return'] = df['Close'].shift(-1) - df['Close']
df['target'] = df['future_return'].apply(lambda x: 1 if x > 0 else 0)

# ------------------ STEP 4: CLEAN DATA ------------------
df.dropna(inplace=True)

# ------------------ STEP 5: TRAINING DATA ------------------
X = df[['rsi','ema','ma50','ma200']]
y = df['target']

# ------------------ STEP 6: TRAIN MODEL ------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X, y)

# ------------------ STEP 7: SAVE MODEL ------------------
joblib.dump(model, "backend/model.pkl")

print("Model trained successfully ✅")