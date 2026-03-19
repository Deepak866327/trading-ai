import streamlit as st
import requests
import yfinance as yf
import plotly.graph_objs as go
import time

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

st.title("📊 AI Trading Dashboard")

# ---------------- SIDEBAR ----------------
stock = st.sidebar.text_input("Stock", "RELIANCE")
analyze = st.sidebar.button("Analyze")

# ---------------- AUTO REFRESH ----------------
if "last_run" not in st.session_state:
    st.session_state.last_run = time.time()

if time.time() - st.session_state.last_run > 30:
    st.rerun()

# ---------------- CHART ----------------
col1, col2 = st.columns([2,1])

with col1:
    df = yf.download(stock+".NS", period="6mo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"]))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig)

# ---------------- AI PANEL ----------------
with col2:
    if analyze:
        res = requests.get(f"http://127.0.0.1:8000/predict?stock={stock}")
        data = res.json()

        if "signal" in data:
            st.success(data["signal"])
            st.metric("Confidence", f"{data['confidence']}%")
            st.metric("Sentiment", data["sentiment"])
            st.info(data["explanation"])
        else:
            st.error(data)

# ---------------- SECTOR ----------------
st.subheader("🏭 Sector Analysis")

try:
    res = requests.get("http://127.0.0.1:8000/sector-analysis")
    data = res.json()

    for sector, info in data.items():
        st.write(f"{sector}: {info['trend']} ({info['score']})")
except:
    st.warning("Sector error")

# ---------------- TOP STOCKS ----------------
st.subheader("🔥 Top AI Picks")

try:
    res = requests.get("http://127.0.0.1:8000/top-stocks")
    data = res.json()

    for stock in data:
        st.write(f"{stock['stock']} → {stock['confidence']}%")
except:
    st.warning("Top stocks error")

# ---------------- CHAT AI ----------------
st.subheader("🤖 Ask AI")

query = st.text_input("Should I buy TCS?")

if query:
    stock_name = query.upper().split()[-1]
    res = requests.get(f"https://trading-ai-backend-zzd3.onrender.com/predict?stock={stock_name}"
)
    data = res.json()

    if "signal" in data:
        st.write(f"{stock_name}: {data['signal']}")
        st.write(f"Confidence: {data['confidence']}%")
        st.write(data["explanation"])