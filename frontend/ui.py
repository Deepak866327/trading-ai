import streamlit as st
import requests

st.set_page_config(page_title="AI Trading Assistant")

st.title("📊 AI Trading Assistant")
st.write("Enter stock name to get AI prediction")

stock = st.text_input("Stock (Example: RELIANCE.NS)")

if st.button("Analyze"):
    if stock == "":
        st.warning("Please enter a stock name")
    else:
        url = f"https://trading-ai-backend-zzd3.onrender.com/predict?stock={stock}"

        try:
            res = requests.get(url)
            data = res.json()

            if "signal" in data:
                st.success(f"Signal: {data['signal']}")
                st.write("Sentiment Score:", data['sentiment'])
            else:
                st.error(data)

        except:
            st.error("Backend not running ❌")