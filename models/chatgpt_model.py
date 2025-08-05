import openai
import pandas as pd
import numpy as np
import os
import streamlit as st

openai.api_key = st.secrets["openai"]["api_key"]

def summarize_market_trend(df):
    df = df.copy()
    df['daily_change'] = df['Close'].pct_change()
    trend = df['daily_change'].tail(5).mean()
    if trend > 0.01:
        return "The market is in a strong uptrend."
    elif trend > 0:
        return "The market is in a mild uptrend."
    elif trend < -0.01:
        return "The market is in a strong downtrend."
    elif trend < 0:
        return "The market is in a mild downtrend."
    else:
        return "The market is moving sideways."

def predict_with_chatgpt(data):
    df = data.copy()
    trend_summary = summarize_market_trend(df)
    latest_price = df['Close'].iloc[-1]
    ticker = df.columns.name or "the asset"

    prompt = f"""
You are a financial AI assistant. Based on the following market trend and the latest closing price, predict the next day's price for {ticker}:

Trend Summary: {trend_summary}
Latest Closing Price: ${latest_price:.2f}

Give your prediction and a short reasoning. Format:
"Predicted Price: $xxx.xx
Reason: ..."
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100,
    )

    content = response.choices[0].message["content"]
    lines = content.splitlines()
    predicted_price = None
    reason = ""

    for line in lines:
        if "Predicted Price" in line:
            try:
                predicted_price = float(line.split("$")[1].strip())
            except:
                predicted_price = None
        elif "Reason" in line:
            reason = line

    return predicted_price, reason