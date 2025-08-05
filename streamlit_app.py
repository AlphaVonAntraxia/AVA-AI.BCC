import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go

# ---- Debug Info ----
st.set_page_config(page_title="AVA-AI.BCC", layout="wide", page_icon="📈")
st.title("Debug — AVA-AI.BCC")
st.write("✅ App started successfully")

# ---- Auto Refresh Every 15 minutes (900000ms) ----
st_autorefresh(interval=900000, limit=None, key="refresh")

# ---- Import models ----
st.write("📦 Importing models...")
from models.catboost_model import predict_with_catboost
from models.xgboost_model import predict_with_xgboost
from models.lightgbm_model import predict_with_lightgbm
from models.llm_model import predict_with_llm

# ---- Light/Dark Mode Toggle ----
theme_mode = st.sidebar.radio("Select Theme", ("Light", "Dark"))
if theme_mode == "Dark":
    st.markdown("""
        <style>
            body { background-color: #0e1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# ---- Title ----
st.title("📊 AVA-AI.BCC — Real-Time Crypto & Stock Prediction")

# ---- Tabs ----
tab1, tab2 = st.tabs(["📊 Crypto", "📊 Stocks"])

# ---- Crypto Tab ----
with tab1:
    st.subheader("💰 Cryptocurrency Prediction")
    selected_symbol = st.selectbox("Choose a Cryptocurrency", ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD"])
    period = st.selectbox("Timeframe", ["30d", "90d", "180d", "1y", "2y"], index=2)

    df = yf.download(selected_symbol, period=period, interval='1d')
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # --- Predictions ---
    lr_df, lr_pred = predict_with_catboost(df.copy())
    xgb_df, xgb_pred = predict_with_xgboost(df.copy())
    lgb_df, lgb_pred = predict_with_lightgbm(df.copy())
    prediction_date, llm_price = predict_with_llm(selected_symbol, df.copy())

    # --- Display Metrics ---
    st.metric("🔮 CatBoost Predicted Price (Next Day)", f"${lr_pred}")
    st.metric("🌱 XGBoost Predicted Price (Next Day)", f"${xgb_pred}")
    st.metric("🌍 LightGBM Predicted Price (Next Day)", f"${lgb_pred}")
    st.metric(f"🧠 ChatGPT/LLM Predicted Price ({prediction_date})", f"${llm_price}")

    # --- Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lr_df['Date'], y=lr_df['Close'], mode='lines+markers', name="Actual Price", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=lr_df['Date'], y=lr_df['Predicted'], name="CatBoost", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=xgb_df['Date'], y=xgb_df['Predicted'], name="XGBoost", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=lgb_df['Date'], y=lgb_df['Predicted'], name="LightGBM", line=dict(dash='solid')))

    fig.update_layout(title=f"{selected_symbol} Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ---- Stocks Tab ----
with tab2:
    st.subheader("📊 Stock Prediction")
    selected_symbol = st.selectbox("Choose a Stock", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
    period = st.selectbox("Timeframe", ["30d", "90d", "180d", "1y", "2y"], index=2, key="stock")

    df = yf.download(selected_symbol, period=period, interval='1d')
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # --- Predictions ---
    lr_df, lr_pred = predict_with_catboost(df.copy())
    xgb_df, xgb_pred = predict_with_xgboost(df.copy())
    lgb_df, lgb_pred = predict_with_lightgbm(df.copy())
    prediction_date, llm_price = predict_with_llm(selected_symbol, df.copy())

    # --- Display Metrics ---
    st.metric("🔮 CatBoost Predicted Price (Next Day)", f"${lr_pred}")
    st.metric("🌱 XGBoost Predicted Price (Next Day)", f"${xgb_pred}")
    st.metric("🌍 LightGBM Predicted Price (Next Day)", f"${lgb_pred}")
    st.metric(f"🧠 ChatGPT/LLM Predicted Price ({prediction_date})", f"${llm_price}")

    # --- Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lr_df['Date'], y=lr_df['Close'], mode='lines+markers', name="Actual Price", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=lr_df['Date'], y=lr_df['Predicted'], name="CatBoost", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=xgb_df['Date'], y=xgb_df['Predicted'], name="XGBoost", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=lgb_df['Date'], y=lgb_df['Predicted'], name="LightGBM", line=dict(dash='solid')))

    fig.update_layout(title=f"{selected_symbol} Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified", height=600)
    st.plotly_chart(fig, use_container_width=True)
