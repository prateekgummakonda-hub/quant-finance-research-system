import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Quant Finance Dashboard", layout="centered")

st.title("Quant Finance Dashboard")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox(
    "Select a stock",
    [
        "SPY", "AAPL", "TSLA", "NVDA",
        "MSFT", "GOOGL", "AMZN",
        "META", "NFLX", "AMD", "INTC"
    ]
)
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))

# -----------------------------
# Download data
# -----------------------------
data = yf.download(ticker, start=start_date, auto_adjust=True)

if data.empty:
    st.error("No data found for this ticker.")
    st.stop()

# Keep only Close
data = data[["Close"]].copy()

st.subheader(f"{ticker} Price Chart")
st.line_chart(data["Close"])

# -----------------------------
# Returns
# -----------------------------
data["Returns"] = data["Close"].pct_change()

# -----------------------------
# Features
# -----------------------------
data["Lag1"] = data["Returns"].shift(1)
data["Lag2"] = data["Returns"].shift(2)
data["Lag3"] = data["Returns"].shift(3)

# Predict whether next day return is positive
data["Target"] = (data["Returns"].shift(-1) > 0).astype(int)

data = data.dropna().copy()

if len(data) < 50:
    st.error("Not enough data after processing.")
    st.stop()

X = data[["Lag1", "Lag2", "Lag3"]]
y = data["Target"]

split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions only on test set
data["Prediction"] = np.nan
data.loc[X_test.index, "Prediction"] = model.predict(X_test)

# Fill missing predictions with 0 before test period
data["Prediction"] = data["Prediction"].fillna(0)

# -----------------------------
# Strategy returns
# IMPORTANT:
# We multiply prediction by RETURNS, not by price
# -----------------------------
data["Strategy_Returns"] = data["Prediction"].shift(1).fillna(0) * data["Returns"]

# -----------------------------
# Curves
# IMPORTANT:
# Both curves must start from 1 and compound returns
# -----------------------------
data["Market_Curve"] = (1 + data["Returns"].fillna(0)).cumprod()
data["Strategy_Curve"] = (1 + data["Strategy_Returns"].fillna(0)).cumprod()

# -----------------------------
# Plot
# -----------------------------
st.subheader("Strategy vs Market")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data["Market_Curve"], label="Market", linewidth=2)
ax.plot(data.index, data["Strategy_Curve"], label="ML Strategy", linewidth=2)
ax.legend()
ax.set_title(f"{ticker}: Market vs ML Strategy")
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Metrics
# -----------------------------
total_return_market = data["Market_Curve"].iloc[-1] - 1
total_return_strategy = data["Strategy_Curve"].iloc[-1] - 1

st.subheader("Metrics")
col1, col2 = st.columns(2)

with col1:
    st.metric("Market Total Return", f"{total_return_market:.2%}")

with col2:
    st.metric("ML Strategy Total Return", f"{total_return_strategy:.2%}")
# -----------------------------
# Next-day prediction
# -----------------------------
st.subheader("Next Trading Day Prediction")

latest_features = X.iloc[[-1]]
latest_prob = model.predict_proba(latest_features)[0][1]
latest_pred = model.predict(latest_features)[0]

if latest_pred == 1:
    direction = "UP"
else:
    direction = "DOWN"

st.metric("Predicted Direction", direction)
st.metric("Confidence", f"{latest_prob:.2%}")
# -----------------------------
# Show prediction table
# -----------------------------
st.subheader("Latest Signals")
display_df = data[["Close", "Returns", "Prediction", "Market_Curve", "Strategy_Curve"]].tail(10)
st.dataframe(display_df)
