import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.title("📈 Quant Finance Dashboard")

# =========================
# STOCK SELECTOR
# =========================
ticker = st.selectbox("Select a stock", ["SPY", "AAPL", "TSLA", "NVDA"])

data = yf.download(ticker, start="2020-01-01", auto_adjust=True)

st.subheader("Stock Price")
st.line_chart(data["Close"])

# =========================
# RETURNS
# =========================
data["Returns"] = data["Close"].pct_change()

# =========================
# FEATURES
# =========================
data["Lag1"] = data["Returns"].shift(1)
data["Lag2"] = data["Returns"].shift(2)
data["Lag3"] = data["Returns"].shift(3)

data["Target"] = (data["Returns"].shift(-1) > 0).astype(int)

data = data.dropna()

X = data[["Lag1", "Lag2", "Lag3"]]
y = data["Target"]

split = int(len(data) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# =========================
# MODEL
# =========================
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

data["Prediction"] = model.predict(X)

# =========================
# STRATEGY
# =========================
data["Strategy_Returns"] = data["Prediction"].shift(1) * data["Returns"]

data["Market"] = (1 + data["Returns"]).cumprod()
data["Strategy"] = (1 + data["Strategy_Returns"].fillna(0)).cumprod()

# =========================
# PLOT
# =========================
st.subheader("Strategy vs Market")

fig, ax = plt.subplots()
ax.plot(data["Market"], label="Market")
ax.plot(data["Strategy"], label="ML Strategy")
ax.legend()

st.pyplot(fig)

# =========================
# METRICS
# =========================
total_return = data["Strategy"].iloc[-1] - 1

st.subheader("Metrics")
st.write(f"Total Return: {round(total_return*100,2)}%")
