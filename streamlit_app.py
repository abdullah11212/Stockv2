# ============================================
# IMPORTS
# ============================================
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import tensorflow as tf
import random
import warnings
warnings.filterwarnings("ignore")

# ============================================
# SEED (IMPORTANT)
# ============================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================
# DATA LOADER
# ============================================
def load_data(ticker):
    data = yf.download(ticker, period="max")
    if "Adj Close" in data.columns:
        data["Price"] = data["Adj Close"]
    else:
        data["Price"] = data["Close"]
    return data.dropna()

# ============================================
# FEATURE ENGINEERING
# ============================================
def add_features(data):
    data["Log_Return"] = np.log(data["Price"] / data["Price"].shift(1))
    data["Volatility"] = data["Log_Return"].rolling(20).std()
    data.dropna(inplace=True)
    return data

# ============================================
# SEQUENCE PREPARATION
# ============================================
def prepare_sequences(data, seq_len=60):
    features = data[["Price", "Volatility"]].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    return (
        X[:split], X[split:],
        y[:split], y[split:],
        scaler, scaled
    )

# ============================================
# GRU MODEL
# ============================================
def build_gru_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        GRU(64, return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=25,
        batch_size=32,
        verbose=0
    )

    return model

# ============================================
# MODEL EVALUATION
# ============================================
def evaluate_model(model, X_test, y_test, scaler, scaled):
    preds = model.predict(X_test, verbose=0)

    tmp = np.zeros((len(preds), scaled.shape[1]))
    tmp[:, 0] = preds[:, 0]
    pred_prices = scaler.inverse_transform(tmp)[:, 0]

    tmp[:, 0] = y_test
    actual_prices = scaler.inverse_transform(tmp)[:, 0]

    return {
        "MAE": mean_absolute_error(actual_prices, pred_prices),
        "RMSE": np.sqrt(mean_squared_error(actual_prices, pred_prices)),
        "R2": r2_score(actual_prices, pred_prices)
    }

# ============================================
# ✅ CORRECT FORECAST (FIXED)
# ============================================
def future_forecast(
    model,
    data,
    scaler,
    scaled_features,
    seq_len=60,
    n_days=30
):
    last_sequence = scaled_features[-seq_len:].copy()
    future_steps = []

    for _ in range(n_days):
        input_seq = last_sequence.reshape(
            1, seq_len, scaled_features.shape[1]
        )

        next_price_scaled = model.predict(input_seq, verbose=0)[0, 0]
        last_vol_scaled = last_sequence[-1, 1]

        next_step = np.array([next_price_scaled, last_vol_scaled])

        last_sequence = np.vstack([last_sequence[1:], next_step])
        future_steps.append(next_step)

    future_steps = np.array(future_steps)
    future_prices = scaler.inverse_transform(future_steps)[:, 0]

    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=n_days,
        freq="B"
    )

    return future_dates, future_prices

# ============================================
# STREAMLIT APP
# ============================================
def main():
    st.set_page_config(page_title="Stock Forecast", layout="wide")
    st.title("📈 Stock Price Forecast (GRU)")

    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    n_days = st.sidebar.number_input("Forecast Days", 1, 365, 30)
    run = st.sidebar.button("Run Model")

    if run:
        data = load_data(ticker)
        data = add_features(data)

        X_train, X_test, y_train, y_test, scaler, scaled = prepare_sequences(data)

        model = build_gru_model(X_train, y_train, X_test, y_test)

        metrics = evaluate_model(model, X_test, y_test, scaler, scaled)

        future_dates, future_prices = future_forecast(
            model, data, scaler, scaled, 60, n_days
        )

        st.subheader("📊 Model Performance")
        st.json(metrics)

        st.subheader("🔮 Price Forecast")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            data.index[-200:],
            data["Price"].iloc[-200:],
            label="Historical"
        )
        ax.plot(
            future_dates,
            future_prices,
            linestyle="--",
            label="Forecast"
        )
        ax.legend()
        ax.set_title(f"{ticker} Price Forecast")

        st.pyplot(fig)

if __name__ == "__main__":
    main()
