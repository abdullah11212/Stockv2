import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import warnings
warnings.filterwarnings("ignore")

# ==========================
# Reproducibility
# ==========================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================
# Load Data
# ==========================
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="max")
    data['Price'] = data['Adj Close'] if 'Adj Close' in data else data['Close']
    data.dropna(inplace=True)
    return data

# ==========================
# HESTON MODEL
# ==========================
def heston_model(data):

    data = data.copy()
    data['Return'] = np.log(data['Price'] / data['Price'].shift(1))
    data.dropna(inplace=True)

    returns = data['Return'].values
    dt = 1 / 252

    def heston_loglike(params):
        kappa, theta, sigma_v, v0 = params
        vt = v0
        ll = 0.0
        for r in returns:
            vt = max(vt + kappa * (theta - vt) * dt, 1e-8)
            var = vt * dt
            ll += -0.5 * (np.log(2*np.pi*var) + (r**2)/var)
        return -ll

    params0 = [1.0, returns.var(), 0.3, returns.var()]
    bounds = [(1e-3, 10), (1e-6, 1), (1e-3, 2), (1e-6, 1)]

    res = minimize(heston_loglike, params0, bounds=bounds, method="L-BFGS-B")
    kappa, theta, sigma_v, v0 = res.x

    v = v0
    vols = []
    for _ in range(len(data)):
        dv = kappa*(theta - v)*dt + sigma_v*np.sqrt(v*dt)*np.random.normal()
        v = max(v + dv, 1e-8)
        vols.append(np.sqrt(v))

    data['Volatility'] = vols

    return {
        "kappa": kappa,
        "theta": theta,
        "sigma_v": sigma_v,
        "v0": v
    }, data

# ==========================
# Prepare GRU Sequences
# ==========================
def prepare_sequences(data, seq_len=60):

    features = data[['Price', 'Volatility']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))

    return (
        X[:split], X[split:],
        y[:split], y[split:],
        scaler, scaled
    )

# ==========================
# Build GRU
# ==========================
def build_gru(X_train):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ==========================
# Future Forecast (GRU + Heston)
# ==========================
def future_forecast(model, data, scaler, scaled, seq_len, n_days, heston):

    kappa, theta, sigma_v = heston["kappa"], heston["theta"], heston["sigma_v"]
    dt = 1/252

    last_seq = scaled[-seq_len:].copy()
    last_price = data['Price'].iloc[-1]
    v = data['Volatility'].iloc[-1] ** 2

    future_prices, future_vols = [], []

    for _ in range(n_days):

        x = last_seq.reshape(1, seq_len, 2)
        price_scaled = model.predict(x, verbose=0)[0, 0]

        dv = kappa*(theta - v)*dt + sigma_v*np.sqrt(v*dt)*np.random.normal()
        v = max(v + dv, 1e-8)
        vol = np.sqrt(v)

        temp = np.array([[last_price, vol]])
        temp_scaled = scaler.transform(temp)

        new_step = np.array([[price_scaled, temp_scaled[0,1]]])
        last_seq = np.vstack([last_seq[1:], new_step])

        real_price = scaler.inverse_transform(new_step)[0,0]

        future_prices.append(real_price)
        future_vols.append(vol)
        last_price = real_price

    dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=n_days,
        freq="B"
    )

    return dates, future_prices, future_vols

# ==========================
# STREAMLIT APP
# ==========================
def main():

    st.set_page_config("GRU + Heston Stock Forecast", layout="wide")
    st.title("📈 GRU + Heston Stock Price Forecast")

    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    n_days = st.sidebar.slider("Forecast Days", 5, 90, 30)

    if st.sidebar.button("Run Model"):

        data = load_data(ticker)
        st.success(f"Loaded {len(data)} rows")

        heston_params, data = heston_model(data)

        X_train, X_test, y_train, y_test, scaler, scaled = prepare_sequences(data)

        model = build_gru(X_train)
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=25,
            batch_size=32,
            verbose=0
        )

        pred = model.predict(X_test, verbose=0)
        temp = np.zeros((len(pred), 2))
        temp[:,0] = pred[:,0]
        preds = scaler.inverse_transform(temp)[:,0]

        actual = scaler.inverse_transform(
            np.column_stack([y_test, np.zeros(len(y_test))])
        )[:,0]

        st.subheader("📊 Test Metrics")
        st.write("MAE:", mean_absolute_error(actual, preds))
        st.write("RMSE:", np.sqrt(mean_squared_error(actual, preds)))
        st.write("R²:", r2_score(actual, preds))

        dates, prices, vols = future_forecast(
            model, data, scaler, scaled, 60, n_days, heston_params
        )

        st.subheader("🔮 Future Forecast")
        df = pd.DataFrame({
            "Date": dates,
            "Price": prices,
            "Volatility": vols
        })
        st.dataframe(df)

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(data.index[-150:], data['Price'].tail(150), label="Historical")
        ax.plot(dates, prices, '--o', label="Forecast")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
