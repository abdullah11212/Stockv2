import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
import random
import tensorflow as tf

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================================
# FUNCTION: Load historical stock data
# ============================================================================
def load_data(ticker, end_date=None):
    """
    Download historical stock data using yfinance.
    Uses Adjusted Close if available, otherwise Close.
    """
    data = yf.download(ticker, period="max", end=end_date)
    if data.empty:
        return data
    
    if 'Adj Close' in data.columns:
        data['Price'] = data['Adj Close']
    else:
        data['Price'] = data['Close']
    
    return data[['Price']]

# ============================================================================
# FUNCTION: Statistical analysis on log returns
# ============================================================================
def statistical_analysis(data):
    log_returns = np.log(data['Price'] / data['Price'].shift(1)).dropna()
    
    mean_return = float(log_returns.mean())
    volatility = float(log_returns.std(ddof=1))
    skewness = float(log_returns.skew())
    excess_kurtosis = float(log_returns.kurt())
    pearson_kurtosis = excess_kurtosis + 3
    
    ks_D, ks_p = stats.kstest(log_returns, 'norm', args=(mean_return, volatility))
    shapiro_stat, shapiro_p = stats.shapiro(log_returns)
    ad_result = stats.anderson(log_returns, dist='norm')
    
    threshold = 3 * volatility
    data['Log_Return'] = np.log(data['Price'] / data['Price'].shift(1))
    data['Jump'] = (np.abs(data['Log_Return'] - mean_return) > threshold).astype(int)
    data.dropna(inplace=True)
    jump_count = int(data['Jump'].sum())
    jump_percent = 100 * jump_count / len(data)
    
    if pearson_kurtosis > 3.1:
        tail_type = "Leptokurtic (Heavy tails → jumps likely)"
    elif pearson_kurtosis < 2.9:
        tail_type = "Platykurtic (Thin tails)"
    else:
        tail_type = "Mesokurtic (≈ Normal)"
    
    normality_conclusion = "❌ Reject Normality" if ad_result.statistic > ad_result.critical_values[2] else "✔ Fail to Reject Normality"
    
    return {
        'mean_return': mean_return, 'volatility': volatility, 'skewness': skewness,
        'excess_kurtosis': excess_kurtosis, 'pearson_kurtosis': pearson_kurtosis,
        'tail_type': tail_type, 'ks_D': ks_D, 'ks_p': ks_p,
        'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p,
        'ad_statistic': ad_result.statistic, 'ad_critical_values': ad_result.critical_values,
        'normality_conclusion': normality_conclusion,
        'threshold': threshold, 'jump_count': jump_count, 'jump_percent': jump_percent,
        'data': data
    }

# ============================================================================
# FUNCTION: Heston stochastic volatility model
# ============================================================================
def heston_model(data):
    data['Return'] = 100 * data['Price'].pct_change()
    data.dropna(inplace=True)
    
    def heston_loglike(params, returns):
        kappa, theta, sigma_v, rho, v0 = params
        dt = 1/252
        vt = v0
        loglike = 0
        for r in returns:
            dv = kappa*(theta - vt)*dt + sigma_v*np.sqrt(max(vt, 1e-8))*np.random.normal()
            vt = max(vt + dv, 1e-8)
            var_ret = vt * dt
            loglike += -0.5*(np.log(2*np.pi*var_ret) + (r**2)/var_ret)
        return -loglike
    
    params0 = [1.0, 0.02, 0.2, -0.3, 0.02]
    res = minimize(heston_loglike, params0, args=(data['Return'].values,), method='Nelder-Mead')
    kappa, theta, sigma_v, rho, v0 = res.x
    
    dt = 1/252
    v_path = []
    v = v0
    for _ in range(len(data)):
        dv = kappa*(theta - v)*dt + sigma_v*np.sqrt(max(v, 1e-8))*np.random.normal()
        v = max(v + dv, 1e-8)
        v_path.append(np.sqrt(v))
    
    data['Volatility'] = v_path
    
    return {'kappa': kappa, 'theta': theta, 'sigma_v': sigma_v, 'rho': rho, 'v0': v0}, data

# ============================================================================
# FUNCTION: Prepare sequences
# ============================================================================
def prepare_sequences(data, seq_len=60):
    features = data[['Price', 'Volatility']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(seq_len, len(scaled_features)):
        X.append(scaled_features[i-seq_len:i])
        y.append(scaled_features[i, 0])
    
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler, scaled_features

# ============================================================================
# FUNCTION: Build and train GRU
# ============================================================================
def build_gru_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=25, batch_size=32, verbose=0)
    return model, history

# ============================================================================
# FUNCTION: Evaluate model
# ============================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, scaled_features):
    def inverse_price(pred_scaled):
        full = np.zeros((len(pred_scaled), 2))
        full[:, 0] = pred_scaled.flatten()
        return scaler.inverse_transform(full)[:, 0]
    
    pred_train = inverse_price(model.predict(X_train, verbose=0))
    pred_test = inverse_price(model.predict(X_test, verbose=0))
    actual_train = inverse_price(y_train)
    actual_test = inverse_price(y_test)
    
    metrics = lambda actual, pred: {
        'mae': mean_absolute_error(actual, pred),
        'rmse': np.sqrt(mean_squared_error(actual, pred)),
        'mape': np.mean(np.abs((actual - pred) / actual)) * 100,
        'r2': r2_score(actual, pred)
    }
    
    train_metrics = metrics(actual_train, pred_train)
    test_metrics = metrics(actual_test, pred_test)
    
    return {
        'actual_prices_train': actual_train, 'pred_rescaled_train': pred_train,
        'actual_prices': actual_test, 'pred_rescaled': pred_test,
        **{f'{k}_train': v for k, v in train_metrics.items()},
        **{f'{k}_test': v for k, v in test_metrics.items()}
    }

# ============================================================================
# FUNCTION: Future forecast (consistent with train/test)
# ============================================================================
def future_forecast(model, scaled_features, scaler, seq_len, n_days, heston_params):
    kappa, theta, sigma_v, v0 = heston_params['kappa'], heston_params['theta'], \
                                heston_params['sigma_v'], heston_params['v0']
    dt = 1/252
    
    last_sequence = scaled_features[-seq_len:].copy()
    future_prices_scaled = []
    future_vols = []
    
    # Start from last actual volatility (inverse scale to get real vol → variance)
    last_scaled_vol = scaled_features[-1, 1]
    dummy = np.zeros((1, 2))
    dummy[0, 1] = last_scaled_vol
    last_vol = scaler.inverse_transform(dummy)[0, 1]
    v_current = last_vol ** 2
    
    for _ in range(n_days):
        input_seq = last_sequence.reshape((1, seq_len, 2))
        next_price_scaled = model.predict(input_seq, verbose=0)[0, 0]
        
        # Heston volatility simulation
        dv = kappa * (theta - v_current) * dt + sigma_v * np.sqrt(max(v_current, 1e-8)) * np.random.normal() * np.sqrt(dt)
        v_current = max(v_current + dv, 1e-8)
        next_vol = np.sqrt(v_current)
        
        # Scale volatility for next input
        dummy_vol = np.array([[0, next_vol]])
        next_vol_scaled = scaler.transform(dummy_vol)[0, 1]
        
        # Update sequence
        next_step = np.array([next_price_scaled, next_vol_scaled])
        last_sequence = np.append(last_sequence[1:], [next_step], axis=0)
        
        future_prices_scaled.append(next_price_scaled)
        future_vols.append(next_vol)
    
    # Inverse transform prices
    pred_array = np.array(future_prices_scaled).reshape(-1, 1)
    dummy_vol_col = np.zeros((len(pred_array), 1))
    full = np.hstack([pred_array, dummy_vol_col])
    future_prices = scaler.inverse_transform(full)[:, 0]
    
    return future_prices, future_vols

# ============================================================================
# STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(page_title="Advanced Stock Predictor: Heston + GRU", layout="wide")
    st.title("📈 Advanced Stock Price Prediction")
    st.markdown("### Heston Stochastic Volatility + GRU Neural Network")
    st.markdown("---")
    
    st.sidebar.header("⚙️ Configuration")
    ticker_input = st.sidebar.text_input("Stock Ticker", value="GBCO.CA").upper().strip()
    n_days = st.sidebar.number_input("Forecast Days Ahead", min_value=1, max_value=365, value=30)
    run = st.sidebar.button("🚀 Run Analysis", type="primary")
    
    if run:
        if not ticker_input:
            st.error("Please enter a ticker.")
            return
        
        progress = st.progress(0)
        status = st.empty()
        
        try:
            # 1. Load data
            status.text("Loading data...")
            progress.progress(10)
            data = load_data(ticker_input)
            if data.empty:
                st.error(f"No data for {ticker_input}")
                return
            
            # 2. Stats
            status.text("Statistical analysis...")
            progress.progress(20)
            stats_res = statistical_analysis(data.copy())
            data = stats_res['data']
            
            st.success(f"Data loaded: {len(data)} points from {data.index[0].date()} to {data.index[-1].date()}")
            
            # Display stats (same as before - omitted for brevity)
            st.markdown("## 📊 Return Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Return", f"{stats_res['mean_return']:.6f}")
            col2.metric("Volatility", f"{stats_res['volatility']:.4f}")
            col3.metric("Skewness", f"{stats_res['skewness']:.3f}")
            col4.metric("Kurtosis", f"{stats_res['pearson_kurtosis']:.2f}")
            st.write(f"**Tail Behavior:** {stats_res['tail_type']}")
            
            # 3. Heston
            status.text("Fitting Heston model...")
            progress.progress(40)
            heston_params, data = heston_model(data)
            
            # 4. Prepare & Train
            status.text("Training GRU...")
            progress.progress(60)
            seq_len = 60
            X_train, X_test, y_train, y_test, scaler, scaled_features = prepare_sequences(data, seq_len)
            model, _ = build_gru_model(X_train, y_train, X_test, y_test)
            
            # 5. Evaluate
            progress.progress(75)
            eval_res = evaluate_model(model, X_train, y_train, X_test, y_test, scaler, scaled_features)
            
            # 6. Forecast
            status.text("Generating future forecast...")
            progress.progress(90)
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='B')
            future_prices, future_vols = future_forecast(model, scaled_features, scaler, seq_len, n_days, heston_params)
            
            # === UNIFIED CHART ===
            st.markdown("## 📈 Complete Prediction Timeline")
            
            train_len = len(eval_res['actual_prices_train'])
            test_len = len(eval_res['actual_prices'])
            plot_start_idx = seq_len
            plot_dates = data.index[plot_start_idx : plot_start_idx + train_len + test_len]
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Historical train
            ax.plot(plot_dates[:train_len], eval_res['actual_prices_train'], 
                    label="Actual (Train)", color="blue", linewidth=2)
            ax.plot(plot_dates[:train_len], eval_res['pred_rescaled_train'], 
                    label="Predicted (Train)", color="lightblue", linestyle="--", linewidth=2)
            
            # Test period
            test_dates = plot_dates[train_len:]
            ax.plot(test_dates, eval_res['actual_prices'], 
                    label="Actual (Test)", color="green", linewidth=2)
            ax.plot(test_dates, eval_res['pred_rescaled'], 
                    label="Predicted (Test)", color="orange", linestyle="--", linewidth=2)
            
            # Future forecast
            ax.plot(future_dates, future_prices, 
                    label=f"Forecast ({n_days} days)", color="red", marker="o", markersize=3, linewidth=2.5)
            
            # Confidence band
            daily_vol_factor = np.array(future_vols) * np.sqrt(1/252)
            upper = future_prices * (1 + 1.96 * daily_vol_factor)
            lower = future_prices * (1 - 1.96 * daily_vol_factor)
            ax.fill_between(future_dates, lower, upper, color="red", alpha=0.15, label="95% Confidence")
            
            # Lines
            ax.axvline(plot_dates[train_len-1], color="gray", linestyle="--", label="Train/Test Split")
            ax.axvline(last_date, color="black", linewidth=2, label="Today")
            
            ax.set_title(f"{ticker_input} — Full Price Prediction (Historical + Forecast)", fontsize=16)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Metrics & Table
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${data['Price'].iloc[-1]:.2f}")
            col2.metric("Tomorrow Forecast", f"${future_prices[0]:.2f}")
            col3.metric(f"Day {n_days} Forecast", f"${future_prices[-1]:.2f}")
            change = (future_prices[-1] / data['Price'].iloc[-1] - 1) * 100
            col4.metric(f"Expected Change", f"{change:+.2f}%")
            
            future_df = pd.DataFrame({
                "Date": future_dates.date,
                "Predicted Price": np.round(future_prices, 2),
                "Volatility": np.round(future_vols, 4)
            })
            st.markdown("### Forecast Table")
            st.dataframe(future_df, use_container_width=True)
            
            progress.progress(100)
            status.text("✅ Complete!")
            st.success("Analysis and forecast completed successfully!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
