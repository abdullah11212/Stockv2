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
def load_data(ticker, end_date="2025-12-18"):
    data = yf.download(ticker, period="max", end=end_date)
    if 'Adj Close' in data.columns:
        data['Price'] = data['Adj Close']
    else:
        data['Price'] = data['Close']
    return data

# ============================================================================
# FUNCTION: Perform statistical analysis
# ============================================================================
def statistical_analysis(data):
    log_returns = np.log(data['Price'] / data['Price'].shift(1)).dropna()
    mean_return = float(log_returns.mean())
    volatility = float(log_returns.std(ddof=1))
    variance = float(log_returns.var(ddof=1))
    skewness = float(log_returns.skew())
    excess_kurtosis = float(log_returns.kurt())
    pearson_kurtosis = excess_kurtosis + 3
    
    ks_D, ks_p = stats.kstest(log_returns, 'norm', args=(mean_return, volatility))
    shapiro_stat, shapiro_p = stats.shapiro(log_returns)
    ad_result = stats.anderson(log_returns, dist='norm')
    
    threshold = 3 * volatility
    data['Log_Return'] = log_returns
    data.dropna(inplace=True)
    data['Jump'] = ((data['Log_Return'] - mean_return).abs() > threshold).astype(int)
    jump_count = int(data['Jump'].sum())
    jump_percent = 100 * jump_count / len(data)
    
    if pearson_kurtosis > 3.1:
        tail_type = "Leptokurtic (Heavy tails → jumps likely)"
    elif pearson_kurtosis < 2.9:
        tail_type = "Platykurtic (Thin tails)"
    else:
        tail_type = "Mesokurtic (≈ Normal)"
        
    if ad_result.statistic > ad_result.critical_values[2]:
        normality_conclusion = "❌ Reject Normality"
    else:
        normality_conclusion = "✔ Fail to Reject Normality"
    
    return {
        'mean_return': mean_return, 'volatility': volatility, 'variance': variance,
        'skewness': skewness, 'excess_kurtosis': excess_kurtosis,
        'pearson_kurtosis': pearson_kurtosis, 'tail_type': tail_type,
        'ks_D': ks_D, 'ks_p': ks_p, 'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p,
        'ad_statistic': ad_result.statistic, 'ad_critical_values': ad_result.critical_values,
        'normality_conclusion': normality_conclusion, 'threshold': threshold,
        'jump_count': jump_count, 'jump_percent': jump_percent, 'data': data
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
    
    heston_params = {'kappa': kappa, 'theta': theta, 'sigma_v': sigma_v, 'rho': rho, 'v0': v0}
    return heston_params, v_path, data

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
# FUNCTION: Build and train GRU model
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
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32, verbose=0)
    return model, history

# ============================================================================
# FUNCTION: Evaluate model
# ============================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, scaled_features):
    # Test set predictions
    pred_scaled = model.predict(X_test, verbose=0)
    test_full = np.zeros((len(pred_scaled), scaled_features.shape[1]))
    test_full[:, 0] = pred_scaled[:, 0]
    pred_rescaled = scaler.inverse_transform(test_full)[:, 0]
    
    actual_full = np.zeros((len(y_test), scaled_features.shape[1]))
    actual_full[:, 0] = y_test
    actual_prices = scaler.inverse_transform(actual_full)[:, 0]
    
    # Training set predictions
    pred_scaled_train = model.predict(X_train, verbose=0)
    train_full = np.zeros((len(pred_scaled_train), scaled_features.shape[1]))
    train_full[:, 0] = pred_scaled_train[:, 0]
    pred_rescaled_train = scaler.inverse_transform(train_full)[:, 0]
    
    actual_full_train = np.zeros((len(y_train), scaled_features.shape[1]))
    actual_full_train[:, 0] = y_train
    actual_prices_train = scaler.inverse_transform(actual_full_train)[:, 0]
    
    return {
        'actual_prices_train': actual_prices_train,
        'pred_rescaled_train': pred_rescaled_train,
        'actual_prices': actual_prices,
        'pred_rescaled': pred_rescaled,
        'mae_train': mean_absolute_error(actual_prices_train, pred_rescaled_train),
        'rmse_train': np.sqrt(mean_squared_error(actual_prices_train, pred_rescaled_train)),
        'r2_train': r2_score(actual_prices_train, pred_rescaled_train),
        'mape_train': np.mean(np.abs((actual_prices_train - pred_rescaled_train) / actual_prices_train)) * 100,
        'mae_test': mean_absolute_error(actual_prices, pred_rescaled),
        'rmse_test': np.sqrt(mean_squared_error(actual_prices, pred_rescaled)),
        'r2_test': r2_score(actual_prices, pred_rescaled),
        'mape_test': np.mean(np.abs((actual_prices - pred_rescaled) / actual_prices)) * 100
    }

# ============================================================================
# FUNCTION: Future Forecast
# ============================================================================
def future_forecast(model, data, scaler, scaled_features, seq_len, n_days, heston_params):
    kappa = heston_params['kappa']
    theta = heston_params['theta']
    sigma_v = heston_params['sigma_v']
    v0 = heston_params['v0']
    dt = 1/252
    
    last_sequence = scaled_features[-seq_len:].copy()
    future_predictions = []
    future_volatilities = []
    
    v_current = data['Volatility'].iloc[-1] ** 2
    
    for day in range(n_days):
        input_seq = last_sequence.reshape(1, seq_len, scaled_features.shape[1])
        next_pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        
        dv = kappa * (theta - v_current) * dt + sigma_v * np.sqrt(max(v_current, 1e-8)) * np.random.normal() * np.sqrt(dt)
        v_current = max(v_current + dv, 1e-8)
        next_vol = np.sqrt(v_current)
        future_volatilities.append(next_vol)
        
        temp_features = np.array([[0, next_vol]])
        scaled_temp = scaler.transform(temp_features)
        next_vol_scaled = scaled_temp[0, 1]
        
        next_step = np.array([[next_pred_scaled, next_vol_scaled]])
        last_sequence = np.vstack([last_sequence[1:], next_step])
        future_predictions.append(next_step[0])
    
    future_predictions = np.array(future_predictions)
    future_prices = scaler.inverse_transform(future_predictions)[:, 0]
    
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='D')
    
    return future_dates, future_prices, future_volatilities

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================
def main():
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    
    st.title("📈 Unified Stock Analysis & Forecast")
    st.markdown("---")
    
    st.sidebar.header("⚙️ Configuration")
    ticker_input = st.sidebar.text_input("Enter Stock Ticker:", value="GBCO.CA")
    n_days = st.sidebar.number_input("Prediction Days:", min_value=1, max_value=365, value=30)
    run_button = st.sidebar.button("🚀 Run Analysis", type="primary")
    
    if run_button:
        ticker = ticker_input.strip().upper()
        if not ticker:
            st.error("Please enter a valid stock ticker.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. Load Data
            status_text.text("📥 Loading data...")
            progress_bar.progress(10)
            data = load_data(ticker)
            if data.empty:
                st.error("No data found.")
                return

            # 2. Stats Analysis
            status_text.text("📊 Analyzing statistics...")
            progress_bar.progress(25)
            stats_results = statistical_analysis(data.copy())
            data = stats_results['data']

            # 3. Heston Model
            status_text.text("🔄 Applying Heston model...")
            progress_bar.progress(40)
            heston_params, v_path, data = heston_model(data)

            # 4. Prepare & Train
            status_text.text("🧠 Training Neural Network...")
            progress_bar.progress(55)
            seq_len = 60
            X_train, X_test, y_train, y_test, scaler, scaled_features = prepare_sequences(data, seq_len)
            model, history = build_gru_model(X_train, y_train, X_test, y_test)

            # 5. Evaluate (Train/Test Predictions)
            status_text.text("📈 Evaluating model...")
            progress_bar.progress(70)
            eval_results = evaluate_model(model, X_train, y_train, X_test, y_test, scaler, scaled_features)

            # 6. Future Forecast
            status_text.text(f"🔮 Forecasting {n_days} days ahead...")
            progress_bar.progress(85)
            future_dates, future_prices, future_vols = future_forecast(
                model, data, scaler, scaled_features, seq_len, n_days, heston_params
            )

            # ========================================================================
            # UNIFIED PLOTTING SECTION
            # ========================================================================
            st.markdown("## 🔮 Comprehensive Forecast Chart")
            st.info("This chart combines historical training/testing validation with future projections.")

            # Create Indices for Plotting
            train_len = len(eval_results['pred_rescaled_train'])
            test_len = len(eval_results['pred_rescaled'])
            
            # The indices align with the sequence length offset
            train_index = data.index[seq_len : seq_len + train_len]
            test_index = data.index[seq_len + train_len : seq_len + train_len + test_len]
            
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # 1. Plot Full Historical Data (Background)
            # We strip the first seq_len because the model can't predict those without history
            plot_data = data.iloc[seq_len:]
            ax.plot(plot_data.index, plot_data['Price'], label='Actual Price', color='black', alpha=0.3, linewidth=2)
            
            # 2. Plot Train Predictions
            ax.plot(train_index, eval_results['pred_rescaled_train'], label='Train Prediction', color='green', alpha=0.8, linewidth=1.5)
            
            # 3. Plot Test Predictions
            ax.plot(test_index, eval_results['pred_rescaled'], label='Test Prediction', color='blue', alpha=0.8, linewidth=1.5)
            
            # 4. Plot Future Forecast
            ax.plot(future_dates, future_prices, label=f'Future Forecast ({n_days} Days)', color='red', linewidth=2.5, linestyle='--')
            
            # 5. Confidence Intervals (Using Forecasted Volatility)
            upper_bound = future_prices + 1.96 * np.array(future_vols) * future_prices
            lower_bound = future_prices - 1.96 * np.array(future_vols) * future_prices
            ax.fill_between(future_dates, lower_bound, upper_bound, color='red', alpha=0.1, label='95% Confidence Interval')

            # Formatting
            ax.axvline(x=data.index[-1], color='gray', linestyle=':', linewidth=2, label='Today')
            ax.set_title(f"{ticker} - Unified Forecast (Train, Test & Future)", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Price", fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

            # ========================================================================
            # METRICS DISPLAY
            # ========================================================================
            st.markdown("### 📊 Model & Forecast Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Test RMSE (Error)", f"{eval_results['rmse_test']:.2f}")
            with col2:
                st.metric("Test Accuracy (R²)", f"{eval_results['r2_test']:.2f}")
            with col3:
                curr_price = data['Price'].iloc[-1]
                st.metric("Current Price", f"{curr_price:.2f}")
            with col4:
                final_pred = future_prices[-1]
                pct_change = ((final_pred - curr_price) / curr_price) * 100
                st.metric(f"Price in {n_days} Days", f"{final_pred:.2f}", f"{pct_change:+.2f}%")

            # Data Table for Future
            with st.expander("📄 View Future Forecast Data"):
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecasted Price': future_prices,
                    'Lower Bound (95%)': lower_bound,
                    'Upper Bound (95%)': upper_bound
                })
                st.dataframe(future_df)

            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
