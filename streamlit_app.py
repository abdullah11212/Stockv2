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
    """
    Download historical stock data using yfinance.
    Uses Adjusted Close if available, otherwise uses Close price.
    
    Args:
        ticker (str): Stock ticker symbol
        end_date (str): End date for data download
    
    Returns:
        pd.DataFrame: Historical price data
    """
    data = yf.download(ticker, period="max", end=end_date)
    
    # Use Adjusted Close when available
    if 'Adj Close' in data.columns:
        data['Price'] = data['Adj Close']
    else:
        data['Price'] = data['Close']
    
    return data

# ============================================================================
# FUNCTION: Perform statistical analysis on log returns
# ============================================================================
def statistical_analysis(data):
    """
    Compute statistical measures and normality tests on log returns.
    Includes jump detection using 3-sigma rule.
    
    Args:
        data (pd.DataFrame): Price data
    
    Returns:
        dict: Statistical results and jump information
    """
    # Calculate log returns
    log_returns = np.log(data['Price'] / data['Price'].shift(1)).dropna()
    
    # Basic statistics
    mean_return = float(log_returns.mean())
    volatility = float(log_returns.std(ddof=1))
    variance = float(log_returns.var(ddof=1))
    
    # Skewness
    skewness = float(log_returns.skew())
    
    # Kurtosis (Correct Form)
    excess_kurtosis = float(log_returns.kurt())  # normal = 0
    pearson_kurtosis = excess_kurtosis + 3  # normal = 3
    
    # Normality Tests
    ks_D, ks_p = stats.kstest(log_returns, 'norm', args=(mean_return, volatility))
    shapiro_stat, shapiro_p = stats.shapiro(log_returns)
    ad_result = stats.anderson(log_returns, dist='norm')
    
    # Jump detection using 3-sigma rule
    threshold = 3 * volatility
    data['Log_Return'] = log_returns
    data.dropna(inplace=True)
    data['Jump'] = ((data['Log_Return'] - mean_return).abs() > threshold).astype(int)
    jump_count = int(data['Jump'].sum())
    jump_percent = 100 * jump_count / len(data)
    
    # Kurtosis interpretation
    if pearson_kurtosis > 3.1:
        tail_type = "Leptokurtic (Heavy tails → jumps likely)"
    elif pearson_kurtosis < 2.9:
        tail_type = "Platykurtic (Thin tails)"
    else:
        tail_type = "Mesokurtic (≈ Normal)"
    
    # Normality conclusion
    if ad_result.statistic > ad_result.critical_values[2]:
        normality_conclusion = "❌ Reject Normality"
    else:
        normality_conclusion = "✔ Fail to Reject Normality"
    
    return {
        'mean_return': mean_return,
        'volatility': volatility,
        'variance': variance,
        'skewness': skewness,
        'excess_kurtosis': excess_kurtosis,
        'pearson_kurtosis': pearson_kurtosis,
        'tail_type': tail_type,
        'ks_D': ks_D,
        'ks_p': ks_p,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'ad_statistic': ad_result.statistic,
        'ad_critical_values': ad_result.critical_values,
        'normality_conclusion': normality_conclusion,
        'threshold': threshold,
        'jump_count': jump_count,
        'jump_percent': jump_percent,
        'data': data
    }

# ============================================================================
# FUNCTION: Heston stochastic volatility model
# ============================================================================
def heston_model(data):
    """
    Estimate Heston model parameters using Maximum Likelihood Estimation.
    Simulate the conditional volatility path.
    
    Args:
        data (pd.DataFrame): Price data with returns
    
    Returns:
        tuple: (Heston parameters, volatility path, updated data)
    """
    # Calculate returns
    data['Return'] = 100 * data['Price'].pct_change()
    data.dropna(inplace=True)
    
    # Heston log-likelihood function
    def heston_loglike(params, returns):
        kappa, theta, sigma_v, rho, v0 = params
        dt = 1/252
        vt = v0
        loglike = 0
        for r in returns:
            # Variance update (Euler discretization)
            dv = kappa*(theta - vt)*dt + sigma_v*np.sqrt(max(vt, 1e-8))*np.random.normal()
            vt = max(vt + dv, 1e-8)
            # Return likelihood
            var_ret = vt * dt
            loglike += -0.5*(np.log(2*np.pi*var_ret) + (r**2)/var_ret)
        return -loglike
    
    # Initial guess for parameters
    params0 = [1.0, 0.02, 0.2, -0.3, 0.02]
    
    # Optimize parameters
    res = minimize(heston_loglike, params0, args=(data['Return'].values,), method='Nelder-Mead')
    kappa, theta, sigma_v, rho, v0 = res.x
    
    # Simulate Heston variance path
    dt = 1/252
    v_path = []
    v = v0
    for _ in range(len(data)):
        dv = kappa*(theta - v)*dt + sigma_v*np.sqrt(max(v, 1e-8))*np.random.normal()
        v = max(v + dv, 1e-8)
        v_path.append(np.sqrt(v))
    
    # Add Heston volatility to data
    data['Volatility'] = v_path
    
    heston_params = {
        'kappa': kappa,
        'theta': theta,
        'sigma_v': sigma_v,
        'rho': rho,
        'v0': v0
    }
    
    return heston_params, v_path, data

# ============================================================================
# FUNCTION: Prepare sequences for GRU training
# ============================================================================
def prepare_sequences(data, seq_len=60):
    """
    Create sequences of features and targets for time series prediction.
    
    Args:
        data (pd.DataFrame): Data with Price and Volatility columns
        seq_len (int): Sequence length (lookback period)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, scaled_features)
    """
    # Extract features
    features = data[['Price', 'Volatility']].values
    
    # Scale features to [0, 1]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(seq_len, len(scaled_features)):
        X.append(scaled_features[i-seq_len:i])
        y.append(scaled_features[i, 0])  # Predict Price
    
    X, y = np.array(X), np.array(y)
    
    # Train-test split (no shuffling for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, scaler, scaled_features

# ============================================================================
# FUNCTION: Build and train GRU model
# ============================================================================
def build_gru_model(X_train, y_train, X_test, y_test):
    """
    Build a GRU neural network for stock price prediction.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
    
    Returns:
        tuple: (trained model, training history)
    """
    # Build GRU model
    model = Sequential([
        # First GRU layer with return_sequences=True to stack another GRU layer
        GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        # Second GRU layer
        GRU(32, return_sequences=False),
        Dropout(0.2),
        # Output layer
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=25,
        batch_size=32,
        verbose=0  # Silent training for Streamlit
    )
    
    return model, history

# ============================================================================
# FUNCTION: Evaluate model performance
# ============================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, scaled_features):
    """
    Compute performance metrics for training and test sets.
    
    Args:
        model: Trained GRU model
        X_train, y_train: Training data
        X_test, y_test: Testing data
        scaler: MinMaxScaler object
        scaled_features: Original scaled features
    
    Returns:
        dict: Performance metrics and predictions
    """
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
    
    # Compute metrics for training set
    mae_train = mean_absolute_error(actual_prices_train, pred_rescaled_train)
    rmse_train = np.sqrt(mean_squared_error(actual_prices_train, pred_rescaled_train))
    r2_train = r2_score(actual_prices_train, pred_rescaled_train)
    mape_train = np.mean(np.abs((actual_prices_train - pred_rescaled_train) / actual_prices_train)) * 100
    
    # Compute metrics for test set
    mae_test = mean_absolute_error(actual_prices, pred_rescaled)
    rmse_test = np.sqrt(mean_squared_error(actual_prices, pred_rescaled))
    r2_test = r2_score(actual_prices, pred_rescaled)
    mape_test = np.mean(np.abs((actual_prices - pred_rescaled) / actual_prices)) * 100
    
    return {
        'actual_prices_train': actual_prices_train,
        'pred_rescaled_train': pred_rescaled_train,
        'actual_prices': actual_prices,
        'pred_rescaled': pred_rescaled,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_train': r2_train,
        'mape_train': mape_train,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'mape_test': mape_test
    }

# ============================================================================
# FUNCTION: Forecast future prices
# ============================================================================
def future_forecast(model, data, scaler, scaled_features, seq_len, n_days, heston_params):
    """
    Perform auto-regressive forecasting for future days with proper Heston volatility simulation.
    
    Args:
        model: Trained GRU model
        data: Historical data
        scaler: MinMaxScaler object
        scaled_features: Scaled historical features
        seq_len: Sequence length
        n_days: Number of days to forecast
        heston_params: Dictionary of Heston model parameters
    
    Returns:
        tuple: (future dates, future prices, future volatilities)
    """
    # Extract Heston parameters
    kappa = heston_params['kappa']
    theta = heston_params['theta']
    sigma_v = heston_params['sigma_v']
    v0 = heston_params['v0']
    dt = 1/252
    
    # Start with the last sequence
    last_sequence = scaled_features[-seq_len:].copy()
    future_predictions = []
    future_volatilities = []
    
    # Initialize volatility state (start from last known variance)
    v_current = data['Volatility'].iloc[-1] ** 2  # Convert volatility to variance
    
    # Auto-regressive forecasting with Heston volatility simulation
    for day in range(n_days):
        # Reshape for prediction
        input_seq = last_sequence.reshape(1, seq_len, scaled_features.shape[1])
        
        # Predict next price (scaled)
        next_pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        
        # Simulate next Heston volatility (Euler-Maruyama discretization)
        dv = kappa * (theta - v_current) * dt + sigma_v * np.sqrt(max(v_current, 1e-8)) * np.random.normal() * np.sqrt(dt)
        v_current = max(v_current + dv, 1e-8)  # Ensure positive variance
        next_vol = np.sqrt(v_current)  # Convert variance back to volatility
        
        # Store the actual volatility before scaling
        future_volatilities.append(next_vol)
        
        # Scale the volatility for model input
        # We need to scale it the same way as during training
        temp_features = np.array([[0, next_vol]])  # Dummy price, real volatility
        scaled_temp = scaler.transform(temp_features)
        next_vol_scaled = scaled_temp[0, 1]
        
        # Create next step with predicted price and simulated volatility
        next_step = np.array([[next_pred_scaled, next_vol_scaled]])
        
        # Update sequence (slide window)
        last_sequence = np.vstack([last_sequence[1:], next_step])
        
        # Store prediction
        future_predictions.append(next_step[0])
    
    # Convert to array and inverse transform to get actual prices
    future_predictions = np.array(future_predictions)
    future_prices = scaler.inverse_transform(future_predictions)[:, 0]
    
    # Generate future dates (business days)
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='D')
    
    return future_dates, future_prices, future_volatilities

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================
def main():
    st.set_page_config(page_title="Stock Price Predictor with Heston-GRU", layout="wide")
    
    st.title("📈 Stock Price Prediction using Heston Model & GRU")
    st.markdown("---")
    
    # Sidebar for user inputs
    st.sidebar.header("⚙️ Configuration")
    
    # Ticker selection
    default_tickers = ["GBCO.CA", "AAPL", "TSLA", "MSFT", "GOOGL"]
    ticker_input = st.sidebar.text_input("Enter Stock Ticker:", value="GBCO.CA")
    
    # Future prediction days
    n_days = st.sidebar.number_input(
        "Number of Future Prediction Days:",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )
    
    # Run button
    run_button = st.sidebar.button("🚀 Run Prediction", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Note:** This app uses Heston stochastic volatility model combined with GRU neural networks for stock price forecasting.")
    
    # Main execution
    if run_button:
        ticker = ticker_input.strip().upper()
        
        if not ticker:
            st.error("Please enter a valid stock ticker.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load Data
            status_text.text("📥 Loading historical data...")
            progress_bar.progress(10)
            data = load_data(ticker)
            
            if data.empty:
                st.error(f"No data found for ticker: {ticker}")
                return
            
            st.success(f"✅ Data loaded successfully! Start date: {data.index.min().date()}")
            st.write(f"**Total data points:** {len(data)}")
            
            # Step 2: Statistical Analysis
            status_text.text("📊 Performing statistical analysis...")
            progress_bar.progress(20)
            stats_results = statistical_analysis(data.copy())
            data = stats_results['data']
            
            # Display statistical results
            st.markdown("## 📊 Statistical Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Return (μ)", f"{stats_results['mean_return']:.6f}")
            with col2:
                st.metric("Volatility (σ)", f"{stats_results['volatility']:.6f}")
            with col3:
                st.metric("Variance", f"{stats_results['variance']:.6f}")
            with col4:
                st.metric("Skewness", f"{stats_results['skewness']:.4f}")
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric("Excess Kurtosis", f"{stats_results['excess_kurtosis']:.4f}")
            with col6:
                st.metric("Pearson Kurtosis", f"{stats_results['pearson_kurtosis']:.4f}")
            
            st.write(f"**Kurtosis Interpretation:** {stats_results['tail_type']}")
            
            # Normality tests in expander
            with st.expander("🔍 Normality Tests"):
                st.write(f"**Kolmogorov-Smirnov Test:** D={stats_results['ks_D']:.5f}, p={stats_results['ks_p']:.2e}")
                st.write(f"**Shapiro-Wilk Test:** stat={stats_results['shapiro_stat']:.5f}, p={stats_results['shapiro_p']:.2e}")
                st.write(f"**Anderson-Darling Test:** stat={stats_results['ad_statistic']:.5f}")
                st.write(f"**AD Critical Values:** {stats_results['ad_critical_values']}")
                st.write(f"**Normality Conclusion:** {stats_results['normality_conclusion']}")
                
                # Add interpretation
                st.info("""
                **Interpretation:** All tests strongly reject normality, which is expected for stock returns. 
                Financial returns typically exhibit:
                - Heavy tails (leptokurtic distribution)
                - Skewness (asymmetric returns)
                - Volatility clustering
                
                This justifies using the Heston stochastic volatility model instead of assuming constant volatility.
                """)
            
            # Jump detection
            with st.expander("🎯 Jump Detection (3σ Rule)"):
                st.write(f"**Jump Threshold:** |r - μ| > {stats_results['threshold']:.6f}")
                st.write(f"**Jump Count:** {stats_results['jump_count']}")
                st.write(f"**Jump Percentage:** {stats_results['jump_percent']:.2f}%")
            
            # Plot price with jumps
            st.markdown("### Price Chart with Jump Detection")
            fig1, ax1 = plt.subplots(figsize=(15, 7))
            ax1.plot(data.index, data['Price'], label=f'{ticker} Price', color='blue')
            jump_dates = data.index[data['Jump'] == 1]
            jump_prices = data['Price'][data['Jump'] == 1]
            ax1.scatter(jump_dates, jump_prices, color='green', label='Jumps', marker='^', s=70)
            ax1.set_title(f'{ticker} Price with Jump Detection (3σ rule)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)
            
            # Step 3: Heston Model
            status_text.text("🔄 Estimating Heston model parameters...")
            progress_bar.progress(40)
            heston_params, v_path, data = heston_model(data)
            
            st.markdown("## 🔄 Heston Stochastic Volatility Model")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("κ (kappa)", f"{heston_params['kappa']:.4f}")
            with col2:
                st.metric("θ (theta)", f"{heston_params['theta']:.4f}")
            with col3:
                st.metric("σ_v (sigma_v)", f"{heston_params['sigma_v']:.4f}")
            with col4:
                st.metric("ρ (rho)", f"{heston_params['rho']:.4f}")
            with col5:
                st.metric("v₀ (v0)", f"{heston_params['v0']:.4f}")
            
            # Plot Heston volatility
            st.markdown("### Heston Conditional Volatility")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(data.index, data['Volatility'], color='purple')
            ax2.set_title("Heston Conditional Volatility")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Volatility")
            st.pyplot(fig2)
            
            # Step 4: Prepare sequences
            status_text.text("🔨 Preparing sequences for GRU training...")
            progress_bar.progress(50)
            seq_len = 60
            X_train, X_test, y_train, y_test, scaler, scaled_features = prepare_sequences(data, seq_len)
            
            # Step 5: Build and train GRU model
            status_text.text("🧠 Training GRU model...")
            progress_bar.progress(60)
            model, history = build_gru_model(X_train, y_train, X_test, y_test)
            
            # Step 6: Evaluate model
            status_text.text("📈 Evaluating model performance...")
            progress_bar.progress(75)
            eval_results = evaluate_model(model, X_train, y_train, X_test, y_test, scaler, scaled_features)
            
            st.markdown("## 📈 Model Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎓 Training Set")
                st.metric("MAE", f"{eval_results['mae_train']:.4f}")
                st.metric("RMSE", f"{eval_results['rmse_train']:.4f}")
                st.metric("MAPE", f"{eval_results['mape_train']:.2f}%")
                st.metric("R²", f"{eval_results['r2_train']:.4f}")
            
            with col2:
                st.markdown("### 🧪 Test Set")
                st.metric("MAE", f"{eval_results['mae_test']:.4f}")
                st.metric("RMSE", f"{eval_results['rmse_test']:.4f}")
                st.metric("MAPE", f"{eval_results['mape_test']:.2f}%")
                st.metric("R²", f"{eval_results['r2_test']:.4f}")
            
            # Plot predictions
            st.markdown("### Historical Price Prediction")
            train_index = data.index[seq_len : seq_len + len(eval_results['actual_prices_train'])]
            test_index = data.index[seq_len + len(eval_results['actual_prices_train']) : 
                                     seq_len + len(eval_results['actual_prices_train']) + len(eval_results['actual_prices'])]
            
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(train_index, eval_results['actual_prices_train'], label="Actual Price (Train)", color='blue')
            ax3.plot(train_index, eval_results['pred_rescaled_train'], label="Predicted Price (Train)", color='red', linestyle='--')
            ax3.plot(test_index, eval_results['actual_prices'], label="Actual Price (Test)", color='green')
            ax3.plot(test_index, eval_results['pred_rescaled'], label="Predicted Price (Test)", color='orange', linestyle='--')
            split_date = train_index[-1]
            ax3.axvline(x=split_date, color='gray', linestyle='--', label="Train/Test Split")
            ax3.set_title(f"{ticker} Price Prediction with GRU")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Price")
            ax3.legend()
            ax3.grid(True)
            st.pyplot(fig3)
            
            # Step 7: Future forecasting
            status_text.text(f"🔮 Forecasting {n_days} days into the future...")
            progress_bar.progress(90)
            future_dates, future_prices, future_vols = future_forecast(
                model, data, scaler, scaled_features, seq_len, n_days, heston_params
            )
            
            st.markdown(f"## 🔮 Future Price Forecast ({n_days} Days)")
            
            # Add warning for long-term predictions
            if n_days > 30:
                st.warning("⚠️ **Warning:** Predictions beyond 30 days may be unreliable due to error amplification in autoregressive forecasting. Consider using shorter horizons (7-21 days) for more accurate results.")
            
            # Calculate prediction statistics
            avg_future_price = np.mean(future_prices)
            std_future_price = np.std(future_prices)
            min_future_price = np.min(future_prices)
            max_future_price = np.max(future_prices)
            
            # Display future predictions
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_prices,
                'Forecasted Volatility': future_vols
            })
            st.dataframe(future_df, use_container_width=True)
            
            # Summary statistics for forecast
            st.markdown("### Forecast Summary Statistics")
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Average Price", f"${avg_future_price:.2f}")
            with col_b:
                st.metric("Std Deviation", f"${std_future_price:.2f}")
            with col_c:
                st.metric("Min Price", f"${min_future_price:.2f}")
            with col_d:
                st.metric("Max Price", f"${max_future_price:.2f}")
            
            # Plot future predictions
            st.markdown("### Future Price Forecast Chart")
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            
            # Plot historical prices (last 180 days for clarity)
            recent_data = data.tail(180)
            ax4.plot(recent_data.index, recent_data['Price'], label='Historical Price', color='blue', linewidth=2)
            
            # Plot future predictions
            ax4.plot(future_dates, future_prices, label=f'Future Forecast ({n_days} days)', 
                    color='red', linestyle='--', marker='o', markersize=4, linewidth=2)
            
            # Add confidence band based on forecasted volatility
            # Simple approximation: ±1.96 * volatility for 95% confidence
            upper_bound = future_prices + 1.96 * np.array(future_vols) * future_prices
            lower_bound = future_prices - 1.96 * np.array(future_vols) * future_prices
            ax4.fill_between(future_dates, lower_bound, upper_bound, 
                            color='red', alpha=0.2, label='95% Confidence Interval')
            
            # Add vertical line at present
            ax4.axvline(x=data.index[-1], color='gray', linestyle='--', linewidth=1.5, label='Present Day')
            
            ax4.set_title(f'{ticker} Price Forecast with Confidence Interval', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Date', fontsize=12)
            ax4.set_ylabel('Price', fontsize=12)
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${data['Price'].iloc[-1]:.2f}")
            with col2:
                st.metric("Predicted Price (Day 1)", f"${future_prices[0]:.2f}")
            with col3:
                st.metric(f"Predicted Price (Day {n_days})", f"${future_prices[-1]:.2f}")
            
            change = ((future_prices[-1] - data['Price'].iloc[-1]) / data['Price'].iloc[-1]) * 100
            st.metric(f"Expected Change over {n_days} days", f"{change:+.2f}%")
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            st.success("🎉 Prediction completed successfully!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
