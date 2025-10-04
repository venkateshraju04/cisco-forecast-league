import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Suppress warnings and logging
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('statsmodels').setLevel(logging.ERROR)

# Global economic indicators - example data
# In a real scenario, this would be loaded from external sources
MACRO_ECONOMIC_INDICATORS = {
    'GDP_Growth': [2.3, 2.1, 2.0, 1.9, 1.8, 1.7, 1.9, 2.2, 2.5],
    'Interest_Rate': [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25],
    'Inflation': [1.8, 2.0, 2.2, 2.3, 2.1, 2.0, 2.2, 2.4, 2.5],
    'Tech_Spending_Index': [105, 108, 112, 118, 122, 125, 130, 135, 140],
    'Genai_Adoption': [10, 15, 25, 40, 60, 85, 110, 140, 180]  # Index of GenAI adoption
}

def perform_eda(historical_df):
    """
    Perform Exploratory Data Analysis to detect seasonality and trends
    """
    print("\n=== PERFORMING EXPLORATORY DATA ANALYSIS ===")
    eda_results = {}
    
    for product in historical_df.index:
        product_data = historical_df.loc[product].dropna()
        
        if len(product_data) < 8:  # Need sufficient data for seasonal analysis
            eda_results[product] = {
                'has_seasonality': False, 
                'trend': 'insufficient_data',
                'outliers': []
            }
            continue
            
        # Convert to series with dummy frequency
        ts = pd.Series(product_data.values, 
                       index=pd.date_range(start='2020-01-01', periods=len(product_data), freq='QE'))
        
        # Check for seasonality using autocorrelation
        acf_values = sm.tsa.acf(ts, nlags=min(8, len(ts)-1))
        has_seasonality = any(acf_values[1:] > 0.5)  # Significant autocorrelation
        
        # Identify trend using linear regression
        X = np.arange(len(ts)).reshape(-1, 1)
        y = ts.values
        slope, _, _, _, _ = stats.linregress(X.flatten(), y)
        
        if slope > 0.05:
            trend = 'increasing'
        elif slope < -0.05:
            trend = 'decreasing'
        else:
            trend = 'stable'
            
        # Detect outliers
        z_scores = stats.zscore(ts.values)
        outliers = np.where(np.abs(z_scores) > 2.5)[0]  # Z-score threshold of 2.5
        
        eda_results[product] = {
            'has_seasonality': has_seasonality,
            'trend': trend,
            'outliers': outliers.tolist()
        }
        
        print(f"Product: {product}")
        print(f"  - Seasonality: {'Detected' if has_seasonality else 'Not detected'}")
        print(f"  - Trend: {trend.capitalize()}")
        print(f"  - Outliers: {len(outliers)} detected")
        
    return eda_results

def load_data_from_excel(file_path='a.xlsx'):
    """Extract data from Excel file instead of manual input"""
    print(f"\n=== LOADING DATA FROM {file_path} ===")
    
    try:
        # Read the first sheet directly, explicitly setting header=None to prevent first row being used as header
        xls = pd.ExcelFile(file_path)
        sheet_name = xls.sheet_names[0]
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        print(f"Using sheet: {sheet_name}")
        print(f"Successfully loaded data with {len(df)} rows.")
        
        # Extract product names from second column (index 1) and convert to strings
        raw_product_names = [str(name).strip() for name in df.iloc[:, 1].tolist()]
        
        # Create unique product names (for cases with duplicates)
        product_names = []
        product_counts = {}
        
        for name in raw_product_names:
            if name in product_counts:
                product_counts[name] += 1
                product_names.append(f"{name} ({product_counts[name]})")
            else:
                product_counts[name] = 1
                product_names.append(name)
        
        # Identify where historical data starts and ends
        # Assuming third column starts historical data
        historical_start_col = 2
        
        # Find the first empty column after historical data
        historical_end_col = historical_start_col
        while historical_end_col < len(df.columns) and not df.iloc[:, historical_end_col].isna().all():
            historical_end_col += 1
        
        # Extract historical data
        historical_data = df.iloc[:, historical_start_col:historical_end_col]
        
        # Create column headers for historical data
        quarter_headers = [f"Q{i+1}" for i in range(historical_data.shape[1])]
        
        # Create final historical DataFrame with product names as index and quarter headers
        historical_df = pd.DataFrame(historical_data.values, 
                                     index=product_names, 
                                     columns=quarter_headers)
        
        # Find where forecast columns start (after the empty column)
        forecast_start_col = historical_end_col + 1
        while forecast_start_col < len(df.columns) and df.iloc[:, forecast_start_col].isna().all():
            forecast_start_col += 1
        
        # Extract forecast columns
        forecast_columns = []
        forecast_data = pd.DataFrame(index=product_names)
        
        if forecast_start_col < len(df.columns):
            # Find how many forecast columns exist
            forecast_end_col = forecast_start_col
            while forecast_end_col < len(df.columns) and not df.iloc[:, forecast_end_col].isna().all():
                col_name = f"Forecast_{forecast_end_col-forecast_start_col+1}"
                forecast_columns.append(col_name)
                forecast_data[col_name] = df.iloc[:, forecast_end_col].values
                forecast_end_col += 1
        
        # Print summary
        print(f"\nFound {len(product_names)} products")
        print(f"Found {historical_data.shape[1]} quarters of historical data")
        
        if forecast_columns:
            print(f"Found {len(forecast_columns)} forecast columns: {', '.join(forecast_columns)}")
        else:
            print("No forecast columns found")
        
        return historical_df, forecast_data, product_names
        
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None, None, None

def apply_gold_outlier_detection(historical_df):
    """
    GOLD (Global Outlier Detection): Detect and handle outliers in time series data
    """
    print("\n=== APPLYING GOLD OUTLIER DETECTION ===")
    cleaned_df = historical_df.copy()
    
    for product in historical_df.index:
        product_data = historical_df.loc[product].dropna()
        
        if len(product_data) < 4:  # Need minimum data for outlier detection
            continue
            
        # Convert to numpy array
        data = product_data.values.reshape(-1, 1)
        
        # Isolation Forest for outlier detection
        clf = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = clf.fit_predict(data)
        
        # Replace outliers with interpolated values
        outlier_indices = np.where(outlier_labels == -1)[0]
        
        if len(outlier_indices) > 0:
            print(f"Product '{product}': {len(outlier_indices)} outliers detected and processed")
            
            for idx in outlier_indices:
                if idx == 0:  # First value is outlier
                    cleaned_df.loc[product, cleaned_df.columns[idx]] = product_data.iloc[1]
                elif idx == len(product_data) - 1:  # Last value is outlier
                    cleaned_df.loc[product, cleaned_df.columns[idx]] = product_data.iloc[-2]
                else:  # Middle values - use linear interpolation
                    left_val = product_data.iloc[idx-1]
                    right_val = product_data.iloc[idx+1]
                    cleaned_df.loc[product, cleaned_df.columns[idx]] = (left_val + right_val) / 2
    
    return cleaned_df

def create_feature_pools(historical_df, eda_results):
    """
    Create feature pools for inclusion in forecasting models
    """
    feature_pools = {}
    
    for product in historical_df.index:
        product_data = historical_df.loc[product].dropna()
        
        if len(product_data) < 4:  # Need minimum data
            continue
            
        # Basic features
        features = {
            'lags': {},
            'rolling_stats': {},
            'macro_indicators': {},
            'seasonality': {}
        }
        
        # 1. Lag features
        for lag in range(1, min(4, len(product_data))):
            lag_values = product_data.shift(lag).dropna()
            features['lags'][f'lag_{lag}'] = lag_values
        
        # 2. Rolling statistics
        for window in [2, 3, 4]:
            if window < len(product_data):
                features['rolling_stats'][f'rolling_mean_{window}'] = product_data.rolling(window).mean().dropna()
                features['rolling_stats'][f'rolling_std_{window}'] = product_data.rolling(window).std().dropna()
                
        # 3. Macroeconomic indicators
        for indicator, values in MACRO_ECONOMIC_INDICATORS.items():
            if len(values) >= len(product_data):
                # Use the most recent indicators that match the length of our data
                features['macro_indicators'][indicator] = values[-len(product_data):]
                
        # 4. Seasonality features
        if eda_results.get(product, {}).get('has_seasonality', False):
            for season in range(4):  # Assuming quarterly data with annual seasonality
                # Create seasonal dummy variables
                seasonal_dummy = [(i % 4 == season) * 1.0 for i in range(len(product_data))]
                features['seasonality'][f'quarter_{season+1}'] = seasonal_dummy
                
        # Special case for server-related products - add GenAI boom weight
        if 'server' in product.lower() or 'network' in product.lower() or 'router' in product.lower():
            if len(MACRO_ECONOMIC_INDICATORS['Genai_Adoption']) >= len(product_data):
                features['macro_indicators']['Genai_Weight'] = [x * 1.5 for x in 
                                            MACRO_ECONOMIC_INDICATORS['Genai_Adoption'][-len(product_data):]]
        
        feature_pools[product] = features
    
    print(f"\n=== CREATED FEATURE POOLS FOR {len(feature_pools)} PRODUCTS ===")
    return feature_pools

def apply_ewma(historical_df):
    """
    Apply Exponentially Weighted Moving Average (EWMA) to each product
    """
    ewma_df = historical_df.copy()
    ewma_forecasts = {}
    future_periods = 4  # Default forecast horizon
    
    for product in historical_df.index:
        product_data = historical_df.loc[product].dropna()
        
        if len(product_data) < 4:  # Need minimum data
            continue
            
        # Apply different spans for EWMA - must be >= 1
        spans = [2, 4, 8]  # Different span parameters (higher = less weight to recent observations)
        ewma_results = {}
        
        for span in spans:
            # Calculate EWMA for the historical series
            ewma_series = product_data.ewm(span=span).mean()
            ewma_df.loc[product] = ewma_series
            
            # Generate forecasts
            last_value = ewma_series.iloc[-1]
            forecast = [last_value] * future_periods
            ewma_results[f'EWMA_{span}'] = np.array(forecast)
        
        ewma_forecasts[product] = ewma_results
    
    print(f"\n=== APPLIED EWMA TO {len(ewma_forecasts)} PRODUCTS ===")
    return ewma_df, ewma_forecasts

def generate_forecasts(historical_df, future_periods=4, eda_results=None, feature_pools=None):
    """Generate forecasts using multiple methods"""
    print(f"\n=== GENERATING FORECASTS FOR {future_periods} QUARTERS AHEAD ===")
    forecasts = {}
    
    # Apply GOLD outlier detection and cleaning
    cleaned_df = apply_gold_outlier_detection(historical_df)
    
    # Apply EWMA
    _, ewma_forecasts = apply_ewma(cleaned_df)
    
    for product in cleaned_df.index:
        product_data = cleaned_df.loc[product].dropna()
        
        if len(product_data) < 4:  # Need minimum data for forecasting
            continue
        
        # Convert to numpy array
        data = product_data.values
        
        # Store forecasts for this product
        product_forecasts = {}
        
        # 1. Simple Moving Average
        try:
            window = min(3, len(data))
            sma_forecast = np.mean(data[-window:]) * np.ones(future_periods)
            product_forecasts['Moving Average'] = sma_forecast
        except Exception:
            pass
        
        # 2. Exponential Smoothing
        try:
            model = ExponentialSmoothing(data, trend='add', seasonal=None)
            fitted_model = model.fit()
            ets_forecast = fitted_model.forecast(future_periods)
            product_forecasts['Exponential Smoothing'] = ets_forecast
        except Exception:
            pass
        
        # 3. ARIMA
        try:
            # Use seasonality information from EDA if available
            seasonal_order = (0, 0, 0, 0)
            if eda_results and product in eda_results:
                if eda_results[product]['has_seasonality']:
                    seasonal_order = (1, 0, 1, 4)  # Simple seasonal ARIMA component
            
            # Select order based on trend
            if eda_results and product in eda_results:
                if eda_results[product]['trend'] == 'increasing':
                    order = (1, 1, 0)  # More differencing for upward trend
                elif eda_results[product]['trend'] == 'decreasing':
                    order = (1, 1, 0)  # More differencing for downward trend
                else:
                    order = (1, 0, 0)  # Less differencing for stable
            else:
                order = (1, 1, 0)  # Default
            
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            arima_forecast = fitted_model.forecast(future_periods)
            product_forecasts['ARIMA'] = arima_forecast
        except Exception:
            pass
        
        # 4. Prophet
        try:
            # Create dummy dates for Prophet
            dates = pd.date_range(start='2020-01-01', periods=len(data), freq='QE')
            prophet_df = pd.DataFrame({'ds': dates, 'y': data})
            
            # Configure Prophet model based on EDA
            if eda_results and product in eda_results:
                seasonality_mode = 'additive'
                yearly_seasonality = eda_results[product]['has_seasonality']
            else:
                seasonality_mode = 'additive'
                yearly_seasonality = 'auto'
            
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode=seasonality_mode
            )
            
            # Add GenAI growth component for server products
            if 'server' in product.lower() or 'network' in product.lower():
                model.add_regressor('genai_growth')
                prophet_df['genai_growth'] = np.linspace(1, 2, len(data))  # Linear growth factor
                
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=future_periods, freq='QE')
            
            # Add regressor values for future periods
            if 'server' in product.lower() or 'network' in product.lower():
                future['genai_growth'] = np.linspace(2, 3, len(future))  # Continued growth
            
            forecast = model.predict(future)
            prophet_forecast = forecast['yhat'].values[-future_periods:]
            
            product_forecasts['Prophet'] = prophet_forecast
        except Exception:
            pass
        
        # 5. Random Forest with feature pools
        try:
            X = []
            y = []
            
            # Basic lag features if no feature pools available
            if not feature_pools or product not in feature_pools:
                lag = min(2, len(data) - 1)
                for i in range(lag, len(data)):
                    X.append(data[i-lag:i])
                    y.append(data[i])
            else:
                # Use feature pools for more advanced modeling
                features = feature_pools[product]
                X_dict = {}
                
                # Combine all available features
                for feature_type, feature_group in features.items():
                    for feature_name, feature_values in feature_group.items():
                        if isinstance(feature_values, list):
                            X_dict[feature_name] = feature_values
                        else:
                            X_dict[feature_name] = feature_values.values
                
                # Find minimum length to align all features
                min_length = min([len(v) for v in X_dict.values()]) if X_dict else 0
                
                if min_length > 0:
                    # Create aligned feature matrix
                    feature_names = list(X_dict.keys())
                    for i in range(min_length):
                        feature_row = [X_dict[f][-(min_length-i)] for f in feature_names]
                        X.append(feature_row)
                    
                    # Target values
                    y = data[-(min_length):]
            
            if len(X) > 0 and len(y) > 0:
                X = np.array(X)
                y = np.array(y)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Generate forecasts
                rf_forecast = []
                
                if not feature_pools or product not in feature_pools:
                    # Simple forecasting with lag features
                    last_values = data[-lag:].copy()
                    
                    for _ in range(future_periods):
                        next_pred = model.predict([last_values])[0]
                        rf_forecast.append(next_pred)
                        last_values = np.append(last_values[1:], next_pred)
                else:
                    # More complex forecasting with feature pools
                    # Just using the last feature row repeated for simplicity
                    # In a real model, you would generate future feature values
                    last_feature_row = X[-1].copy()
                    
                    for i in range(future_periods):
                        next_pred = model.predict([last_feature_row])[0]
                        rf_forecast.append(next_pred)
                        
                        # Update feature row for next prediction (simplified)
                        # A real model would update the feature row more intelligently
                        for j, feat in enumerate(feature_names):
                            if 'lag' in feat:
                                last_feature_row[j] = next_pred
                
                product_forecasts['Random Forest'] = np.array(rf_forecast)
        except Exception as e:
            pass
        
        # Add EWMA forecasts if available
        if product in ewma_forecasts:
            for method, forecast in ewma_forecasts[product].items():
                product_forecasts[method] = forecast
        
        # Store all forecasts for this product if we have any valid forecasts
        if product_forecasts:
            forecasts[product] = product_forecasts
    
    return forecasts

def calculate_resemblance_weights(historical_df, external_forecasts):
    """Calculate weights based on resemblance to historical patterns"""
    weights = {}
    
    # Check if external forecasts exist
    if external_forecasts is None or external_forecasts.empty:
        return weights
    
    # Fill missing values with median for each product
    historical_filled = historical_df.copy()
    for product in historical_filled.index:
        product_data = historical_filled.loc[product]
        median = product_data.median()
        historical_filled.loc[product] = product_data.fillna(median)
    
    # For each external forecast column
    for column in external_forecasts.columns:
        weights[column] = {}
        
        for product in historical_df.index:
            if product not in external_forecasts.index:
                continue
                
            # Get historical values for this product
            historical_values = historical_filled.loc[product].values.reshape(1, -1)
            
            # Get external forecast value
            ext_forecast = external_forecasts.loc[product, column]
            
            if pd.isna(ext_forecast):
                continue
            
            # Normalize data
            scaler = StandardScaler()
            historical_scaled = scaler.fit_transform(historical_values.reshape(-1, 1)).reshape(1, -1)
            
            # Calculate "resemblance" based on historical pattern
            # Higher weight for forecast values close to recent historical average
            recent_avg = np.mean(historical_values[0][-3:])
            max_diff = np.max(historical_values) - np.min(historical_values)
            
            if max_diff == 0:  # Avoid division by zero
                similarity = 0.5
            else:
                # Measure how close the external forecast is to recent average
                normalized_diff = abs(ext_forecast - recent_avg) / max_diff
                similarity = max(0, 1 - normalized_diff)
            
            # Store weights
            weights[column][product] = similarity
    
    return weights

def create_ensemble_forecast(forecasts, future_periods=4):
    """
    Create ensemble forecast by combining predictions from multiple methods
    
    Parameters:
    -----------
    forecasts : dict
        Dictionary of forecasts from multiple methods for each product
    future_periods : int
        Number of periods to forecast
        
    Returns:
    --------
    pandas DataFrame
        Ensemble forecasts for each product and future period
    """
    print(f"\n=== CREATING ENSEMBLE FORECASTS ===")
    
    # Create DataFrame to store ensemble forecasts
    ensemble_df = pd.DataFrame(index=forecasts.keys())
    
    # Create column headers for future periods
    forecast_columns = [f"Q+{i+1}" for i in range(future_periods)]
    
    # Initialize DataFrame with NaN values
    for col in forecast_columns:
        ensemble_df[col] = np.nan
    
    # Calculate ensemble forecasts for each product
    for product, product_forecasts in forecasts.items():
        # Count valid forecasts for each period
        for period in range(future_periods):
            valid_forecasts = []
            
            # Collect valid forecasts from each method
            for method, forecast_values in product_forecasts.items():
                if period < len(forecast_values) and not np.isnan(forecast_values[period]):
                    valid_forecasts.append(forecast_values[period])
            
            # Calculate ensemble forecast if we have valid forecasts
            if valid_forecasts:
                ensemble_forecast = np.mean(valid_forecasts)
                ensemble_df.loc[product, forecast_columns[period]] = ensemble_forecast
    
    # Print summary
    print(f"Created ensemble forecasts for {len(ensemble_df)} products across {future_periods} periods")
    return ensemble_df

def integrate_with_external_forecasts(ensemble_forecasts, external_forecasts, alpha=0.5, resemblance_weights=None):
    """
    Integrate ensemble forecasts with external forecasts using a weighted approach
    
    Parameters:
    -----------
    ensemble_forecasts : pandas DataFrame
        Ensemble forecasts for each product and future period
    external_forecasts : pandas DataFrame
        External forecasts from business sources
    alpha : float
        Weight given to external forecasts (0-1)
    resemblance_weights : dict
        Weights based on how well external forecasts resemble historical patterns
        
    Returns:
    --------
    pandas DataFrame
        Integrated forecasts combining ensemble and external forecasts
    """
    print(f"\n=== INTEGRATING FORECASTS (ALPHA={alpha:.4f}) ===")
    
    if external_forecasts is None or external_forecasts.empty:
        print("No external forecasts available. Returning ensemble forecasts.")
        return ensemble_forecasts
    
    # Create DataFrame to store integrated forecasts
    integrated_forecasts = pd.DataFrame(index=ensemble_forecasts.index)
    
    # Process each external forecast column
    for col in external_forecasts.columns:
        integrated_col = pd.Series(index=ensemble_forecasts.index)
        
        for product in ensemble_forecasts.index:
            # Skip if product not in external forecasts
            if product not in external_forecasts.index:
                continue
                
            # Get external forecast value
            ext_value = external_forecasts.loc[product, col]
            
            # Skip if external value is missing
            if pd.isna(ext_value):
                continue
            
            # Get ensemble forecast for first period
            ens_value = ensemble_forecasts.loc[product, ensemble_forecasts.columns[0]]
            
            # Apply resemblance weight if available
            effective_alpha = alpha
            if resemblance_weights and col in resemblance_weights and product in resemblance_weights[col]:
                weight = resemblance_weights[col][product]
                # Adjust alpha based on resemblance
                effective_alpha = alpha * weight
                print(f"Product '{product}': Resemblance weight={weight:.4f}, Effective alpha={effective_alpha:.4f}")
            
            # Calculate integrated forecast using weighted formula
            integrated_value = effective_alpha * ext_value + (1 - effective_alpha) * ens_value
            
            # Store integrated value
            integrated_col[product] = integrated_value
        
        # Add to integrated forecasts DataFrame
        integrated_forecasts[col] = integrated_col
    
    print(f"Successfully integrated {len(integrated_forecasts.columns)} external forecast columns")
    return integrated_forecasts

def main():
    """Main function to run the forecasting tool"""
    print("\n===== CISCO FORECAST TOOL =====")
    
    # Get the Excel file path
    default_file = "a.xlsx"
    file_path = input(f"Enter Excel file path (default: {default_file}): ").strip()
    if not file_path:
        file_path = default_file
    
    # Load data from Excel
    historical_df, external_df, product_names = load_data_from_excel(file_path)
    
    if historical_df is None:
        print("Error: Could not load data. Exiting.")
        return
    
    # Get user inputs
    future_periods = int(input("Enter number of quarters to forecast (default: 4): ") or 4)
    alpha = float(input("Enter weight for external forecasts (0-1, default: 0.5): ") or 0.5)
    
    # Perform Exploratory Data Analysis for seasonal patterns
    eda_results = perform_eda(historical_df)
    
    # Create feature pools for advanced forecasting
    feature_pools = create_feature_pools(historical_df, eda_results)
    
    # Generate forecasts
    forecasts = generate_forecasts(historical_df, future_periods, eda_results, feature_pools)
    
    # Create ensemble forecast
    ensemble_forecasts = create_ensemble_forecast(forecasts, future_periods)
    
    # Integrate with external forecasts if available
    integrated_forecasts = None
    if external_df is not None and not external_df.empty:
        # Calculate resemblance weights
        resemblance_weights = calculate_resemblance_weights(historical_df, external_df)
        
        # Integrate forecasts
        integrated_forecasts = integrate_with_external_forecasts(
            ensemble_forecasts, external_df, alpha, resemblance_weights
        )
    
    # Display results
    print("\n===== RESULTS =====")
    
    # Show historical data
    print("\nHistorical Data:")
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(historical_df)
    
    # Show ensemble forecasts
    if ensemble_forecasts is not None:
        print("\nEnsemble Forecasts:")
        print(ensemble_forecasts)
    
    # Show external forecasts
    if external_df is not None and not external_df.empty:
        print("\nExternal Forecasts:")
        print(external_df)
    
    # Show integrated forecasts
    if integrated_forecasts is not None:
        print("\nIntegrated Forecasts:")
        print(integrated_forecasts)
        
        # If there are multiple external forecasts, create a final combined forecast
        if len(external_df.columns) > 1:
            final_forecast = integrated_forecasts.mean(axis=1)
            print("\nFinal Combined Forecast:")
            print(final_forecast)
            
            # Show comparison for first forecast period
            comparison = pd.DataFrame(index=product_names)
            
            if ensemble_forecasts is not None:
                comparison['Ensemble'] = ensemble_forecasts.iloc[:, 0]
                
            for col in external_df.columns:
                comparison[f'External ({col})'] = external_df[col]
                
            for col in integrated_forecasts.columns:
                comparison[f'Integrated ({col})'] = integrated_forecasts[col]
                
            comparison['Final Combined'] = final_forecast
            
            print("\nComparison (First Forecast Period):")
            print(comparison)
    
    print("\n===== FORECAST COMPLETE =====")
    
    # Save results to a new Excel file if needed
    save_option = input("\nSave results to Excel? (y/n, default: n): ").strip().lower()
    if save_option == 'y':
        output_file = input("Enter output file name (default: forecast_results.xlsx): ").strip()
        if not output_file:
            output_file = "forecast_results.xlsx"
        
        # Create a writer to save to Excel
        with pd.ExcelWriter(output_file) as writer:
            historical_df.to_excel(writer, sheet_name='Historical Data')
            ensemble_forecasts.to_excel(writer, sheet_name='Ensemble Forecasts')
            
            if external_df is not None and not external_df.empty:
                external_df.to_excel(writer, sheet_name='External Forecasts')
            
            if integrated_forecasts is not None:
                integrated_forecasts.to_excel(writer, sheet_name='Integrated Forecasts')
                
                if len(external_df.columns) > 1:
                    pd.DataFrame(final_forecast, columns=['Final Forecast']).to_excel(writer, sheet_name='Final Combined')
                    comparison.to_excel(writer, sheet_name='Comparison')
        
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 