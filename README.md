# Cisco Sales Forecasting

## Overview
This project was developed as part of the **Cisco Forecast League** competition, where we built an **ensemble-based sales forecasting system** that integrates statistical, machine learning, and time-series models to predict future demand across multiple product categories.

Our model achieved an **accuracy of 90.46%** and ranked **7th among RV institutions (RVCE, RVU, and RVITM)**.

---

## Features
- **Multi-Model Forecasting Pipeline**
  - Combines **ARIMA**, **Prophet**, **Exponential Smoothing**, **Random Forest Regressor**, and **EWMA**.
- **Advanced Feature Engineering**
  - Incorporates lag variables, rolling statistics, macroeconomic indicators, and GenAI adoption trends.
- **Outlier Detection (GOLD)**
  - Uses **Isolation Forest** and **Z-score** methods to identify and replace anomalies.
- **Ensemble Forecasting**
  - Averages predictions from multiple models for improved accuracy and stability.
- **External Data Integration**
  - Blends forecasts with external business data using **resemblance-weighted ensemble integration**.
- **Automatic EDA**
  - Detects seasonality, trends, and outliers across time-series data.

---

## Methodology

### 1. Data Preprocessing
- Extracts historical and forecast data from Excel (`a.xlsx`)
- Cleans missing values and applies outlier detection (GOLD method)

### 2. Exploratory Data Analysis (EDA)
- Detects seasonality and trends using autocorrelation and regression
- Identifies anomalies using Z-scores

### 3. Feature Engineering
- Lag features
- Rolling mean and standard deviation
- Macroeconomic indicators (GDP, Inflation, Interest Rate, Tech Spending)
- Seasonal dummy variables
- GenAI Adoption Index (for tech-related products)

### 4. Model Ensemble
- Trains multiple forecasting models: **ARIMA, Prophet, Exponential Smoothing, Random Forest, EWMA**
- Combines results through mean ensemble to reduce model bias

### 5. Integration with External Forecasts
- Calculates resemblance weights using **cosine similarity** and statistical closeness
- Applies weighted blending with ensemble outputs to generate the final forecast

---

## Tech Stack
- **Language:** Python 3.x  
- **Libraries:**  
  `pandas`, `numpy`, `statsmodels`, `prophet`, `scikit-learn`, `matplotlib`, `scipy`  
- **Forecasting Models:** ARIMA, Prophet, Exponential Smoothing, EWMA, Random Forest  

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/venkateshraju04/cisco-forecast-league.git
cd cisco-forecast-league
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the model
```
python cisco_forecast_tool.py
```
