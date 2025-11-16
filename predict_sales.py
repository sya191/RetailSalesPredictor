"""
Sales Prediction Module

This module provides functionality for training and using a Decision Tree model
to predict retail sales. It includes feature engineering with lag and rolling
statistics, model training, and prediction interfaces for both single dates and
weekly forecasts.

The module automatically trains a model on all available data when imported,
saving the trained model and historical data for use in predictions.

Public API:
----------
- predict_sales(store_id, product_id, date_or_week, is_week=False)
  Main interface for making predictions. Automatically loads model and data.
  
- predict_sales_for_date(store_id, product_id, target_date, model, historical_data)
  Predict sales for a single date. Requires pre-loaded model and data.
  
- predict_sales_for_week(store_id, product_id, week_start_date, model, historical_data)
  Predict sales for a full week (7 days). Requires pre-loaded model and data.
  
- load_model_and_data()
  Load the trained model and historical data from saved files.

Private Functions:
------------------
- _get_last_known_values(store_id, product_id, historical_data)
  Internal helper to retrieve last known values for a store-product combination.
  
- _create_features_for_date(store_id, product_id, target_date, historical_data)
  Internal helper to create feature vector for a specific date.

Example Usage:
--------------
    # Simple usage (recommended for front-end)
    from predict_sales import predict_sales
    
    # Predict for a single date
    prediction = predict_sales("S001", "P0001", "2023-01-15", is_week=False)
    
    # Predict for a week
    week_predictions = predict_sales("S001", "P0001", "2023-01-15", is_week=True)
    
    # Advanced usage (for better performance)
    from predict_sales import load_model_and_data, predict_sales_for_date
    model, historical_data = load_model_and_data()
    prediction = predict_sales_for_date("S001", "P0001", "2023-01-15", model, historical_data)

Files Generated:
----------------
- sales_prediction_model.pkl: Trained scikit-learn pipeline
- historical_data.pkl: Historical sales data for feature creation

Author: Sales Prediction System
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# LOAD + PREPARE DATA
# ============================================================

print("Loading and preparing data...")
data = pd.read_csv("sales_data.csv")

# Drop leakage columns
leakage_cols = ["Demand", "Units Ordered"]
data = data.drop(columns=[col for col in leakage_cols if col in data.columns])

# Convert date to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Sort by date for proper time series features
data = data.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

# Extract date features
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["Month"] = data["Date"].dt.month
data["Year"] = data["Date"].dt.year
data["DaysSinceStart"] = (data["Date"] - data["Date"].min()).dt.days


# ============================================================
# CREATE LAG + ROLLING FEATURES
# ============================================================

print("Creating lag and rolling features...")

# Lag features
data["Lag_1"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(1)
data["Lag_7"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(7)
data["Lag_14"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(14)
data["Lag_30"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(30)

# Rolling statistics
for window in [7, 14, 30]:
    data[f"Rolling_{window}_mean"] = (
        data.groupby(["Store ID", "Product ID"])["Units Sold"]
            .shift(1).rolling(window, min_periods=1).mean()
    )
    data[f"Rolling_{window}_std"] = (
        data.groupby(["Store ID", "Product ID"])["Units Sold"]
            .shift(1).rolling(window, min_periods=1).std()
    )

# Rolling price features
data["Rolling_Price_7"] = (
    data.groupby(["Store ID", "Product ID"])["Price"]
        .shift(1).rolling(7, min_periods=1).mean()
)
data["PriceChange_7"] = data["Price"] - data["Rolling_Price_7"]

# Rolling inventory features
data["Rolling_Inventory_7"] = (
    data.groupby(["Store ID", "Product ID"])["Inventory Level"]
        .shift(1).rolling(7, min_periods=1).mean()
)
data["InventoryChange_7"] = data["Inventory Level"] - data["Rolling_Inventory_7"]

# Fill NaN values
for col in data.columns:
    if 'Rolling' in col or 'Lag' in col:
        data[col] = data.groupby(["Store ID", "Product ID"])[col].ffill()
        data[col] = data[col].fillna(0)

# Drop rows where Lag_1 is missing
data = data.dropna(subset=["Lag_1"])

# Fill any remaining NaN
numeric_cols_before = data.select_dtypes(include=[np.number]).columns
data[numeric_cols_before] = data[numeric_cols_before].fillna(0)

print(f"Data prepared: {data.shape[0]} rows, {data.shape[1]} columns")


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

y = data["Units Sold"]
X = data.drop(columns=["Units Sold", "Date"])

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")


# ============================================================
# TRAIN MODEL ON ALL DATA
# ============================================================

print("\nTraining Decision Tree model on all data...")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder='drop'
)

# Decision Tree model
dt_model = DecisionTreeRegressor(
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", dt_model)
])

# Train on all data
pipeline.fit(X, y)

print("Model trained successfully!")

# Save the model
with open("sales_prediction_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Save the data for reference (needed for creating lag features for future predictions)
data.to_pickle("historical_data.pkl")

print("Model and historical data saved!")


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def _get_last_known_values(store_id, product_id, historical_data):
    """
    Private helper function to get the last known values for a store-product combination.
    
    This function retrieves the most recent historical data point for a specific
    store and product combination, which is used as a baseline for creating
    features for future predictions.
    
    Parameters:
    -----------
    store_id : str
        Store identifier (e.g., "S001", "S002")
    product_id : str
        Product identifier (e.g., "P0001", "P0002")
    historical_data : pd.DataFrame
        DataFrame containing historical sales data with columns including
        "Store ID", "Product ID", "Date", and other features
    
    Returns:
    --------
    pd.Series
        Series containing the last known values for all features for the
        specified store-product combination
    
    Raises:
    -------
    ValueError
        If no historical data is found for the given store_id and product_id
    
    Examples:
    ---------
    >>> last_row = _get_last_known_values("S001", "P0001", historical_data)
    >>> print(last_row["Units Sold"])
    102.0
    """
    store_product_data = historical_data[
        (historical_data["Store ID"] == store_id) & 
        (historical_data["Product ID"] == product_id)
    ].sort_values("Date")
    
    if len(store_product_data) == 0:
        raise ValueError(f"No historical data found for Store {store_id}, Product {product_id}")
    
    last_row = store_product_data.iloc[-1]
    return last_row


def _create_features_for_date(store_id, product_id, target_date, historical_data):
    """
    Private helper function to create a feature vector for a specific date.
    
    This function creates a feature row for prediction by taking the last known
    values for a store-product combination and updating date-related features
    to match the target date. Other features (price, inventory, etc.) are
    kept at their last known values.
    
    Parameters:
    -----------
    store_id : str
        Store identifier (e.g., "S001")
    product_id : str
        Product identifier (e.g., "P0001")
    target_date : str or pd.Timestamp
        Target date for prediction (e.g., "2023-01-15")
    historical_data : pd.DataFrame
        DataFrame containing historical sales data
    
    Returns:
    --------
    pd.Series
        Series containing feature values for the target date, with "Units Sold"
        removed (as it's the target variable)
    
    Notes:
    ------
    - Date-related features (DayOfWeek, Month, Year, DaysSinceStart) are
      updated based on the target_date
    - Other features (Price, Inventory Level, etc.) use the last known values
    - The "Units Sold" column is removed as it's the target variable
    
    Examples:
    ---------
    >>> features = _create_features_for_date("S001", "P0001", "2023-01-15", historical_data)
    >>> print(features["DayOfWeek"])
    6  # Sunday
    """
    target_date = pd.to_datetime(target_date)
    
    # Get last known values
    last_row = _get_last_known_values(store_id, product_id, historical_data)
    
    # Create feature row
    feature_row = last_row.copy()
    
    # Update date-related features
    feature_row["Date"] = target_date
    feature_row["DayOfWeek"] = target_date.dayofweek
    feature_row["Month"] = target_date.month
    feature_row["Year"] = target_date.year
    feature_row["DaysSinceStart"] = (target_date - historical_data["Date"].min()).days
    
    # For future dates, we'll use the last known values for other features
    # In a real scenario, you might want to allow users to input these
    # For now, we'll use the last known values
    
    # Remove Units Sold (target) and Date (will be dropped later)
    feature_row = feature_row.drop(["Units Sold"])
    
    return feature_row


def predict_sales_for_date(store_id, product_id, target_date, model, historical_data):
    """
    Predict sales for a specific store, product, and date.
    
    This function creates features for a single target date and uses the trained
    model to predict the number of units that will be sold. The prediction is
    based on historical patterns, lag features, and rolling statistics.
    
    Parameters:
    -----------
    store_id : str
        Store identifier (e.g., "S001", "S002")
    product_id : str
        Product identifier (e.g., "P0001", "P0002")
    target_date : str or pd.Timestamp
        Target date for prediction. Can be a string in format "YYYY-MM-DD"
        or a pandas Timestamp object (e.g., "2023-01-15")
    model : sklearn.pipeline.Pipeline
        Trained scikit-learn pipeline containing preprocessor and model
    historical_data : pd.DataFrame
        DataFrame containing historical sales data needed to create lag and
        rolling features
    
    Returns:
    --------
    float
        Predicted number of units sold. Always non-negative (minimum value is 0).
    
    Raises:
    -------
    ValueError
        If no historical data is found for the given store_id and product_id
    
    Examples:
    ---------
    >>> model, historical_data = load_model_and_data()
    >>> prediction = predict_sales_for_date("S001", "P0001", "2023-01-15", model, historical_data)
    >>> print(f"Predicted sales: {prediction:.2f} units")
    Predicted sales: 96.21 units
    
    Notes:
    ------
    - The function uses the last known values for features like Price and
      Inventory Level. For more accurate predictions, consider updating these
      values if they are known for the target date.
    - Lag features are based on historical sales data.
    - Predictions are clipped to be non-negative.
    """
    # Create features for the target date
    feature_row = _create_features_for_date(store_id, product_id, target_date, historical_data)
    
    # Convert to DataFrame (single row)
    feature_df = pd.DataFrame([feature_row])
    
    # Drop Date column
    feature_df = feature_df.drop(columns=["Date"], errors='ignore')
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    
    # Ensure non-negative
    return max(0, prediction)


def predict_sales_for_week(store_id, product_id, week_start_date, model, historical_data):
    """
    Predict sales for a full week (7 days) starting from week_start_date.
    
    This function predicts sales for each day of a week, updating lag and rolling
    features as predictions are made. This allows each day's prediction to be
    influenced by the previous day's predicted sales, creating a more realistic
    multi-step forecast.
    
    Parameters:
    -----------
    store_id : str
        Store identifier (e.g., "S001", "S002")
    product_id : str
        Product identifier (e.g., "P0001", "P0002")
    week_start_date : str or pd.Timestamp
        Starting date of the week. Can be a string in format "YYYY-MM-DD"
        or a pandas Timestamp object (e.g., "2023-01-15")
    model : sklearn.pipeline.Pipeline
        Trained scikit-learn pipeline containing preprocessor and model
    historical_data : pd.DataFrame
        DataFrame containing historical sales data needed to create lag and
        rolling features
    
    Returns:
    --------
    dict
        Dictionary with dates as keys (format: "YYYY-MM-DD") and predicted
        units sold as values (floats). Contains 7 entries, one for each day
        of the week.
    
    Raises:
    -------
    ValueError
        If no historical data is found for the given store_id and product_id
    
    Examples:
    ---------
    >>> model, historical_data = load_model_and_data()
    >>> week_predictions = predict_sales_for_week("S001", "P0001", "2023-01-15", model, historical_data)
    >>> for date, pred in week_predictions.items():
    ...     print(f"{date}: {pred} units")
    2023-01-15: 85.35 units
    2023-01-16: 85.35 units
    ...
    
    Notes:
    ------
    - The function uses a recursive prediction approach: each day's prediction
      becomes the Lag_1 feature for the next day.
    - Rolling features (7-day, 14-day, 30-day averages) are updated as
      predictions are made.
    - This creates a more realistic forecast where predictions influence
      future predictions, but may accumulate error over longer horizons.
    - All predictions are clipped to be non-negative.
    """
    week_start = pd.to_datetime(week_start_date)
    predictions = {}
    
    # Get last known values to start with
    last_row = _get_last_known_values(store_id, product_id, historical_data)
    
    # Initialize lag values from last known data
    lag_1 = last_row.get("Lag_1", 0)
    lag_7 = last_row.get("Lag_7", 0)
    lag_14 = last_row.get("Lag_14", 0)
    lag_30 = last_row.get("Lag_30", 0)
    
    # Initialize rolling window with last known sales
    store_product_data = historical_data[
        (historical_data["Store ID"] == store_id) & 
        (historical_data["Product ID"] == product_id)
    ].sort_values("Date")
    
    # Get last 30 sales for rolling features
    rolling_values = list(store_product_data["Units Sold"].tail(30).values) if len(store_product_data) > 0 else [0] * 30
    price_values = list(store_product_data["Price"].tail(7).values) if len(store_product_data) > 0 else [last_row.get("Price", 0)] * 7
    inventory_values = list(store_product_data["Inventory Level"].tail(7).values) if len(store_product_data) > 0 else [last_row.get("Inventory Level", 0)] * 7
    
    # Predict each day of the week
    for day in range(7):
        current_date = week_start + pd.Timedelta(days=day)
        
        # Create feature row
        feature_row = last_row.copy()
        feature_row["Date"] = current_date
        feature_row["DayOfWeek"] = current_date.dayofweek
        feature_row["Month"] = current_date.month
        feature_row["Year"] = current_date.year
        feature_row["DaysSinceStart"] = (current_date - historical_data["Date"].min()).days
        
        # Update lag features
        feature_row["Lag_1"] = lag_1
        feature_row["Lag_7"] = lag_7
        feature_row["Lag_14"] = lag_14
        feature_row["Lag_30"] = lag_30
        
        # Update rolling features
        for window in [7, 14, 30]:
            window_vals = rolling_values[-window:] if len(rolling_values) >= window else rolling_values
            feature_row[f"Rolling_{window}_mean"] = np.mean(window_vals) if len(window_vals) > 0 else 0
            feature_row[f"Rolling_{window}_std"] = np.std(window_vals) if len(window_vals) > 1 else 0
        
        # Update price rolling
        feature_row["Rolling_Price_7"] = np.mean(price_values[-7:])
        feature_row["PriceChange_7"] = feature_row["Price"] - feature_row["Rolling_Price_7"]
        
        # Update inventory rolling
        feature_row["Rolling_Inventory_7"] = np.mean(inventory_values[-7:])
        feature_row["InventoryChange_7"] = feature_row["Inventory Level"] - feature_row["Rolling_Inventory_7"]
        
        # Remove Units Sold and Date
        feature_row = feature_row.drop(["Units Sold", "Date"])
        feature_df = pd.DataFrame([feature_row])
        
        # Make prediction
        pred = model.predict(feature_df)[0]
        pred = max(0, pred)  # Ensure non-negative
        
        predictions[current_date.strftime("%Y-%m-%d")] = round(pred, 2)
        
        # Update lag values for next iteration
        lag_30 = lag_14
        lag_14 = lag_7
        lag_7 = lag_1
        lag_1 = pred
        
        # Update rolling windows
        rolling_values.append(pred)
        rolling_values = rolling_values[-30:]  # Keep last 30
        price_values.append(feature_row["Price"])
        price_values = price_values[-7:]
        inventory_values.append(feature_row["Inventory Level"])
        inventory_values = inventory_values[-7:]
    
    return predictions


# ============================================================
# HELPER FUNCTIONS FOR FRONT-END
# ============================================================

def load_model_and_data():
    """
    Load the trained model and historical data from saved files.
    
    This function loads the pre-trained Decision Tree model and historical
    data that were saved during model training. These files must exist in
    the current directory:
    - sales_prediction_model.pkl: The trained scikit-learn pipeline
    - historical_data.pkl: The historical sales data DataFrame
    
    Returns:
    --------
    tuple
        A tuple containing:
        - model (sklearn.pipeline.Pipeline): The trained prediction pipeline
        - historical_data (pd.DataFrame): Historical sales data
    
    Raises:
    -------
    FileNotFoundError
        If either "sales_prediction_model.pkl" or "historical_data.pkl"
        cannot be found in the current directory
    
    Examples:
    ---------
    >>> model, historical_data = load_model_and_data()
    >>> print(type(model))
    <class 'sklearn.pipeline.Pipeline'>
    >>> print(historical_data.shape)
    (76000, 32)
    
    Notes:
    ------
    - This function should be called before making predictions if you're
      using the advanced API (predict_sales_for_date or predict_sales_for_week)
    - The simplified API (predict_sales) calls this function automatically
    - Make sure you've run the training script first to generate these files
    """
    with open("sales_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    historical_data = pd.read_pickle("historical_data.pkl")
    return model, historical_data


def predict_sales(store_id, product_id, date_or_week, is_week=False):
    """
    Main prediction function for front-end integration.
    
    This is the primary interface function for making sales predictions. It
    automatically loads the trained model and historical data, then makes
    predictions for either a single date or a full week.
    
    Parameters:
    -----------
    store_id : str
        Store identifier (e.g., "S001", "S002")
    product_id : str
        Product identifier (e.g., "P0001", "P0002")
    date_or_week : str or pd.Timestamp
        Target date for single prediction, or week start date for weekly
        prediction. Can be a string in format "YYYY-MM-DD" or a pandas
        Timestamp object (e.g., "2023-01-15")
    is_week : bool, default=False
        If True, predicts sales for a full week (7 days) starting from
        date_or_week. If False, predicts for a single date.
    
    Returns:
    --------
    float or dict
        - If is_week=False: Returns a float representing predicted units sold
          for the single date
        - If is_week=True: Returns a dictionary with dates as keys (format:
          "YYYY-MM-DD") and predicted units sold as values (floats)
    
    Raises:
    -------
    FileNotFoundError
        If model or historical data files cannot be found
    ValueError
        If no historical data is found for the given store_id and product_id
    
    Examples:
    ---------
    Single date prediction:
    >>> prediction = predict_sales("S001", "P0001", "2023-01-15", is_week=False)
    >>> print(f"Predicted sales: {prediction:.2f} units")
    Predicted sales: 96.21 units
    
    Weekly prediction:
    >>> week_predictions = predict_sales("S001", "P0001", "2023-01-15", is_week=True)
    >>> print(week_predictions)
    {'2023-01-15': 85.35, '2023-01-16': 85.35, ..., '2023-01-21': 104.0}
    
    Notes:
    ------
    - This function automatically loads the model and data, so it's the
      simplest interface for front-end integration
    - For better performance in production, consider loading the model once
      and using predict_sales_for_date or predict_sales_for_week directly
    - All predictions are non-negative (minimum value is 0)
    """
    model, historical_data = load_model_and_data()
    
    if is_week:
        return predict_sales_for_week(store_id, product_id, date_or_week, model, historical_data)
    else:
        return predict_sales_for_date(store_id, product_id, date_or_week, model, historical_data)




