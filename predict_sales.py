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
# TRAINING FUNCTION (for endpoints and scripts)
# ============================================================

def train_and_save_model(csv_path="sales_data.csv"):
    """
    Train the Decision Tree model on all available data and save artifacts.

    This function mirrors the original module-level training code but
    exposes it as a callable function so that a web backend (e.g. app.py)
    can trigger re-training on demand.

    Parameters
    ----------
    csv_path : str, default "sales_data.csv"
        Path to the CSV file containing the raw sales data.

    Returns
    -------
    dict
        A small summary with the number of rows and columns used for
        training. The model itself is persisted to disk as
        ``sales_prediction_model.pkl`` and the engineered historical
        data as ``historical_data.pkl``.
    """
    print("Loading and preparing data...")
    data = pd.read_csv(csv_path)
    
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
    return {"rows": int(data.shape[0]), "cols": int(data.shape[1])}


# ============================================================
# PREDICTION FUNCTIONS (unchanged from your original)
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
        "Store ID", "Product ID", "Date", "Units Sold", and engineered features.
    
    Returns:
    --------
    pd.Series
        A row representing the last known data point for the given store and product.
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
    data point for a store-product combination and rolling it forward to the
    target date, updating date-related features, lag values, and rolling metrics.
    
    Parameters:
    -----------
    store_id : str
        Store identifier (e.g., "S001", "S002")
    product_id : str
        Product identifier (e.g., "P0001", "P0002")
    target_date : str or pd.Timestamp
        Target date for prediction. Can be a string in format "YYYY-MM-DD"
        or a pandas Timestamp object (e.g., "2023-01-15")
    historical_data : pd.DataFrame
        DataFrame containing historical sales data with engineered features.
    
    Returns:
    --------
    pd.DataFrame
        A single-row DataFrame containing the feature vector for the target date.
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    last_row = _get_last_known_values(store_id, product_id, historical_data)
    
    # Copy the last row and update date
    feature_row = last_row.copy()
    feature_row["Date"] = target_date
    
    # Update date-based features
    feature_row["DayOfWeek"] = target_date.dayofweek
    feature_row["Month"] = target_date.month
    feature_row["Year"] = target_date.year
    feature_row["DaysSinceStart"] = (target_date - historical_data["Date"].min()).days
    
    # Update lag features by shifting from historical data
    store_product_data = historical_data[
        (historical_data["Store ID"] == store_id) &
        (historical_data["Product ID"] == product_id)
    ].sort_values("Date")
    
    # Compute new lag values
    recent_sales = store_product_data["Units Sold"].values
    feature_row["Lag_1"] = recent_sales[-1] if len(recent_sales) >= 1 else 0
    feature_row["Lag_7"] = recent_sales[-7] if len(recent_sales) >= 7 else feature_row["Lag_1"]
    feature_row["Lag_14"] = recent_sales[-14] if len(recent_sales) >= 14 else feature_row["Lag_7"]
    feature_row["Lag_30"] = recent_sales[-30] if len(recent_sales) >= 30 else feature_row["Lag_14"]
    
    # Update rolling features using available historical data
    rolling_values = recent_sales.tolist()
    price_values = store_product_data["Price"].values.tolist()
    inventory_values = store_product_data["Inventory Level"].values.tolist()
    
    # Update rolling statistics
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
    
    # Remove Units Sold and Date (target for prediction, not a feature)
    feature_row = feature_row.drop(["Units Sold", "Date"])
    
    # Convert to DataFrame
    feature_df = feature_row.to_frame().T
    return feature_df


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
        Predicted units sold for the specified date.
    """
    feature_df = _create_features_for_date(store_id, product_id, target_date, historical_data)
    prediction = model.predict(feature_df)[0]
    return max(0.0, float(prediction))


def predict_sales_for_week(store_id, product_id, week_start_date, model, historical_data):
    """
    Predict sales for a full week starting from a given date.
    
    This function generates daily predictions for 7 consecutive days starting
    from the specified week_start_date.
    
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
        Dictionary where keys are dates (format: "YYYY-MM-DD") and values are
        predicted units sold for each day of the week.
    """
    if isinstance(week_start_date, str):
        week_start_date = pd.to_datetime(week_start_date)
    
    predictions = {}
    for i in range(7):
        current_date = week_start_date + pd.Timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        predictions[date_str] = predict_sales_for_date(
            store_id, product_id, current_date, model, historical_data
        )
    
    return predictions


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