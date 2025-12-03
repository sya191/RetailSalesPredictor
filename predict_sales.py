import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

# Training functions for endpoints and scripts
def train_and_save_model(csv_path="sales_data.csv"):
    # training hte decision tree on the full dataset and saving the results
    # returns a small summary of the data used, the traied model and enginerred
    # features will be saved to disk
    print("Loading and preparing data...")
    data = pd.read_csv(csv_path)
    
    # Drop leakage columns
    leakage_cols = ["Demand", "Units Ordered"]
    data = data.drop(columns=[col for col in leakage_cols if col in data.columns])
    
    # Convert date to datetime
    data["Date"] = pd.to_datetime(data["Date"])
    
    # Sort by date for proper time series features
    data = data.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    
    # Convert epidemic to categorical feature
    data["Epidemic"] = data["Epidemic"].astype("category")
    
    # Extract date features
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["Year"] = data["Date"].dt.year
    data["DaysSinceStart"] = (data["Date"] - data["Date"].min()).dt.days
    data["IsHoliday"] = data["Date"].apply(_is_holiday)
    data["IsWeekend"] = data["Date"].dt.dayofweek >= 5  # Saturday=5, Sunday=6
    
    # Competitor Pricing Features
    if "Competitor Pricing" in data.columns:
        # Fill missing competitor prices safely
        data["Competitor Pricing"] = (
            data["Competitor Pricing"]
            .fillna(method="ffill")
            .fillna(method="bfill")
        )

        # Price difference & price ratio
        data["PriceDiff"] = data["Price"] - data["Competitor Pricing"]
        data["PriceRatio"] = data["Price"] / data["Competitor Pricing"]
        
    
    # creating lag +rolling features
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
    
    #preparing the features and setting the target    
    y = data["Units Sold"]
    X = data.drop(columns=["Units Sold", "Date"])
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    
    
    #train the model on all data     
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
    print("Numeric:", numeric_cols)
    print("Categorical:", categorical_cols)
    return {"rows": int(data.shape[0]), "cols": int(data.shape[1])}


# Prediction functions


# Add holiday feature
def _is_holiday(date):
    """Check if a date is a common holiday"""
    month = date.month
    day = date.day
    year = date.year
    
    # Fixed date holidays
    holidays = [
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas
        (12, 31), # New Year's Eve
    ]
    
    if (month, day) in holidays:
        return 1
    
    # Thanksgiving (4th Thursday of November)
    if month == 11:
        first_day = pd.Timestamp(year, 11, 1).dayofweek
        days_until_thursday = (3 - first_day) % 7
        if days_until_thursday == 0:
            days_until_thursday = 7
        thanksgiving = 1 + days_until_thursday + 21
        if day == thanksgiving:
            return 1
    
    # Black Friday (day after Thanksgiving)
    if month == 11:
        first_day = pd.Timestamp(year, 11, 1).dayofweek
        days_until_thursday = (3 - first_day) % 7
        if days_until_thursday == 0:
            days_until_thursday = 7
        thanksgiving = 1 + days_until_thursday + 21
        if day == thanksgiving + 1:
            return 1
    
    # Labor Day (first Monday of September)
    if month == 9 and day <= 7:
        first_day = pd.Timestamp(year, 9, 1).dayofweek
        if first_day == 0:
            return 1 if day == 1 else 0
        days_until_monday = (7 - first_day) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        labor_day = 1 + days_until_monday - 1
        if day == labor_day:
            return 1
    
    return 0


def _create_features_for_date(store_id, product_id, target_date, historical_data):
    # Convert string date to datetime
    target_date = pd.to_datetime(target_date)

    # Filter historical rows for the specific store + product
    sp = historical_data[
        (historical_data["Store ID"] == store_id) &
        (historical_data["Product ID"] == product_id)
    ].sort_values("Date")

    if sp.empty:
        raise ValueError(f"No data for Store {store_id}, Product {product_id}")

    # CASE A:  Date exists in historical data -> return REAL feature row
    real_row = sp[sp["Date"] == target_date]
    if not real_row.empty:
        return real_row.drop(columns=["Units Sold", "Date"]).reset_index(drop=True)

    # CASE B: Future prediction -> build synthetic row from last known datapoint
    last = sp.iloc[-1].copy()

    feature_row = last.copy()
    feature_row["Date"] = target_date

    # Update date features
    feature_row["DayOfWeek"] = target_date.dayofweek
    feature_row["Month"] = target_date.month
    feature_row["Year"] = target_date.year
    feature_row["DaysSinceStart"] = (target_date - historical_data["Date"].min()).days
    feature_row["IsHoliday"] = _is_holiday(target_date)
    feature_row["IsWeekend"] = target_date.dayofweek >= 5

    # Build updated rolling history (sales, price, inventory)
    sales_hist = sp["Units Sold"].tolist()
    price_hist = sp["Price"].tolist()
    inv_hist = sp["Inventory Level"].tolist()

    # Lags
    feature_row["Lag_1"] = sales_hist[-1]
    feature_row["Lag_7"] = sales_hist[-7] if len(sales_hist) >= 7 else sales_hist[-1]
    feature_row["Lag_14"] = sales_hist[-14] if len(sales_hist) >= 14 else feature_row["Lag_7"]
    feature_row["Lag_30"] = sales_hist[-30] if len(sales_hist) >= 30 else feature_row["Lag_14"]

    # Rolling windows
    for w in [7, 14, 30]:
        window_vals = sales_hist[-w:] if len(sales_hist) >= w else sales_hist
        feature_row[f"Rolling_{w}_mean"] = float(np.mean(window_vals))
        feature_row[f"Rolling_{w}_std"] = float(np.std(window_vals)) if len(window_vals) > 1 else 0.0

    # Price rolling
    feature_row["Rolling_Price_7"] = float(np.mean(price_hist[-7:]))
    feature_row["PriceChange_7"] = feature_row["Price"] - feature_row["Rolling_Price_7"]

    # Competitor pricing features
    if "Competitor Pricing" in sp.columns:
        comp = float(last["Competitor Pricing"])
        feature_row["PriceDiff"] = feature_row["Price"] - comp
        feature_row["PriceRatio"] = feature_row["Price"] / comp if comp != 0 else 1.0

    # Inventory rolling
    feature_row["Rolling_Inventory_7"] = float(np.mean(inv_hist[-7:]))
    feature_row["InventoryChange_7"] = feature_row["Inventory Level"] - feature_row["Rolling_Inventory_7"]

    # Drop target + date
    feature_row = feature_row.drop(["Units Sold", "Date"])

    return feature_row.to_frame().T

def predict_sales_for_date(store_id, product_id, target_date, model, historical_data):
    
    feature_df = _create_features_for_date(store_id, product_id, target_date, historical_data)
    pred = model.predict(feature_df)[0]
    return max(0.0, float(pred))


def predict_sales_for_week(store_id, product_id, week_start_date, model, historical_data):
    
    # Predict sales for a full week recursively.
    # Where each day's prediction is appended back into the historical_data
    # So that lag features update properly for the next day.
    if isinstance(week_start_date, str):
        week_start_date = pd.to_datetime(week_start_date)

    # creating a copy so we don't overwrite the historical data
    hist = historical_data.copy()
    
    # Filters the store+product to speed things up
    store_product_hist = hist[
        (hist["Store ID"] == store_id) &
        (hist["Product ID"] == product_id)
    ].sort_values("Date").copy()

    if store_product_hist.empty:
        raise ValueError(f"No historical data for {store_id}, {product_id}")

    predictions = {}

    # Extract static features that don't change in the future (price, inventory, etc.)
    last_known_price = float(store_product_hist["Price"].iloc[-1])
    last_known_inventory = float(store_product_hist["Inventory Level"].iloc[-1])

    for i in range(7):
        current_date = week_start_date + pd.Timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")

        # Recursively create feature row for this day 
        feature_row = _create_features_for_date(store_id, product_id, current_date, hist)

        # prediction
        pred = model.predict(feature_row)[0]
        pred = max(0.0, float(pred))
        predictions[date_str] = pred

        # Append prediction to history
        new_row = {
            "Store ID": store_id,
            "Product ID": product_id,
            "Date": current_date,
            "Units Sold": pred,
            "Price": last_known_price,
            "Inventory Level": last_known_inventory,
        }

        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

    return predictions


def load_model_and_data():
    # Loads the train model and historical data from disk
    with open("sales_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    historical_data = pd.read_pickle("historical_data.pkl")
    #Returns the saved prediction pipeline and the historical DataFrame,
    return model, historical_data


def predict_sales(store_id, product_id, date_or_week, is_week=False):

    #Main prediction function for front-end integration.
    #Load the trained model and historical data from disk.  
    #Returns the saved prediction pipeline and the historical DataFrame based on user input on weeks/days
    model, historical_data = load_model_and_data()
    
    if is_week:
        return predict_sales_for_week(store_id, product_id, date_or_week, model, historical_data)
    else:
        return predict_sales_for_date(store_id, product_id, date_or_week, model, historical_data)
    

def predict_and_recommend(
    store_id,
    product_id,
    date_str,
    existing_stock,
    is_week=False,
    safety_stock=0,
    incoming_stock=0
):
    # Make a daily or weekly sales prediction and compute the recommended order
    # Quantity, factoring in existing, incoming, and safety stock
    # Returns either a single-day prediction or a 7-day forecast along with the suggested order amount.
    # Load model + historical data
    model, historical_data = load_model_and_data()


    # weekly prediction 
    if is_week:
        week_pred = predict_sales_for_week(store_id, product_id, date_str, model, historical_data)

        total_predicted_demand = sum(week_pred.values())
        inventory_position = existing_stock + incoming_stock

        order_qty = max(total_predicted_demand + safety_stock - inventory_position, 0.0)

        return {
            "success": True,
            "prediction_type": "week",
            "store_id": store_id,
            "product_id": product_id,
            "predictions": week_pred,
            "total_predicted_demand": total_predicted_demand,
            "recommended_order_qty": order_qty,
        }

    # single day prediction
    else:
        pred = predict_sales_for_date(store_id, product_id, date_str, model, historical_data)

        inventory_position = existing_stock + incoming_stock
        order_qty = max(pred + safety_stock - inventory_position, 0.0)

        return {
            "success": True,
            "prediction_type": "day",
            "store_id": store_id,
            "product_id": product_id,
            "date": date_str,
            "units_predicted": pred,
            "recommended_order_qty": order_qty,
        }